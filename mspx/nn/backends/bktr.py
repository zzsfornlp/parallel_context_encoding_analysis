#

import os.path
from collections import OrderedDict, Counter
from typing import Union

import numpy as np
import torch
from torch.nn import Module
import torch.distributed as dist

from mspx.utils import Conf, zlog, zwarn, WithWrapper, ZHelper
from .common import NIConf

class BKNIConf(NIConf):
    def __init__(self):
        # basic
        self.seed0 = 1377  # seed0 mainly for param init
        self.seed1 = 1377  # seed1 mainly for actual running
        self.num_threads = 4  # maximum NUM_THREADS if using cpu
        self.my_dtype = ''  # by default do nothing
        self.my_device = -1  # which device?
        # compile here?
        self.torch_compile = False  # whether compile at wrapping time
        # fsdp & related
        self.precision = ""  # mixed precision for fsdp: bf16/bf16m/fp16/fp16m/...
        self.use_fsdp = False  # fsdp strategy
        self.fsdp_auto_wrap_keys = []  # module names for auto wrap!
        self.fsdp_shard = "no"  # full/op/no/hybrid
        self.fsdp_cpu_offload = False  # whether CPU-offload
        self.fsdp_ac_wrap_keys = []  # wrap names for AC
        self.fsdp_use_orig_params = False
        self.fsdp_checkpoint_type = "full"  # full/shard
        # --

def init_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    zlog(f"Init torch seed as {seed}")
    # --

def init_seed1():
    init_seed(get_bk_conf().seed1)

BK_my_dtype = torch.get_default_dtype()
BK_my_device = torch.device('cpu')  # default device
BK_default_conf = None

def init(conf: BKNIConf, **kwargs):
    import os
    conf.direct_update(**kwargs)
    torch.set_num_threads(min(conf.num_threads, int(os.environ.get('OMP_NUM_THREADS', 100))))
    init_seed(conf.seed0)  # init seed0 here!
    # --
    LOCAL_RANK, RANK, WORLD_SIZE = get_rank_info()
    if conf.use_fsdp:  # dist.init
        # init the process group
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group("nccl")
        torch.cuda.empty_cache()  # clear gpu cache!
        zlog(f"Init NN.dist with {(LOCAL_RANK, RANK, WORLD_SIZE)}.")
    if WORLD_SIZE > 1:  # mp
        conf.my_device = LOCAL_RANK  # overwrite
    # --
    set_my_dtype(conf.my_dtype)  # no torch.set_default_dtype since some inner parts (like torch.compile) do not support
    set_my_device(conf.my_device)
    zlog(f"Init NN with my_dtype={get_my_dtype()}, my_device={get_my_device()}")
    # --
    global BK_default_conf
    BK_default_conf = conf
    # --

def get_bk_conf(conf=None):
    if conf is None:
        return BK_default_conf
    else:
        return conf

# wrap for model
def wrap_model(model, conf: BKNIConf = None):
    conf: BKNIConf = get_bk_conf(conf)
    ret = model
    # --
    # cast model?
    orig_dtype = get_first_param(ret).dtype
    if conf.precision == "bf16":
        ret.to(torch.bfloat16)
    elif conf.precision == "fp16":
        ret.to(torch.float16)
    new_dtype = get_first_param(ret).dtype
    zlog(f"Cast model dtype (before fsdp) from {orig_dtype} to {new_dtype}!")
    # --
    if conf.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import CPUOffload
        from .fsdp import get_mixed_precision_policy, get_auto_wrapping_policy, get_sharding_strategy, apply_fsdp_checkpointing
        # --
        mixed_precision_policy = get_mixed_precision_policy(conf.precision)
        auto_wrapping_policy = get_auto_wrapping_policy(model, conf.fsdp_auto_wrap_keys)
        sharding_strategy = get_sharding_strategy(conf.fsdp_shard)
        ret = FSDP(
            ret,
            auto_wrap_policy=auto_wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if conf.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=(conf.torch_compile or conf.fsdp_use_orig_params),
            param_init_fn=None,  # TODO(!): enable this to allow meta-lazy-init?
        )
        if conf.fsdp_ac_wrap_keys:
            apply_fsdp_checkpointing(ret, conf.fsdp_ac_wrap_keys)
            setattr_borrow(model, "_use_activation_checkpointing", True)  # note: set to the original model!
        zlog(f"After fsdp wrapping, we have:\n{ret}")
    # --
    if conf.torch_compile:
        ret = torch.compile(ret)
        zlog(f"After model compiling, we have:\n{ret}")
    return ret

def unwrap_model(model, unwrap_compile=True):
    ret = model
    if unwrap_compile:
        from torch._dynamo import OptimizedModule
        if isinstance(ret, OptimizedModule):
            return ret._orig_mod
    return ret

def is_fsdp_model(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return isinstance(model, FSDP)

def setup_deepspeed(conf: BKNIConf = None):
    conf: BKNIConf = get_bk_conf(conf)
    # --
    if "bf16" in conf.precision:
        ret = {"bf16": {"enabled": True}}
    elif "fp16" in conf.precision:
        ret = {"fp16": {"enabled": True}}
    else:
        ret = {}
    return ret

# for fp16-mix
def get_autocast_and_scaler(conf: BKNIConf = None):
    conf: BKNIConf = get_bk_conf(conf)
    # --
    autocast = None
    scaler = None
    _is_fp16m, _is_bf16m = [conf.precision.startswith(z) for z in ["fp16m", "bf16m"]]
    # if _is_fp16m or _is_bf16m:
    if _is_fp16m:  # note: only need these for fp16!
        dtype = torch.float16 if _is_fp16m else torch.bfloat16
        autocast = lambda: my_autocast(dtype=dtype)
        if conf.use_fsdp:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            scaler = ShardedGradScaler()
        else:
            scaler = torch.cuda.amp.GradScaler()
    zlog(f"Obtain autocast={autocast}, scaler={scaler}")
    return autocast, scaler

def clip_gradient_(model, gradient_clipping: float):
    if is_fsdp_model(model):
        model.clip_grad_norm_(gradient_clipping)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

def barrier(timeout=None):
    if timeout is None:  # use the default one
        return dist.barrier()
    else:  # seconds!
        return dist.monitored_barrier(timeout=float(timeout))

as_tensor = torch.as_tensor
arange = torch.arange
concat = torch.concat
stack = torch.stack
ones = torch.ones
ones_like = torch.ones_like
zeros_like = torch.zeros_like
full = torch.full
full_like = torch.full_like
no_grad = torch.no_grad
inference_mode = torch.inference_mode
compile = torch.compile
where = torch.where

rand = torch.rand
randint = torch.randint

# types
float32 = torch.float32
int64 = torch.int64

def set_my_dtype(dtype):
    global BK_my_dtype
    BK_my_dtype = get_my_dtype(dtype)

def get_my_dtype(dtype=None):
    if dtype is None or dtype == "":
        return BK_my_dtype
    elif isinstance(dtype, str):
        return {"float32": torch.float32, "fp32": torch.float32, "float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[dtype]
    else:
        return dtype

def set_my_device(device):
    global BK_my_device
    BK_my_device = get_my_device(device)

def get_my_device(device=""):
    if device is None or device == "":
        return BK_my_device
    else:
        if isinstance(device, str) and str.isdigit(device):
            device = int(device)
        if isinstance(device, int) and device < 0:
            return torch.device('cpu')
        else:
            return torch.device(device)

def get_dtype_max(t):
    return torch.finfo(t.dtype).max

def get_dtype_min(t):
    return torch.finfo(t.dtype).min

def get_shape(t):
    return list(t.shape)

def get_first_param(m):
    return next(m.parameters())

def get_first_device(m):
    return get_first_param(m).device

def my_autocast(dtype=None):
    _device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.autocast(device_type=_device_type, dtype=dtype)

# return numpy values
def get_value(t):
    return t.detach().cpu().numpy()

def get_rank_info(return_str=False, return_local_rank=False, return_rank=False, return_world_size=False):
    import os
    cur_env = os.environ
    LOCAL_RANK, RANK, WORLD_SIZE = int(cur_env.get('LOCAL_RANK', 0)), int(cur_env.get('RANK', 0)), int(cur_env.get('WORLD_SIZE', 1))
    assert sum([int(z) for z in [return_local_rank, return_rank, return_world_size]]) <= 1
    if return_local_rank: return LOCAL_RANK
    if return_rank: return RANK
    if return_world_size: return return_world_size
    if return_str:
        return f"[LOCAL_RANK={LOCAL_RANK}/RANK={RANK}/WORLD_SIZE={WORLD_SIZE}]"
    else:
        return LOCAL_RANK, RANK, WORLD_SIZE

def is_local_main_process():
    return get_rank_info()[0] == 0  # rank0 is the main one!!

def is_main_process():
    return get_rank_info()[1] == 0  # rank0 is the main one!!

# do not go through Model.__setattr__, thus will not add it!
def setattr_borrow(mod, key: str, value: object, assert_nonexist=True):
    if assert_nonexist:
        assert not hasattr(mod, key)
    object.__setattr__(mod, key, value)

# is tensor?
def is_tensor(t):
    return torch.is_tensor(t)

# --
# save and load (v2)

# return saving dir & model-name
def _get_paths(path):
    if os.path.isdir(path):
        return path, os.path.join(path, "model.pt")  # default model name
    else:
        dir_path = os.path.dirname(path)
        return dir_path, path

def try_load_model(model, state_dict, strict):
    if strict is not None:
        model.load_state_dict(state_dict, strict=strict)
    else:  # otherwise, first try strict, then relax if there are errors
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            import traceback
            zwarn(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
            model.load_state_dict(state_dict, strict=False)

def _get_sd_info(sd):
    return f"SD[{len(sd)}]: {list(sd.keys())[:20]} ..."

def _check_optim_loading(optimizer, optim_sd):
    if optimizer is not None and optim_sd is None:
        zwarn("No sd available to restore optimizer!")
    elif optimizer is None and optim_sd is not None:
        zwarn("No optimizer available to assign sd!")

def _get_model_sd(model, filter_keys):
    if not filter_keys:
        return model.state_dict()
    else:
        return {k: v for k, v in model.state_dict().items() if ZHelper.match_name(k, filter_keys)}

def load_my_checkpoint(path: Union[str, dict], model, optimizer=None, conf=None, quiet=False, strict=None):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, FullOptimStateDictConfig
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    import torch.distributed.checkpoint as DCP
    # --
    conf: BKNIConf = get_bk_conf(conf)
    _STR_RANK = get_rank_info(return_str=True)
    if isinstance(path, str):
        path_dir, path_model = _get_paths(path)
        input_sd = None
    else:
        path_dir = path_model = None
        input_sd = path
    model = unwrap_model(model)
    if is_fsdp_model(model):
        dist.barrier()  # barrier!
        if conf.fsdp_checkpoint_type == "full":
            _str_info = f"[{_STR_RANK}] fsdp_full <- {path_model}"
            _policy0 = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            _policy1 = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, _policy0, _policy1):
                state_dict = torch.load(path_model)
                try_load_model(model, state_dict["model"], strict)
                _optim_sd = state_dict.get("optim")
                _check_optim_loading(optimizer, _optim_sd)
                if optimizer is not None and _optim_sd is not None:
                    flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=_optim_sd)
                    optimizer.load_state_dict(flattened_osd)
        elif conf.fsdp_checkpoint_type == "shard":
            _str_info = f"[{_STR_RANK}] fsdp_shard <- {path_dir}"
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                state_dict = {"model": model.state_dict()}
                DCP.load_state_dict(state_dict=state_dict, storage_reader=DCP.FileSystemReader(path_dir))
                try_load_model(model, state_dict["model"], strict)
                _optim_sd = state_dict.get("optim")
                _check_optim_loading(optimizer, _optim_sd)
                if optimizer is not None and _optim_sd is not None:
                    flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=_optim_sd)
                    optimizer.load_state_dict(flattened_osd)
        else:
            raise NotImplementedError()
        dist.barrier()  # barrier!
    else:
        _str_info = f"[{_STR_RANK}] plain <- {path_dir} | {path_model}"
        if input_sd is not None:
            state_dict = input_sd
        else:
            if os.path.exists(path_model):
                state_dict = torch.load(path_model)
            else:  # read from dir
                state_dict = {"model": model.state_dict()}
                DCP.load_state_dict(state_dict=state_dict, storage_reader=DCP.FileSystemReader(path_dir), no_dist=True)
                try_load_model(model, state_dict["model"], strict)
        if "model" in state_dict:  # one layer of model
            try_load_model(model, state_dict["model"], strict)
        else:  # directly load!
            try_load_model(model, state_dict, strict)
        _optim_sd = state_dict.get("optim")
        _check_optim_loading(optimizer, _optim_sd)
        if optimizer is not None and _optim_sd is not None:
            optimizer.load_state_dict(_optim_sd)
    # --
    if not quiet:
        ck = state_dict.get("model", state_dict)
        zlog(f"Load Checkpoint ({_str_info}) {_get_sd_info(ck)}")
    # --

def save_my_checkpoint(path: str, model, optimizer=None, conf=None, quiet=False, filter_keys=None):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, FullOptimStateDictConfig
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    import torch.distributed.checkpoint as DCP
    # --
    conf: BKNIConf = get_bk_conf(conf)
    _STR_RANK = get_rank_info(return_str=True)
    path_dir, path_model = _get_paths(path)
    model = unwrap_model(model)
    if is_fsdp_model(model):
        dist.barrier()  # barrier!
        if conf.fsdp_checkpoint_type == "full":  # save fully by streaming to the Rank0 CPU
            _str_info = f"[{_STR_RANK}] fsdp_full -> {path_model}"
            _policy0 = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            _policy1 = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, _policy0, _policy1):
                state_dict = {"model": _get_model_sd(model, filter_keys)}
                if optimizer is not None:
                    state_dict["optim"] = FSDP.optim_state_dict(model, optimizer)
                if is_main_process():
                    torch.save(state_dict, path_model)
        elif conf.fsdp_checkpoint_type == "shard":
            _str_info = f"[{_STR_RANK}] fsdp_shard -> {path_dir}"
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                state_dict = {"model": _get_model_sd(model, filter_keys)}
                if optimizer is not None:
                    state_dict["optim"] = FSDP.optim_state_dict(model, optimizer)
                distributed_writer = DCP.FileSystemWriter(path_dir)  # save to dir!
                DCP.save_state_dict(state_dict=state_dict, storage_writer=distributed_writer)
        else:
            raise NotImplementedError()
        dist.barrier()  # barrier!
    else:
        _str_info = f"[{_STR_RANK}] plain -> {path_model}"
        state_dict = {"model": _get_model_sd(model, filter_keys)}
        if optimizer is not None:
            state_dict["optim"] = optimizer.state_dict()
        torch.save(state_dict, path_model)
    # --
    if not quiet:
        ck = state_dict.get("model", state_dict)
        zlog(f"Save Checkpoint ({_str_info}) {_get_sd_info(ck)}")
    # --

# --
# wrappers

def get_profile_wrapper(do_profile: bool):
    if do_profile:
        p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        return WithWrapper(f_end=(lambda: zlog((p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)))), item=p)
    else:
        return WithWrapper()

def get_dtype_env(dtype=None):
    dtype = get_my_dtype(dtype)
    old_dtype = torch.get_default_dtype()
    return WithWrapper(f_start=(lambda: torch.set_default_dtype(dtype)), f_end=(lambda: torch.set_default_dtype(old_dtype)))

# --
# modules

# replace module
def replace_module(root_module, f_filter, f_replace, inplace=False):
    _repl_list = []
    _module_set = set(id(z) for z in root_module.modules())
    def _rec(_parent, _path, _current):
        if f_filter(_current, _path):  # lambda module, path -> bool
            if inplace:
                f_replace(_current, _path)  # inplace modification!
                _new_current = _current
            else:
                _new_current = f_replace(_current, _path)  # lambda module, path -> module
                assert isinstance(_parent, Module) and len(_path) > 0, f"Bad path {_path} or parent type {type(_parent)}"
                _key = _path[-1]
                _parent.__setattr__(_key, _new_current)  # directly replace!
            _repl_list.append((_current, _new_current))
            _current = _new_current
        for _name, _child in _current.named_children():
            if id(_child) not in _module_set:  # sometimes there may be modules added, check this!
                continue
            _path.append(_name)
            _rec(_current, _path, _child)  # recursive call!
            _path.pop()
    # --
    _rec(None, [], root_module)
    zlog(f"Replace module with {Counter([f'{type(a)} -> {type(b)}' for a, b in _repl_list])}")
    return _repl_list

# --
# peft helper
from .peft import PeftConf, PeftHelper, ModuleWrapper
# --

def set_trainable(model, trainable_names):
    for n, p in model.named_parameters():
        if ZHelper.match_name(n, trainable_names):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

# --
# other helper functions
from .helper import *
# --

# get batch idxes
def get_batch_idxes(t_input, t_local_idxes=None):
    _MAX_BS = 1000  # note: this should be enough!
    _bs = t_input.shape[0]
    assert _bs <= _MAX_BS, "Default max batch size not enough!"
    LOCAL_RANK, RANK, WORLD_SIZE = get_rank_info()
    if t_local_idxes is None:
        t_local_idxes = torch.arange(_bs, dtype=torch.long, device=t_input.device)
    ret = t_local_idxes + RANK * _MAX_BS  # [bs]
    return ret
