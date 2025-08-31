#

# model engine (mainly for causal LM)

__all__ = [
    "MyEngineConf", "MyEngine", "register_toker_and_model",
    "load_toker_and_model", "load_toker", "load_model", "setup_conf",
]

from typing import List
import numpy as np
import torch
from collections import OrderedDict

from mspx.utils import get_global_cache_dir, zlog, zwarn, Constants, Conf, Configurable, ZHelper, ConfEntryCallback, zglob
from mspx.nn import BK
from .modules import get_mod_confs
from .helper import *
from .cache import MyCache, ListParInput

class MyEngineConf(Conf):
    def __init__(self):
        # basic
        self.model_name = 'gpt2'  # model's name; eg: gtp2-xl, meta-llama/Llama-2-7b-hf
        self.model_extra_config = {}  # updating extra configs of models
        self.model_reconfig_vocab_size = True  # whether reconfig according to new vocab
        self.model_nopre = False  # no loading pretrained weights
        self.toker_name = ''  # by default the same as model_name
        self.toker_use_fast = True  # by defaut use_fast
        self.tf_token = ""  # auto-token for transformers
        self.tf_load_kwargs = {}  # extra kwargs for "from_pretrained ..."
        self.model_max_length = Constants.INT_PRAC_MAX  # max length for forwarding
        self.add_special_tokens = False  # whether add_special_tokens for the first seg? note: simply nope by default!
        self.model_compile = False  # whether compile the model?
        # mods
        self.mods = ConfEntryCallback((lambda s: get_mod_confs(self, s)), default_s="")  # allow modifiers
        # more advanced ones
        self.model_peft_conf = BK.PeftConf()  # peft configs
        self.model_peft_target_names = []  # targets to apply peft
        self.model_trainable_names = []  # only those to be trainable (empty means ALL)
        # loading specific
        self.auto_device_map = False  # use accelerator to auto_map
        self.model_init_device = ""
        self.model_init_dtype = ""
        self.load_in_8bit = False  # load with quant-8bit
        # --

@MyEngineConf.conf_rd()
class MyEngine(Configurable):
    def __init__(self, conf: MyEngineConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyEngineConf = self.conf
        # --
        # get base model
        self.toker, self.model = load_toker_and_model(conf)
        # add mods
        self.mods = OrderedDict()
        if conf.mods:
            _dtype = BK.get_my_dtype(conf.model_init_dtype)
            _device = BK.get_my_device(conf.model_init_device)
            with _device, BK.get_dtype_env(_dtype):
                for mod_name in conf.mods:
                    _mod_conf = getattr(conf, mod_name)
                    _mod = _mod_conf.make_node()
                    _mod.modify_model(self.model, self.toker)  # inplace modification!
                    self.mods[mod_name] = _mod
        # set peft
        if conf.model_peft_target_names:
            peft_helper = BK.PeftHelper(conf.model_peft_conf)
            self.model = peft_helper.get_peft_model(self.model, conf.model_peft_target_names)
        # set trainable
        if conf.model_trainable_names:
            BK.set_trainable(self.model, conf.model_trainable_names)
        # --
        if conf.model_compile:  # todo(+N): it takes more mem with compile + deepspeed?
            zlog("Start to compile the model")
            self.model = BK.compile(self.model)
            zlog("End to compile the model")
        # --
        # setup
        self._model_max_length = min(self.toker.model_max_length, self.conf.model_max_length)
        BK.setattr_borrow(self.model, 'toker', self.toker)
        BK.setattr_borrow(self.model, 'engine', self)
        BK.setattr_borrow(self.model, '_zreal_forward', self.model.forward)
        BK.setattr_borrow(self.model, 'forward', self._do_forward, assert_nonexist=False)  # setup a wrapper function
        zlog(f"Get init toker and model:\n{self.simple_toker_repr(self.toker)}\n{self.model}")
        BK.init_seed1()  # init seed1 after we got our model!
        # --

    # --
    # general helpers

    @staticmethod
    def simple_toker_repr(toker):
        ret = str(toker)
        return "\n".join(ret.split("\n")[:5]) + " ...... "

    @property
    def model_type(self):  # from config!
        return self.model.config.model_type

    @property
    def max_length(self):  # default max length
        return self._model_max_length

    # first param
    def my_first_param(self):
        return next(self.model.parameters())

    def my_first_device(self):
        return self.my_first_param().device

    # --
    # about tokenization
    def do_subtok(self, s: str, convert_id=True):
        return TokenizerHelper.do_subtok(s, self.toker, convert_id=convert_id)

    def do_tokenize_prefixes(self, prefixes: List, add_special_tokens: bool, prepare_tensor=False):
        toker = self.toker
        # note: at least these are ok with add_special_tokens?
        # if self.model_type in ['llama', 'gpt2', 'pythia']:
        if True:
            _orig_side = toker.padding_side
            toker.padding_side = 'left'
            ret_toker = toker(prefixes, padding=True, add_special_tokens=add_special_tokens, return_tensors='pt')
            t_pre, t_pmask = ret_toker['input_ids'], ret_toker['attention_mask']
            toker.padding_side = _orig_side
        # else:
        #     raise NotImplementedError()
        if prepare_tensor:
            t_pre, t_pmask = self.do_prepare_tensors(t_pre, t_pmask)
        return t_pre, t_pmask

    def do_tokenize_choices(self, choices: List):
        toker = self.toker
        c_ids = [[self.do_subtok(one_s) for one_s in choices_l2] for choices_l2 in choices]
        _pad_id = toker.pad_token_id
        t_choices, t_cmask = DataPadder.batch_3d(c_ids, _pad_id, ret_mask=True, ret_tensor=True)  # [bs, C, L]
        return t_choices, t_cmask

    def should_add_special_tokens(self, has_past_key_values):
        my_cache = getattr(self.model, "my_cache", None)
        if my_cache is None:
            has_prev = has_past_key_values
        else:  # if using my_cache mode ...
            assert not has_past_key_values, "My_cache do not use past_key_values!!"
            has_prev = my_cache.has_prev
        return (has_prev and self.conf.add_special_tokens)

    # --
    # forward

    def do_prepare_forw_kwargs(self, t_ids, t_masks, past_key_values=None, max_length=None, **input_kwargs):
        conf: MyEngineConf = self.conf
        _max_len = Constants.INT_PRAC_MAX if max_length is None else max_length
        if past_key_values is not None:
            _past_length = past_key_values.get_seq_length()
            _max_len = max(1, _max_len - _past_length)  # minus previous idx
        else:
            _past_length = 0
        _curr_len = t_ids.shape[-1]
        if t_masks.shape[-1] > _curr_len:
            t_masks = t_masks[..., -_curr_len:]  # process input masks, make it compatible for both input/full masks!
        if _curr_len > _max_len:  # seq's length too large
            zwarn(f"Seq-length seems to be larger than allowed, simply truncate! -> {_curr_len} >= {_max_len}")
            t_ids, t_masks = t_ids[..., -_max_len:], t_masks[..., -_max_len:]
        t_masks0 = t_masks  # original ones
        # --
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            past_key_values.expand_batch_size_as(t_ids)
            t_past_mask, t_past_posi, t_past_input = past_key_values.get_inputs(["mask", "posi", "input"])
            t_masks = torch.cat([t_past_mask, t_masks], -1)
        else:
            t_past_posi = 0
            t_past_input = None
        # --
        t_posi = t_past_posi + ((t_masks0.long().cumsum(-1) - 1).clamp(min=0).to(t_ids))  # [bs, lc]
        _max_posi_idx = self.toker.model_max_length
        if t_posi.max().item() >= _max_posi_idx:  # position index too large
            zwarn(f"Positions seem to be larger than allowed, simply clamp! -> {t_posi.max().item()} >= {_max_posi_idx}")
            t_posi = t_posi.clamp(max=_max_posi_idx-1)
        next_past_posi = t_past_posi + t_masks0.long().sum(-1, keepdims=True).to(t_ids)  # [bs, 1]
        next_past_input = t_ids if t_past_input is None else torch.cat([t_past_input, t_ids], -1)  # [bs, Lfull]
        # --
        # note: avoid ALL-PAD seq (which may cause NAN); todo(+N): might have problem if there are future valid tokens!
        next_past_mask = t_masks.clone()
        t_masks_forw = t_masks.clone()
        _full_invalid = (t_masks_forw.sum(-1) <= 0)  # [bs, 1]
        t_masks_forw[_full_invalid] = 1  # simply make them all 1 for inputs!
        # --
        ret = input_kwargs.copy()
        ret.update({'input_ids': t_ids, 'attention_mask': t_masks_forw, 'past_key_values': past_key_values,
                    'position_ids': t_posi, 'use_cache': True, 'return_dict': True})
        # note: past_step details down to batch insts, while past_idx is an int
        ret_next_past = {'posi': next_past_posi, 'mask': next_past_mask, 'input': next_past_input}
        return ret, ret_next_past  # model inputs, extra ones

    def _prepare_listpar_posi(self, inputs):
        _max_len = max([len(z.tok_ids) for z in inputs], default=1)
        l_posi, l_next = [], []
        for one_input in inputs:
            one_posi, one_next = one_input.get_positions()
            _gap = _max_len - len(one_posi)
            if _gap > 0:  # fill in the last one!
                one_posi.extend([one_next] * _gap)
            l_posi.append(one_posi)
            l_next.append(one_next)
        # --
        ret_posi = torch.as_tensor(l_posi).float()
        ret_next = torch.as_tensor(l_next).float()
        return ret_posi, ret_next

    def do_prepare_listpar_forw_kwargs(self, inputs, past_key_values=None, no_smask=False, **input_kwargs):
        # TODO(+N): no handling of max-length here, for simplicity
        # first prepare inputs
        _pad_id = self.toker.pad_token_id
        t_ids, t_masks = DataPadder.batch_2d([z.tok_ids for z in inputs], _pad_id, ret_mask=True, ret_tensor=True, pad_left=True)  # [bs, L]
        t_pids, _ = DataPadder.batch_2d([z.piece_ids for z in inputs], -1, ret_mask=False, ret_tensor=True, pad_left=True)  # [bs, L]
        t_iids, _ = DataPadder.batch_2d([z.item_ids for z in inputs], -1, ret_mask=False, ret_tensor=True, pad_left=True)  # [bs, L]
        t_ids, t_masks, t_pids, t_iids = self.do_prepare_tensors(t_ids, t_masks, t_pids, t_iids)
        t_inp_posi, t_inp_next = [z.to(self.my_first_device()) for z in self._prepare_listpar_posi(inputs)]  # float; [bs, L], [bs]
        # --
        _past_length = 0
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            past_key_values.expand_batch_size_as(t_ids)
            t_past_mask, t_past_posi, t_past_input = past_key_values.get_inputs(["mask", "posi", "input"])
            t_masks = torch.cat([t_past_mask, t_masks], -1)
            _past_length = t_past_mask.shape[-1]
        else:
            t_past_posi = 0.
            t_past_input = None
        # --
        # how to specify positions?
        # t_masks0 = t_masks  # original ones
        # t_posi = t_past_posi + ((t_masks0.long().cumsum(-1) - 1).clamp(min=0).to(t_ids).float())  # [bs, lc]
        # next_past_posi = t_past_posi + t_masks0.long().sum(-1, keepdims=True).to(t_ids).float()  # [bs, 1]
        t_posi = t_past_posi + t_inp_posi  # [bs, lc]
        next_past_posi = t_past_posi + t_inp_next[..., None]  # [bs, 1]
        next_past_input = t_ids if t_past_input is None else torch.cat([t_past_input, t_ids], -1)  # [bs, Lfull]
        # to specify special_mask -> not (pid==pid & iid != iid)
        if no_smask:
            special_attention_mask = None
        else:
            t_smask = (t_pids.unsqueeze(-1) != t_pids.unsqueeze(-2)) | (t_iids.unsqueeze(-1) == t_iids.unsqueeze(-2))  # [bs, L, L] no need to care other masks since intersection!
            special_attention_mask = BK.add_paddings(t_smask, (None, [1] * _past_length))  # [bs, L, Lp+L]
        # --
        # note: avoid ALL-PAD seq (which may cause NAN); todo(+N): might have problem if there are future valid tokens!
        next_past_mask = t_masks.clone()
        t_masks_forw = t_masks.clone()
        _full_invalid = (t_masks_forw.sum(-1) <= 0)  # [bs, 1]
        t_masks_forw[_full_invalid] = 1  # simply make them all 1 for inputs!
        # --
        ret = input_kwargs.copy()
        ret.update({'input_ids': t_ids, 'attention_mask': t_masks_forw, 'past_key_values': past_key_values, 'special_attention_mask': special_attention_mask,
                    'position_ids': t_posi, 'use_cache': True, 'return_dict': True})
        # note: past_step details down to batch insts, while past_idx is an int
        ret_next_past = {'posi': next_past_posi, 'mask': next_past_mask, 'input': next_past_input}
        # breakpoint()
        return ret, ret_next_past  # model inputs, extra ones

    def do_prepare_choices(self, choices, choices_mask):
        if BK.is_tensor(choices):
            t_choices, t_cmask = choices, choices_mask
        else:
            t_choices, t_cmask = self.do_tokenize_choices(choices)
        t_choices, t_cmask = self.do_prepare_tensors(t_choices, t_cmask)  # [bs, C, L]
        return t_choices, t_cmask

    def handle_choices(self, t_choices, t_cmask, model_output, model):
        _shape = BK.get_shape(t_choices)  # [bs, C, L]
        if t_choices.shape[-1] == 1:  # if all the answers contain one subtok
            t_c, t_m = t_choices.squeeze(-1), t_cmask.squeeze(-1)  # [bs, C]
            t_logprobs = model_output.logits[..., -1, :].log_softmax(-1)  # [bs, V]
            choice_logprobs = t_logprobs.gather(-1, t_c)  # [bs, C]
        else:  # further forwarding
            _shape2 = [np.prod(_shape[:-1]).item(), _shape[-1]]  # [bs*C, L]
            t_c, t_m = t_choices.view(_shape2), t_cmask.view(_shape2)  # [bs*C, L]
            _res = self._do_forward(t_c, t_m, past_key_values=model_output.past_key_values)
            _logit_p1 = model_output.logits[..., -1:, :].repeat_interleave(_shape[-2], dim=0)  # [bs*C, 1, V]
            _logit_p2 = _res.logits[..., :-1, :]  # [bs*C, L-1, V]
            t_all_logits = BK.concat([_logit_p1, _logit_p2], dim=-2)  # [bs*C, L, V]
            t_logprobs = t_all_logits.log_softmax(-1)  # [bs*C, L, V]
            choice_logprobs = t_logprobs.gather(-1, t_c.unsqueeze(-1)).squeeze(-1)  # [bs*C, L]
        # --
        _NEGINF = -10000.
        choice_logprobs = t_m * choice_logprobs + (1. - t_m) * _NEGINF
        model_output.choice_logprobs = choice_logprobs.view(_shape)  # [bs, C, L]
        model_output.choice_masks = t_cmask  # [bs, C, L]
        # --

    # prepare for external tensor
    def do_prepare_tensors(self, input_ids, *input_ints):
        # prepare inputs for tensor & array
        device = self.my_first_device()
        # move to target device
        input_ids = input_ids.to(device).long()
        new_ints = [(None if m is None else m.to(device).long()) for m in input_ints]  # also make it LONG!
        return input_ids, *new_ints

    # wrapper for forward: note: simply depends on input_ids & past_key_values!
    def _do_forward(self, input_ids, attention_mask=None, past_key_values=None,  # main ones
                    choices=None, choices_mask=None, max_length=None, **kwargs):  # extra
        conf: MyEngineConf = self.conf
        # if kwargs:
        #     zwarn(f"Ignore extra kwargs: {kwargs.keys()}", warn_once=True)
        # if past_key_values is None and self.tmp_past_key_values is not None:
        #     past_key_values = self.tmp_past_key_values  # using tmp external ones!
        #     self.set_tmp_past_key_values(None)
        # --
        # handle past_key_values
        CACHE_CLS = self.model.get_cache_cls() if hasattr(self.model, "get_cache_cls") else MyCache
        if past_key_values is not None:
            assert isinstance(past_key_values, CACHE_CLS) or past_key_values.get_seq_length() is None or past_key_values.get_seq_length() == 0, f"Things might be missing from other caches: {type(past_key_values)}"
            past_key_values = CACHE_CLS.from_other_cache(past_key_values)  # copy it to avoid inplace modification!
        # --
        # encode prefixes and obtain outputs
        if isinstance(input_ids, ListParInput) or isinstance(input_ids[0], ListParInput):
            if isinstance(input_ids, ListParInput):
                input_ids = [input_ids]
            forw_dict, _next_past_dict = self.do_prepare_listpar_forw_kwargs(input_ids, past_key_values, **kwargs)
        else:
            if not BK.is_tensor(input_ids):
                assert attention_mask is None
                input_ids, attention_mask = self.do_tokenize_prefixes(
                    input_ids, add_special_tokens=self.should_add_special_tokens(past_key_values is not None))
            else:
                if attention_mask is None:
                    zwarn(f"attention_mask is None for tensor inputs!", warn_once=True)
                    attention_mask = BK.ones_like(input_ids)
            # prepare inputs
            input_ids, attention_mask = self.do_prepare_tensors(input_ids, attention_mask)
            _max_len = self.max_length if max_length is None else min(self.max_length, max_length)
            forw_dict, _next_past_dict = self.do_prepare_forw_kwargs(input_ids, attention_mask, past_key_values, max_length=_max_len, **kwargs)
        # forward
        zwarn(f"Forward with shape//512: {forw_dict['input_ids'].shape[-1]//512*512}", warn_once=True)
        ret = self.model._zreal_forward(**forw_dict)  # forward model
        if ret.past_key_values is not None:
            ret.past_key_values = CACHE_CLS.from_other_cache(ret.past_key_values)  # maybe not the same cls!
            ret.past_key_values.update_inputs(**_next_past_dict)
        # --
        # check the choices
        t_choices, t_cmask = self.do_prepare_choices(choices, choices_mask) if choices is not None else (None, None)
        if choices is not None:
            self.handle_choices(t_choices, t_cmask, ret, self.model)
        # --
        return ret

    def orig_forward(self, *args, **kwargs):
        ret = self.model._zreal_forward(*args, **kwargs)  # forward model
        return ret

# --
# loading utils

# list of {"func": f, "cls_model": ..., "cls_toker": ...}
# func: lambda model_name: str -> bool
_REGISTERED_TOKER_AND_MODELS = []
def register_toker_and_model(func, cls_model, cls_toker, cls_config):
    _REGISTERED_TOKER_AND_MODELS.append({"func": func, "cls_model": cls_model, "cls_toker": cls_toker, "cls_config": cls_config})

def load_toker_and_model(conf: MyEngineConf = None, **kwargs):
    # toker
    toker = load_toker(conf, **kwargs)
    if toker.pad_token_id is None:  # add padding!
        toker.pad_token_id = toker.eos_token_id
        zwarn(f"Adding pad-token for toker: {toker.pad_token_id} = {toker.convert_ids_to_tokens([toker.pad_token_id])}")
    # model
    model = load_model(conf, **kwargs)
    # --
    zlog(f"*** Obtain model & toker: {type(model)} & {type(toker)}")
    return toker, model

def setup_conf(conf: MyEngineConf = None, **kwargs):
    if conf is None or kwargs:
        conf = MyEngineConf.direct_conf(conf, copy=True, **kwargs)
    # --
    general_other_kwargs = {}
    general_other_kwargs.update(conf.tf_load_kwargs)
    cache_dir = get_global_cache_dir()
    if cache_dir:
        general_other_kwargs['cache_dir'] = cache_dir
        zlog(f"Setting cache_dir = {cache_dir}")
    if conf.tf_token:
        general_other_kwargs['token'] = conf.tf_token
    # --
    model_other_kwargs = general_other_kwargs.copy()
    if conf.load_in_8bit:
        model_other_kwargs['load_in_8bit'] = True
    if conf.auto_device_map:
        if torch.cuda.is_available():
            model_other_kwargs['device_map'] = 'auto'
        else:  # all on cpu!
            model_other_kwargs['device_map'] = {'': 'cpu'}
    else:
        model_other_kwargs['device_map'] = BK.get_my_device(conf.model_init_device)
    model_other_kwargs['torch_dtype'] = BK.get_my_dtype(conf.model_init_dtype)
    return conf, general_other_kwargs, model_other_kwargs

# syntax for toker's name: 1) myv:<path>, 2) auto_name
def load_toker(conf: MyEngineConf = None, **kwargs):
    conf, general_other_kwargs, model_other_kwargs = setup_conf(conf, **kwargs)
    # --
    toker_name = conf.toker_name if conf.toker_name else conf.model_name
    _MARK = "myv:"
    _MARK2 = "mym:"
    if toker_name.startswith(_MARK):
        from .vocab import Vocab
        toker_name = toker_name[len(_MARK):]
        _path = zglob(toker_name, only_one=True)
        toker = Vocab.load(_path)
    else:
        toker_name = toker_name[len(_MARK2):] if toker_name.startswith(_MARK2) else toker_name
        from transformers import AutoTokenizer
        _CLS_TOKER = AutoTokenizer
        for _vv in _REGISTERED_TOKER_AND_MODELS:  # search
            if _vv["func"](toker_name) and _vv["cls_toker"]:
                _CLS_TOKER = _vv["cls_toker"]
                break
        general_other_kwargs2 = general_other_kwargs.copy()
        general_other_kwargs2.update({"use_fast": conf.toker_use_fast, "trust_remote_code": True})
        toker = _CLS_TOKER.from_pretrained(toker_name, **general_other_kwargs2)
    return toker

# syntax for model's name: 1) mym:<sig>:<path> 2) auto_name
def load_model(conf: MyEngineConf = None, toker=None, **kwargs):
    from transformers import AutoModelForCausalLM, AutoConfig
    # --
    conf, general_other_kwargs, model_other_kwargs = setup_conf(conf, **kwargs)
    # --
    def _update_conf(_conf, _update_vocab_size=False):
        _update_confs = {}
        if toker is not None and (_update_vocab_size or conf.model_nopre):  # update vocab size!
            _update_confs["vocab_size"] = toker.vocab_size
        for k, v in conf.model_extra_config.items():
            if hasattr(_conf, k):  # update configs
                v_assign = Conf.typed_convert(v, getattr(_conf, k))
                _update_confs[k] = v_assign
            else:  # directly assign!
                _update_confs[k] = v
        if _update_confs:
            zlog(f"Update pretrained-confs with {_update_confs}")
            for k, v in _update_confs.items():
                setattr(_conf, k, v)
    # --
    _dtype = BK.get_my_dtype(conf.model_init_dtype)
    _device = BK.get_my_device(conf.model_init_device)
    # --
    _model_name = conf.model_name
    _MARK = "mym:"
    if _model_name.startswith(_MARK):
        from .mym import MymConfig, MymForCausalLM
        _CLS_MODEL = MymForCausalLM
        _CLS_CONFIG = MymConfig
        _model_name = _model_name[len(_MARK):]
    else:
        _CLS_MODEL = AutoModelForCausalLM
        _CLS_CONFIG = AutoConfig
        for _vv in _REGISTERED_TOKER_AND_MODELS:  # search
            if _vv["func"](_model_name) and _vv["cls_model"]:
                _CLS_MODEL = _vv["cls_model"]
                _CLS_CONFIG = _vv["cls_config"]
                break
    zlog(f"*** Creating[no-pretrain={conf.model_nopre}] model {_model_name}.")
    model_conf = _CLS_CONFIG.from_pretrained(_model_name, **general_other_kwargs)
    _update_conf(model_conf, conf.model_reconfig_vocab_size)  # update conf
    if conf.model_nopre:  # note: no other_kwargs since some are not supported!
        with torch.device(_device), BK.get_dtype_env(_dtype):
            # note: from_config needs explicit assigning this from kwargs!
            if hasattr(_CLS_MODEL, "from_config"):  # the Auto ones
                model = _CLS_MODEL.from_config(config=model_conf, attn_implementation=getattr(model_conf, "_attn_implementation_internal", None))
            else:
                model = _CLS_MODEL._from_config(config=model_conf, attn_implementation=getattr(model_conf, "_attn_implementation_internal", None))
    else:
        model = _CLS_MODEL.from_pretrained(_model_name, config=model_conf, **model_other_kwargs)
    # --
    model.eval()
    return model
