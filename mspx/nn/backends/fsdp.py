#

# tools for fsdp setup

from functools import partial
import torch
from torch.nn import Module
from torch.distributed.fsdp import MixedPrecision
import torch.distributed as dist

from mspx.utils import Conf, zlog, zwarn

# --
# mixed-precision

# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

fpSixteen_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

bfSixteen_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

def get_mixed_precision_policy(precision):
    if precision == "bf16m":
        mixed_precision_policy = bfSixteen
    elif precision == "fp16m":
        mixed_precision_policy = fpSixteen
    elif precision == "bf16m2":
        mixed_precision_policy = bfSixteen_mixed
    elif precision == "fp16m2":
        mixed_precision_policy = fpSixteen_mixed
    elif precision in ["fp16", "bf16", "fp32"]:
        mixed_precision_policy = None
    else:
        raise NotImplementedError()
    zlog(f"Obtain mixed_precision_policy = {mixed_precision_policy}")
    return mixed_precision_policy

def _get_wrap_cls(model, keys):
    cls_set = set()  # cls set
    for m in model.modules():
        if any(z in m.__class__.__name__ for z in keys):
            cls_set.add(type(m))
    return cls_set

def get_auto_wrapping_policy(model: Module, wrap_keys):
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    cls_set = _get_wrap_cls(model, wrap_keys)
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=cls_set)
    zlog(f"Apply auto_wrapping_policy with {cls_set}")
    return auto_wrap_policy

def get_sharding_strategy(sharding_strategy: str):
    from torch.distributed.fsdp import ShardingStrategy
    ret = {"full": ShardingStrategy.FULL_SHARD, "op": ShardingStrategy.SHARD_GRAD_OP,
           "no": ShardingStrategy.NO_SHARD, "hybrid": ShardingStrategy.HYBRID_SHARD}[sharding_strategy]
    return ret

def apply_fsdp_checkpointing(model: Module, wrap_keys):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    cls_set = _get_wrap_cls(model, wrap_keys)
    _check_fn = lambda submodule: isinstance(submodule, tuple(list(cls_set)))
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=_check_fn)
    zlog(f"Apply AC with {cls_set}")
    # --
