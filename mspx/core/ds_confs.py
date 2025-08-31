#

# configs for deepspeed

_BASE = {
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.00025,
      "betas": [
        0.9,
        0.95
      ],
      "weight_decay": 0.1
    }
  },
  "fp16": {
    "enabled": False
  },
  "bf16": {
     "enabled": False
  },
  "amp": {
      "enabled": False,
      "opt_level": "O1",
  },
  # "zero_optimization": {
  #   "stage": 0,
  #   "overlap_comm": True,
  #   "reduce_bucket_size": 5e8
  # }
}

LIB_DS = {
# --
"": _BASE | {
  "zero_optimization": {
    "stage": 0,
    "overlap_comm": True,
    "reduce_bucket_size": 5e8
  }
},
"zero2": _BASE | {
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": True,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  }
},
"zero3": _BASE | {
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": True,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 1e7,
    "stage3_param_persistence_threshold": 1e5,
    "reduce_bucket_size": 1e7,
    "sub_group_size": 1e9,
    # "offload_optimizer": {
    #   "device": "cpu"
    # },
    # "offload_param": {
    #   "device": "cpu"
    # }
  }
},
# --
}
