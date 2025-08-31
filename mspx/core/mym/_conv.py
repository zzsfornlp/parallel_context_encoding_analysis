#

__all__ = [
    "convert_conf", "convert_sd",
]

from collections import OrderedDict
from mspx.utils import zwarn
from ..modules._conv import replace_with_maps

# --
# conversion for configs (simplifying)

# helper
def _convert_dict(d: dict, table):
    for old_name, new_names in table:
        v = d[old_name]
        del d[old_name]
        if not isinstance(new_names, list):
            new_names = [new_names]
        for new_name in new_names:
            d[new_name] = v
    return d

# from gemma to llama
def _conv_gemma2(config_dict: dict):
    zwarn("*****\n*****It still seems slightly strange when testing gemma2 with these codes, might need to be careful!\n*****")  # todo(+N)
    TABLE = [("hidden_activation", "hidden_act")]
    ret = _convert_dict(config_dict, TABLE)
    # --
    _sliding_window = ret.pop("sliding_window", None)
    _sliding_window_func = f"(lambda layer_idx, layer: ({int(_sliding_window) if _sliding_window else 'None'} if not bool(layer_idx % 2) else None))"
    # --
    ret.update({"mym_att_func": "flash2", "mym_ln_type": "gemma2_rms", "sliding_window": _sliding_window, "sliding_window_func": _sliding_window_func, "gemma2_more_ln": True, "gemma2_input_normalizer": True, "tie_word_embeddings": True})
    return ret

# from mistral to llama
def _conv_mistral(config_dict: dict):
    assert not config_dict["sliding_window"], "More recent models do not use this!"
    return config_dict

# from qwen2 to llama
def _conv_qwen2(config_dict: dict):
    assert not config_dict["use_sliding_window"], "More recent models do not use this!"
    config_dict["attention_bias"] = True
    return config_dict

CONV_CONF = {
    "llama": (lambda x: x),
    "gemma2": _conv_gemma2,
    "mistral": _conv_mistral,
    "qwen2": _conv_qwen2,
}

def convert_conf(config_dict):
    ff = CONV_CONF[config_dict['model_type']]
    ret = ff(config_dict)
    return ret

# --
# conversion for state_dict

# old_str -> new_str
CONV_SD = {
    "llama": (lambda sd, conf: sd),  # no need to convert
    "gemma2": (lambda sd, conf: sd),
    "mistral": (lambda sd, conf: sd),
    "qwen2": (lambda sd, conf: sd),
}

def convert_sd(src_model_type: str, src_sd, conf):
    tmp_src_sd = {"."+k: v for k, v in src_sd.items()}  # allow easier mapping
    new_sd = CONV_SD[src_model_type](tmp_src_sd, conf)
    ret = OrderedDict({k.lstrip("."): v for k, v in new_sd.items()})  # remove extra dots!
    return ret
