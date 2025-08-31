#

# utils for module/config converting

__all__ = [
    "convert_conf", "convert_sd",
]

from collections import OrderedDict
import torch
from mspx.utils import zwarn, zlog, Conf

# --
# utils

def replace_with_maps(state_dict, maps):
    rm_states = {}
    curr_state_dict = state_dict
    for r0, r1 in maps:
        new_state_dict = OrderedDict()
        for k in list(curr_state_dict.keys()):
            k2 = k.replace(r0, r1)
            if k2:
                new_state_dict[k2] = curr_state_dict[k]
            else:  # removed since no name!
                rm_states[k] = curr_state_dict[k]
        curr_state_dict = new_state_dict
    if rm_states:
        zwarn(f"Remove states: {rm_states.keys()}")
    return curr_state_dict

# --
# conversion for configs

# old_name -> new_name(s); new_name -> value; (lambda src_conf, trg_conf: ...)
# (conv_name_table, conv_direct_table, conv_function)
CONV_CONF = {
    # --
    "llama": ({}, {}, None),  # no need to convert
    "mistral":  ({}, {}, None),  # todo(+N): currently no handling of sliding window
    # gpt2
    "gpt2": ({"n_positions": "max_position_embeddings", "n_embd": "hidden_size", "n_layer": "num_hidden_layers", "n_head": "num_attention_heads", "n_inner": "intermediate_size", "activation_function": "hidden_act", "layer_norm_epsilon": "rms_norm_eps"}, {"att_use_rope": False, "mlp_use_gate": False, "use_rms_norm": False, "use_abs_posi": True, "num_key_value_heads": None, "mlp_proj_bias": True, "attention_bias": True}, None),
    # gpt_neox
    "gpt_neox": ({"rotary_emb_base": "rope_theta", "hidden_dropout": ["embd_pdrop", "resid_pdrop"], "layer_norm_eps": "rms_norm_eps"}, {"mlp_use_gate": False, "use_rms_norm": False, "mlp_proj_bias": True, "attention_bias": True}, None),
}

# ignore some fields!
_IGNORE_FIELDS = {"model_type", "is_composition", "attribute_map", "_auto_class", "task_specific_params", "architectures"}

def convert_conf(src_model_type: str, src_conf, trg_conf, quiet=False):
    conv_name_table, conv_direct_table, conv_f = CONV_CONF[src_model_type]
    # prepare src
    if src_conf is None:
        src_conf = {}
    if isinstance(src_conf, dict):  # first change it to names
        src_conf_dict = src_conf.copy()
    else:
        src_conf_dict = src_conf.__dict__.copy()
    # handle renames
    for old_name, new_names in conv_name_table.items():
        v = src_conf_dict[old_name]
        del src_conf_dict[old_name]
        if not isinstance(new_names, list):
            new_names = [new_names]
        for new_name in new_names:
            src_conf_dict[new_name] = v
    # add them in
    repl_confs, unk_confs = {}, {}
    for dd in [src_conf_dict, conv_direct_table]:  # first assign src ones, then direct ones
        for k, v in dd.items():
            if k not in _IGNORE_FIELDS and hasattr(trg_conf, k):
                v_old = getattr(trg_conf, k)
                v_assign = Conf.typed_convert(v, v_old)
                if v_assign != v_old:
                    repl_confs[k] = v_assign
                    setattr(trg_conf, k, v_assign)
            else:
                unk_confs[k] = v
    # final verify
    if conv_f:
        conv_f(src_conf, trg_conf)
    # repl info
    if not quiet:
        zlog(f"Convert conf (src={src_model_type}): R={repl_confs} ||| U={unk_confs}")
    return trg_conf

# --
# conversion for parameters

def _conv_sd_gpt2(state_dict, trg_conf):
    m = [("transformer.", "model."), (".wte.", ".embed_tokens."), (".h.", ".layers."), (".attn.", ".self_attn."), (".ln_1.", ".input_layernorm."), (".mlp.c_proj.", ".mlp.down_proj."), (".c_proj.", ".o_proj."), (".ln_2.", ".post_attention_layernorm."), (".c_fc.", ".up_proj."), ("model.ln_f.", "model.norm.")]
    ret = replace_with_maps(state_dict, m)
    _repls = ["q_proj", "k_proj", "v_proj"]
    for k in list(ret.keys()):  # [input, output] -> [output, input]
        if ".up_proj.weight" in k or ".down_proj.weight" in k or ".o_proj.weight" in k:
            ret[k] = ret[k].transpose(0, 1)
        if "c_attn.bias" in k:
            v = ret[k]
            del ret[k]
            _assigns = v.chunk(3, dim=0)
            for ii, rr in enumerate(_repls):
                ret[k.replace("c_attn", _repls[ii])] = _assigns[ii]
        if "c_attn.weight" in k:
            v = ret[k]
            del ret[k]
            _assigns = v.chunk(3, dim=1)
            for ii, rr in enumerate(_repls):  # [input, output] -> [output, input]
                ret[k.replace("c_attn", _repls[ii])] = _assigns[ii].transpose(0, 1)
    return ret

def _conv_sd_gpt_neox(state_dict, trg_conf):
    # m = [("gpt_neox.", "model."), (".embed_in.", ".embed_tokens."), (".attention.", ".self_attn."), (".mlp.c_proj.", ".mlp.down_proj."), (".c_proj.", ".o_proj."), (".dense_h_to_4h.", ".up_proj."), (".dense_4h_to_h.", ".down_proj."), ("model.final_layer_norm", "model.norm"), ("embed_out.weight", "lm_head.weight"), ("self_attn.dense", "self_attn.o_proj")]
    m = [("gpt_neox.", "model."), (".embed_in.", ".embed_tokens."), (".attention.", ".self_attn."), (".mlp.c_proj.", ".mlp.down_proj."), (".c_proj.", ".o_proj."), (".dense_h_to_4h.", ".up_proj."), (".dense_4h_to_h.", ".down_proj."), ("model.final_layer_norm", "model.norm"), ("embed_out.weight", "lm_head.weight"), (".dense", ".o_proj")]
    ret = replace_with_maps(state_dict, m)
    _repls = ["q_proj", "k_proj", "v_proj"]
    _nhead = trg_conf.num_attention_heads
    for k in list(ret.keys()):  # split
        if "query_key_value.bias" in k or "query_key_value.weight" in k:
            v = ret[k]  # [input, H*3*D]
            del ret[k]
            head_splits = v.chunk(_nhead, dim=0)
            head_qkv_splits = [z.chunk(3, dim=0) for z in head_splits]
            _assigns = [torch.cat([z[i] for z in head_qkv_splits], dim=0) for i in range(3)]
            for ii, rr in enumerate(_repls):
                ret[k.replace("query_key_value", _repls[ii])] = _assigns[ii]
    for k in list(ret.keys()):  # no need for this!
        if "inv_freq" in k:
            del ret[k]
    return ret

def _conv_sd_bloom(state_dict, trg_conf):
    m = [("transformer.", "model."), (".word_embeddings.", ".embed_tokens."), (".word_embeddings_layernorm.", ".emb_ln."), ("model.h.", "model.layers."), (".self_attention.", ".self_attn."), (".self_attn.dense.", ".self_attn.o_proj."), (".dense_h_to_4h.", ".up_proj."), (".dense_4h_to_h.", ".down_proj."), ("model.ln_f", "model.norm")]
    ret = replace_with_maps(state_dict, m)
    _repls = ["q_proj", "k_proj", "v_proj"]
    _nhead = trg_conf.num_attention_heads
    for k in list(ret.keys()):  # split
        if "query_key_value.bias" in k or "query_key_value.weight" in k:
            v = ret[k]  # [input, H*3*D]
            del ret[k]
            head_splits = v.chunk(_nhead, dim=0)
            head_qkv_splits = [z.chunk(3, dim=0) for z in head_splits]
            _assigns = [torch.cat([z[i] for z in head_qkv_splits], dim=0) for i in range(3)]
            for ii, rr in enumerate(_repls):
                ret[k.replace("query_key_value", _repls[ii])] = _assigns[ii]
    return ret

# old_str -> new_str
CONV_SD = {
    "llama": (lambda sd, trg_conf: sd),  # no need to convert
    "mistral": (lambda sd, trg_conf: sd),
    "gpt2": _conv_sd_gpt2,
    "gpt_neox": _conv_sd_gpt_neox,
    "bloom": _conv_sd_bloom,
}

def convert_sd(src_model_type: str, src_sd, trg_conf):
    tmp_src_sd = {"."+k: v for k, v in src_sd.items()}  # allow easier mapping
    new_sd = CONV_SD[src_model_type](tmp_src_sd, trg_conf)
    ret = OrderedDict({k.lstrip("."): v for k, v in new_sd.items()})  # remove extra dots!
    return ret
