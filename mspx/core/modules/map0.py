#

import re
from typing import Optional, Tuple, List
import math
from functools import partial
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from mspx.nn import BK
from mspx.utils import zwarn_once, Conf, zwarn, Constants, zlog, ZDEBUG
from ._base import BaseModifierConf, BaseModifier

# --
class MyMap0ModifierConf(BaseModifierConf):
    def __init__(self):
        super().__init__()
        self.map0_conf = MyMap0LayerConf()

@MyMap0ModifierConf.conf_rd()
class MyMap0Modifier(BaseModifier):
    def __init__(self, conf: MyMap0ModifierConf):
        super().__init__(conf)

    def modify_model(self, model, toker):
        BK.replace_module(model, (lambda m, p: p and m.__class__.__name__.endswith("DecoderLayer")), self.convert_module)
        # --
        # TODO(+N): for simplicity, just re-init
        model.apply(model._init_weights)
        model.tie_weights()
        # --

    def convert_module(self, module, path):
        from ._conv import convert_conf, convert_sd
        _CLS_ACONF, _CLS_ATT = MyMap0LayerConf, MyMap0Layer
        m_cls_name = module.__class__.__name__
        # which cls?
        m_cls_type = None
        for cls_type, pattern in [("llama", r"Llama.*")]:
            if re.match(pattern, m_cls_name):
                m_cls_type = cls_type
        assert m_cls_type is not None, f"Unknown cls_type for {module}"
        # first convert conf
        att_conf0 = _CLS_ACONF()  # a new trg conf
        # att_conf1 = convert_conf(m_cls_type, module.config, att_conf0)  # convert
        att_conf1 = convert_conf(m_cls_type, module.self_attn.config, att_conf0)  # convert
        # make a new module
        conf_kwargs = {k: v for k, v in att_conf1.__dict__.items()}
        conf_kwargs["_orig_cls_type"] = m_cls_type
        conf_kwargs.update(self.conf.map0_conf.diff_with_default())  # update with updated ones
        att_conf = _CLS_ACONF.direct_conf(**conf_kwargs)
        att_module = _CLS_ATT(att_conf, layer_idx=None)
        # copy weights
        # TODO(+N): no weight-loading for now!
        # orig_sd = module.state_dict()
        # new_sd = convert_sd(m_cls_type, orig_sd, att_conf)  # convert config!
        # BK.try_load_model(att_module, new_sd, None)  # load them!
        # --
        # BK.setattr_borrow(att_module, "_orig_module", module)  # mainly for debug!!
        # --
        return att_module

# --
# My Map0 Layer

class MyMap0LayerConf(Conf):
    def __init__(self):
        self.hidden_size = 2048
        self.hidden_act = 'silu'
        self.intermediate_size = 5632
        self.rms_norm_eps = 1e-5
        self.att_act = 'silu'
        self.max_position_embeddings = 2048

# --
# from llama
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class LlamaRope(torch.nn.Module):
    def __init__(self, dim: int, max_position_embeddings=4096, base=10000., scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.long).float() / self.dim))  # [F]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)  # [bs, F, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [bs, 1, L]
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # [bs, L, F]
            emb = torch.cat((freqs, freqs), dim=-1)  # [bs, L, D]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def rotate_half(x):
        # Rotates half the hidden dims of the input.
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, dim: int):
        # qk: [bs, ..., L, ..., D], cos/sin: [bs, L, D]
        _dim = dim
        if _dim is not None:
            cos = cos.unsqueeze(_dim)
            sin = sin.unsqueeze(_dim)
        q_embed = (q * cos) + (LlamaRope.rotate_half(q) * sin)
        k_embed = (k * cos) + (LlamaRope.rotate_half(k) * sin)
        return q_embed, k_embed

# --

class MyMap0Layer(nn.Module):
    def __init__(self, config: MyMap0LayerConf, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # --
        # modules
        conf = self.config
        self.mlp = LlamaMLP(conf)
        self.input_layernorm = LlamaRMSNorm(conf.hidden_size, eps=conf.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(conf.hidden_size, eps=conf.rms_norm_eps)
        # --
        self.self_attn = MyMap0Att(conf, layer_idx=layer_idx)
        self.comb_gate = nn.Linear(conf.hidden_size, conf.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # --
        residual = hidden_states  # [bs, L, D]
        t_input = self.input_layernorm(hidden_states)  # [bs, L, D]
        # time-mixing
        t_att, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=t_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )  # [bs, L, D]
        # channel-mixing
        t_mlp = self.mlp(t_input)  # [bs, L, D]
        # combine
        t_comb_weight = F.sigmoid(self.comb_gate(t_input))  # [bs, L, D]
        t_comb = t_mlp * t_comb_weight + t_att * (1. - t_comb_weight)  # [bs, L, D]
        t_ret = residual + t_comb  # [bs, L, D]
        # return
        outputs = (t_ret, )
        if output_attentions:
            outputs += (self_attn_weights, )
        if use_cache:
            outputs += (present_key_value, )
        return outputs

class MyMap0Att(nn.Module):
    def __init__(self, config: MyMap0LayerConf, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # --
        conf = self.config
        self.k1_proj = nn.Linear(conf.hidden_size, conf.hidden_size, bias=False)
        self.k2_proj = nn.Linear(conf.hidden_size, conf.hidden_size, bias=False)
        self.v_proj = nn.Linear(conf.hidden_size, conf.hidden_size, bias=False)
        self.act_fn = ACT2FN[conf.att_act]
        self.rotary_emb = LlamaRope(conf.hidden_size, max_position_embeddings=conf.max_position_embeddings)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        # --
        _hidden_size = self.config.hidden_size
        _dim_sqrt = math.sqrt(_hidden_size)
        # proj
        t_k1 = self.k1_proj(hidden_states) / _dim_sqrt  # [bs, Lc, D]
        t_k2 = self.k2_proj(hidden_states) / _dim_sqrt
        t_v = self.v_proj(hidden_states) / _dim_sqrt
        # concat cache
        cos, sin = self.rotary_emb(t_v, position_ids)
        t_q, t_k1 = LlamaRope.apply_rotary_pos_emb(hidden_states, t_k1, cos, sin, dim=None)
        if past_key_value is not None and (hasattr(past_key_value, 'get_seq_length') and past_key_value.get_seq_length() > 0):
            if isinstance(past_key_value, tuple):
                past_k1, past_k2, past_v = past_key_value
                t_k1, t_k2, t_v = BK.concat([past_k1, t_k1], dim=-2), BK.concat([past_k2, t_k2], dim=-2), BK.concat([past_v, t_v], dim=-2)  # [bs, Lp+Lc, D]
            else:
                raise NotImplementedError(f"Cache not yet supported: {past_key_value}")
        # "att"
        t_att_logits = torch.matmul(t_q, t_k1.transpose(-1, -2))  # [bs, Lc, Lp+Lc]
        t_att_act = self.act_fn(t_att_logits)  # [bs, Lc, Lp+Lc]
        t_linear0 = torch.matmul(t_q, t_k2.transpose(-1, -2))  # [bs, Lc, Lp+Lc]
        t_lact = t_att_act * t_linear0  # [bs, Lc, Lp+Lc]
        # mask
        if attention_mask is None:  # implicit causal
            t_mask = BK.full_like(t_lact, fill_value=1.)  # [bs, Lc, Lp+Lc]
            _len_curr = hidden_states.shape[-2]  # [Lc]
            _t_arange = BK.arange(_len_curr)  # [Lc, Lc]
            t_mask[:, :, -_len_curr:] *= (_t_arange.unsqueeze(-1) >= _t_arange.unsqueeze(-2)).to(t_mask)  # [bs, Lc, Lp+Lc]
        else:
            t_mask = attention_mask
        if len(t_mask.shape) == 4:
            t_mask = t_mask[:, 0]  # note: just one head for now!
        elif len(t_mask.shape) == 2:
            t_mask = t_mask.unsqueeze(-2)  # [bs, 1, Lp+Lc]
        else:
            assert len(t_mask.shape) == 3
        t_lact = t_lact * (t_mask > 0).to(t_lact)  # [bs, Lc, Lp+Lc]; force zero!
        t_linear1 = torch.matmul(t_lact, t_v)  # [bs, Lc, D]
        # --
        # return: output, "att"-weights, cache
        return t_linear1, t_lact, (t_k1, t_k2, t_v)
