#

# mostly adapted from "modeling_llama" (slightly simplifying)
# note: using transformers.__version__ = "4.44.0"

from typing import Tuple, Optional, List, Union
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import rope_config_validation

from mspx.nn import BK
from mspx.utils import zlog, Conf, zwarn_once, GET_ENV_VAR, GlobalObjects

# --
# config

class MymConfig(PretrainedConfig):
    model_type = "mym"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        # --
        # original ones from llama
        self.vocab_size = kwargs.pop("vocab_size", 32000)
        self.hidden_size = kwargs.pop("hidden_size", 4096)
        self.intermediate_size = kwargs.pop("intermediate_size", 11008)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 32)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 32)
        # for backward compatibility
        _num_key_value_heads = kwargs.pop("num_key_value_heads", 32)
        if _num_key_value_heads is None:
            _num_key_value_heads = self.num_attention_heads
        self.num_key_value_heads = _num_key_value_heads
        self.hidden_act = kwargs.pop("hidden_act", "silu")
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 2048)
        self.initializer_range = kwargs.pop("initializer_range", 0.02)
        self.rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-6)
        self.pretraining_tp = kwargs.pop("pretraining_tp", 1)
        self.use_cache = kwargs.pop("use_cache", True)
        self.rope_theta = kwargs.pop("rope_theta", 10000.0)
        self.rope_scaling = kwargs.pop("rope_scaling", None)
        self.attention_bias = kwargs.pop("attention_bias", False)
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.mlp_bias = kwargs.pop("mlp_bias", False)
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        # --
        # gemma2
        self.head_dim = kwargs.pop("head_dim", None)
        self.final_logit_softcapping = kwargs.pop("final_logit_softcapping", None)
        self.attn_logit_softcapping = kwargs.pop("attn_logit_softcapping", None)
        self.query_pre_attn_scalar = kwargs.pop("query_pre_attn_scalar", None)
        self.sliding_window = kwargs.pop("sliding_window", None)  # note: still keep this for some external usage such as in HybridCache
        self.sliding_window_func = kwargs.pop("sliding_apply_func", None)  # lambda idx, layer: sliding_window
        self.gemma2_more_ln = kwargs.pop("gemma2_more_ln", False)  # more LN as gemma2
        self.gemma2_input_normalizer = kwargs.pop("gemma2_input_normalizer", False)  # input multiplying normalizer
        # --
        # mistral
        # --
        # qwen2
        # --
        # mine
        self.mym_ln_type = kwargs.pop("mym_ln_type", "llama_rms")  # layer-norm type
        self.mym_orig_model_type = kwargs.pop("model_type", None)  # converting from which original model type
        self.mym_att_func = kwargs.pop("mym_att_func", "sdpa")  # plain/flash2/sdpa
        # --
        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", None),
            bos_token_id=kwargs.pop("bos_token_id", 1),
            eos_token_id=kwargs.pop("eos_token_id", 2),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )
        # --

    def get_sliding_window(self, layer_idx, layer):
        if self.sliding_window_func:
            return eval(self.sliding_window_func)(layer_idx, layer)
        return None

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        from ._conv import convert_conf
        final_dict = convert_conf(config_dict)
        ret = super().from_dict(final_dict, **kwargs)
        return ret

    @classmethod
    def from_json_file(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls.from_dict(config_dict)

def get_cache_cls():
    from ..cache import MyCache
    return MyCache

# --
# modules

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as MymRotaryEmbedding
from transformers.models.llama.modeling_llama import rotate_half, apply_rotary_pos_emb

def get_layer_norm(ln_type: str, hidden_size: int, eps):
    if ln_type == "llama_rms":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        return LlamaRMSNorm(hidden_size, eps)
    elif ln_type == "gemma2_rms":
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
        return Gemma2RMSNorm(hidden_size, eps)
    else:
        from torch.nn import LayerNorm
        return LayerNorm(hidden_size, eps)

class MymMLP(nn.Module):
    def __init__(self, config: MymConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        t_up = self.up_proj(x)  # [..., Dh]
        t_gate = self.act_fn(self.gate_proj(x))  # [..., Dh]
        t_out = self.down_proj(t_gate * t_up)  # [..., D]
        return t_out

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class MymAttention(nn.Module):
    def __init__(self, config: MymConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self._attn_func = getattr(self, f"_attn_{config.mym_att_func}")
        # --
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (self.hidden_size // self.num_heads) if config.head_dim is None else config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        if config.head_dim is None:  # for the default case
            assert (self.head_dim * self.num_heads) == self.hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        # self.rotary_emb = self._get_rotary_emb(self.config)
        self.scaling = (config.query_pre_attn_scalar**-0.5) if config.query_pre_attn_scalar is not None else (self.head_dim**-0.5)
        # --

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin)
    ):
        # --
        bsz, q_len, _ = hidden_states.size()
        # proj
        query_states = self.q_proj(hidden_states)  # [bs, Lq, D]
        key_states = self.k_proj(hidden_states)  # [bs, Lk, D]
        value_states = self.v_proj(hidden_states)  # [bs, Lv, D]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [bs, H, Lq, D']
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bs, H, Lk, D']
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # [bs, H, Lv, D']
        # apply rope
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # concat past
        if past_key_value is not None:  # reuse k, v, self_attention
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)  # [bs, H, Lk, D']
        value_states = repeat_kv(value_states, self.num_key_value_groups)  # [bs, H, Lv, D']
        # --
        # attention
        attn_drop_rate = self.attention_dropout if self.training else 0.
        attn_output_mid, attn_weights = self._attn_func(query_states, key_states, value_states, attention_mask, attn_drop_rate)  # [bs, H, Lq, Dh]
        # final output
        attn_output = attn_output_mid.transpose(1, 2).contiguous().view(bsz, q_len, -1)  # [bs, Lq, D]
        attn_output = self.o_proj(attn_output)  # [bs, Lq, D]
        # --
        return attn_output, attn_weights, past_key_value

    # --
    # attn implementations

    def _attn_plain(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling  # [*, H, Lq, Lk]
        _attn_logit_softcapping = self.config.attn_logit_softcapping
        if _attn_logit_softcapping is not None:
            attn_weights = attn_weights / _attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * _attn_logit_softcapping
        if attention_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attn_drop_rate, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)  # [*, H, Lq, D]
        # --
        return attn_output, attn_weights

    def _attn_flash2(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float):
        from flash_attn import flash_attn_func
        # note: simply ignore attention_mask here, might be errored if having special attention_mask!!
        if attention_mask is not None:
            zwarn_once("Ignore attention_mask in _attn_flash2!")
        _sliding_window = self.config.get_sliding_window(self.layer_idx, self)
        if _sliding_window is not None and _sliding_window >= 0:
            _window_size = (_sliding_window, _sliding_window)
        else:
            _window_size = (-1, -1)
        _softcap = self.config.attn_logit_softcapping if self.config.attn_logit_softcapping else 0.
        attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), dropout_p=attn_drop_rate, causal=self.is_causal, softcap=_softcap, softmax_scale=self.scaling, window_size=_window_size)
        return attn_output.transpose(1, 2), None

    def _attn_sdpa(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float):
        assert self.config.attn_logit_softcapping is None, "attn_logit_softcapping not implemented for this mode!"
        if query_states.device.type == "cuda" and attention_mask is not None:  # to avoid potential sdpa bug?
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True, dropout_p=attn_drop_rate)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=attn_drop_rate, scale=self.scaling)
        return attn_output, None

class MymDecoderLayer(nn.Module):
    def __init__(self, config: MymConfig, layer_idx: int):
        super().__init__()
        self.config = config
        # --
        self.self_attn = MymAttention(config, layer_idx)
        self.layer_idx = layer_idx
        self.mlp = MymMLP(config)
        self.input_layernorm = get_layer_norm(config.mym_ln_type, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_layer_norm(config.mym_ln_type, config.hidden_size, eps=config.rms_norm_eps)
        if config.gemma2_more_ln:
            self.gemma2_more_ln = True
            self.pre_feedforward_layernorm = get_layer_norm(config.mym_ln_type, config.hidden_size, eps=config.rms_norm_eps)
            self.post_feedforward_layernorm = get_layer_norm(config.mym_ln_type, config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.gemma2_more_ln = False
        # --

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin)
    ):
        # --
        if past_key_value is not None:
            past_key_value = past_key_value.clone()  # copy to avoid in-place modification to cause problems when gradient-checkpointing
        # --
        attn_input = self.input_layernorm(hidden_states)  # [bs, L, D]
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
        )  # [bs, L, D]
        if self.gemma2_more_ln:
            attn_output = self.post_attention_layernorm(attn_output)
        attn_output = attn_output + hidden_states  # [bs, L, D]
        if self.gemma2_more_ln:
            mlp_input = self.pre_feedforward_layernorm(attn_output)
        else:
            mlp_input = self.post_attention_layernorm(attn_output)
        mlp_output = self.mlp(mlp_input)  # [bs, L, D]
        if self.gemma2_more_ln:
            mlp_output = self.post_feedforward_layernorm(mlp_output)
        final_output = mlp_output + attn_output  # [bs, L, D]
        # --
        return final_output, attn_weights, present_key_value

# --
# models

class MymPreTrainedModel(PreTrainedModel):
    config_class = MymConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MymDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # --
        if hasattr(module, "reset_parameters"):
            getattr(module, "reset_parameters")()  # directly call it
        # --
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# --
# helper for causal mask
def _prep_mask(t_input, t_mask, t_smask, sliding_window: int = None):
    _dtype, _device = t_mask.dtype, t_mask.device
    # first prepare it as: 0=nope, 1=valid
    size_bs, size_lc = list(t_input.shape)[:2]  # [bs, Lc]
    size_lpc = t_mask.shape[-1]  # [Lp+Lc]
    # first prepare the causal ones for the input
    t_arange_c = torch.arange(size_lc).to(_device)  # [Lc]
    t_arange_pc = torch.arange(size_lpc).to(_device) - (size_lpc - size_lc)  # [Lp+Lc]
    ret = (t_arange_c.unsqueeze(-1) >= t_arange_pc)  # [Lc, Lp+Lc]
    if sliding_window is not None and sliding_window >= 0:
        ret = ret & (t_arange_c.unsqueeze(-1) <= t_arange_pc + sliding_window)  # [Lc, Lp+Lc]
    ret = ret.to(_dtype)
    # then apply input mask
    ret = ret * t_mask.unsqueeze(-2)  # [bs, Lc, Lp+Lc]
    if t_smask is not None:
        ret = ret * t_smask  # [bs, Lc, Lp+Lc]
    ret = ret.unsqueeze(-3)  # [bs, 1, Lc, Lp+Lc]
    # final converting: use float!
    final_ret = torch.zeros(ret.shape, dtype=t_input.dtype, device=t_input.device)
    # note: no need to mask if the full seq is NIL
    final_ret.masked_fill_(((ret <= 0) & (ret.sum(-1, keepdims=True) > 0)), torch.finfo(t_input.dtype).min)  # [bs, 1, Lc, Lp+Lc]
    # breakpoint()
    return final_ret
# --

class MymModel(MymPreTrainedModel):
    def __init__(self, config: MymConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MymDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = get_layer_norm(config.mym_ln_type, config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = self._get_rotary_emb(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        # --

    def _get_rotary_emb(self, config):
        config2 = deepcopy(config)
        if config2.head_dim is not None:  # note: trick for new rotary cls
            config2.hidden_size = config2.num_attention_heads * config2.head_dim
        ret = MymRotaryEmbedding(config=config2)
        return ret

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        special_attention_mask = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # --
        assert not self.gradient_checkpointing and self.config.pretraining_tp <= 1, "Simply not supported!"
        # --
        # prepare inputs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        # --
        # handle past
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()
            seq_length_with_past = seq_length_with_past + past_key_values_length
        # --
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:  # [bs, Lp+Lc]
            attention_mask = torch.ones((batch_size, seq_length_with_past)).to(inputs_embeds)
        _mask_pool = {None: _prep_mask(inputs_embeds, attention_mask, special_attention_mask)}  # [bs, 1, Lc, Lp+Lc]
        # --
        # go through the layers
        hidden_states = inputs_embeds
        if self.config.gemma2_input_normalizer:
            normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # cos/sin
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        if use_cache and past_key_values is None:
            past_key_values = self.get_cache_cls()()  # make an empty new one
        for idx, decoder_layer in enumerate(self.layers):
            # --
            # get mask for this layer!
            _curr_sliding_window = self.config.get_sliding_window(idx, decoder_layer)
            if _curr_sliding_window not in _mask_pool:
                _mask_pool[_curr_sliding_window] = _prep_mask(inputs_embeds, attention_mask, special_attention_mask, sliding_window=_curr_sliding_window)
            _curr_mask = _mask_pool[_curr_sliding_window]
            # --
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            t_output, t_attn, past_key_values = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=_curr_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                position_embeddings=position_embeddings,
            )
            hidden_states = t_output
            if output_attentions:
                all_self_attns += (t_attn, )
        hidden_states = self.norm(hidden_states)  # final norm!
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = past_key_values if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_cache_cls(self):
        return get_cache_cls()

class MymForCausalLM(MymPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MymModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_output_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        special_attention_mask = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # --
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            special_attention_mask=special_attention_mask,
        )
        hidden_states = outputs[0]
        logits, loss = self.get_output_logits(hidden_states), None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits.float(), shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, position_ids=None, **kwargs
    ):
        assert inputs_embeds is None, "Only support inputs_ids as input"
        if past_key_values is not None:
            _slen = past_key_values.get_seq_length()
            if _slen is not None and _slen > 0:
                input_ids = input_ids[:, _slen:]  # get current input
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[-1]:]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)
        model_inputs = {"input_ids": input_ids, "position_ids": position_ids, "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask}
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.get("config", None)
        if config is None:  # no extra config provided
            return super().from_pretrained(*args, **kwargs)
        else:
            assert isinstance(config, MymConfig)
            _model_type = config.mym_orig_model_type
            if _model_type in ["mym", "llama", "gemma2", "mistral", "qwen2"]:  # for simplicity, no need to change
                return super().from_pretrained(*args, **kwargs)
            else:  # need some name converting!
                from transformers import AutoModelForCausalLM
                from ._conv import convert_sd
                kwargs2 = kwargs.copy()
                del kwargs2["config"]
                model0 = AutoModelForCausalLM.from_pretrained(*args, **kwargs2)  # get a model first
                model0.eval()
                sd0 = model0.state_dict()
                del model0
                sd2 = convert_sd(_model_type, sd0, config)
                m = super().from_pretrained(*args, **kwargs, state_dict=sd2)
                return m
        # --

    def get_cache_cls(self):
        return get_cache_cls()

# --
# b mspx/core/mym/modeling:??
