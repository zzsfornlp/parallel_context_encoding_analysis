#

__all__ = [
    "MyAttentionModifierConf", "MyAttentionModifier", "MyAttentionConf", "MyAttention", "MyCache",
]

import re
from typing import Optional, Tuple, List
import math
from functools import partial
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

from mspx.nn import BK
from mspx.utils import zwarn_once, Conf, zwarn, Constants, zlog, ZDEBUG
from ._base import BaseModifierConf, BaseModifier

# --
class MyAttentionModifierConf(BaseModifierConf):
    def __init__(self):
        super().__init__()
        self.att_conf = MyAttentionConf()

@MyAttentionModifierConf.conf_rd()
class MyAttentionModifier(BaseModifier):
    def __init__(self, conf: MyAttentionModifierConf):
        super().__init__(conf)

    def modify_model(self, model, toker):
        my_cache = MyCache(self.conf)
        self.init_special_token(model, toker)  # add special tokens
        BK.replace_module(model, (lambda m, p: p and m.__class__.__name__.endswith("Attention")), partial(self.convert_module, my_cache=my_cache))
        BK.setattr_borrow(model, '_orig_nomodatt_forward', model.forward)
        BK.setattr_borrow(model, '_orig_nomodatt_generate', model.generate)
        BK.setattr_borrow(model, '_orig_nomodatt_prepare_inputs_for_generation', model.prepare_inputs_for_generation)
        BK.setattr_borrow(model, 'forward', partial(self.new_forward, model=model, toker=toker), assert_nonexist=False)  # setup a wrapper function
        BK.setattr_borrow(model, 'generate', partial(self.new_generate, model=model), assert_nonexist=False)  # setup a wrapper function
        BK.setattr_borrow(model, 'prepare_inputs_for_generation', partial(self.new_prepare_inputs_for_generation, model=model), assert_nonexist=False)  # setup a wrapper function
        BK.setattr_borrow(model, 'my_cache', my_cache)
        BK.setattr_borrow(model, 'set_my_cache', partial(self.new_set_my_cache, model=model))

    def new_forward(self, *args, no_reset_cache=False, model=None, toker=None, **kwargs):
        forw_dict = self.do_prepare_forw_kwargs(*args, model=model, toker=toker, **kwargs)
        _use_activation_checkpointing = model.training and getattr(model, "_use_activation_checkpointing", False)  # for checkpointing activation
        with model.my_cache.env_no_update(no_update=_use_activation_checkpointing):  # only for model forwarding (useful for gradient checkpointing)
            ret = model._orig_nomodatt_forward(**forw_dict)
        t_revidxes = model.my_cache.get("t_revidxes", tmp=True)
        if getattr(ret, "logits", None) is not None:
            ret.logits = BK.gather_first_dims(ret.logits, t_revidxes, dim=1)  # [bs, orig_len, ...]
        if getattr(ret, "loss", None) is not None:
            ret.loss = ret.loss.gather(-1, t_revidxes)
        if getattr(ret, "hidden_states", None) is not None:
            ret.hidden_states = tuple([BK.gather_first_dims(z, t_revidxes, dim=1) for z in ret.hidden_states])
        if not _use_activation_checkpointing:  # no resetting here if use AC!
            _full_reset = not (no_reset_cache or getattr(model, "_no_reset_cache", False))
            model.my_cache.reset_cache(full_reset=_full_reset)
        return ret

    def new_generate(self, *args, model=None, **kwargs):
        assert kwargs.get('num_beams', 1) <= 1, "Currently no support for beam search with this model since difficult for reorder_cache!!"
        _orig_flag = getattr(model, "_no_reset_cache", False)
        setattr(model, "_no_reset_cache", True)
        ret = model._orig_nomodatt_generate(*args, **kwargs)
        setattr(model, "_no_reset_cache", _orig_flag)
        model.my_cache.reset_cache()  # reset at the end of generation
        return ret

    def new_prepare_inputs_for_generation(self, input_ids, model=None, **kwargs):
        my_cache = model.my_cache
        if my_cache.has_prev:
            input_ids = input_ids[:, my_cache.past_length:].contiguous()  # past length indicates sum of len of original input ids
        ret = model._orig_nomodatt_prepare_inputs_for_generation(input_ids, **kwargs)
        return ret

    def new_reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError("")  # TODO(+N): difficult to implement with internal cache, so currently no support for beam search!

    def new_set_my_cache(self, cache, model=None):
        model.my_cache = cache  # simply set cache!
        for mod in cache.mods.values():
            cache.register_mod(mod)

    def convert_module(self, module, path, my_cache):
        from ._conv import convert_conf, convert_sd
        _CLS_ACONF, _CLS_ATT = MyAttentionConf, MyAttention
        m_cls_name = module.__class__.__name__
        # which cls?
        m_cls_type = None
        for cls_type, pattern in [("llama", r"Llama.*Attention"), ("gpt2", r"GPT2.*Attention"), ("gpt_neox", r"GPTNeoX.*Attention"), ("mistral", r"Mistral.*Attention")]:
            if re.match(pattern, m_cls_name):
                m_cls_type = cls_type
        assert m_cls_type is not None, f"Unknown cls_type for {module}"
        # first convert conf
        att_conf0 = _CLS_ACONF()  # a new trg conf
        att_conf1 = convert_conf(m_cls_type, module.config, att_conf0, quiet=(my_cache.num_of_registered_mods != 0))  # convert
        # make a new module
        conf_kwargs = {k: v for k, v in att_conf1.__dict__.items()}
        conf_kwargs["_orig_cls_type"] = m_cls_type
        conf_kwargs.update(self.conf.att_conf.diff_with_default())  # update with updated ones
        att_conf = _CLS_ACONF.direct_conf(**conf_kwargs)
        att_module = _CLS_ATT(att_conf, layer_idx=None)
        my_cache.register_mod(att_module)
        # copy weights
        orig_sd = module.state_dict()
        new_sd = convert_sd(m_cls_type, orig_sd, att_conf)  # convert config!
        BK.try_load_model(att_module, new_sd, None)  # load them!
        # --
        # BK.setattr_borrow(att_module, "_orig_module", module)  # mainly for debug!!
        # --
        return att_module

    # init special tokens and related information
    def init_special_token(self, model, toker):
        att_conf: MyAttentionConf = self.conf.att_conf
        # --
        # prepare embeddings and init tensor
        m_emb0 = model.get_input_embeddings()  # original embeddings
        assert toker.vocab_size == m_emb0.num_embeddings, f"Unmatched vocab size: {toker.vocab_size} != {m_emb0.num_embeddings}"
        with BK.no_grad():
            _att_ctok_inits = att_conf.att_ctok_inits
            if not _att_ctok_inits:  # simply average all
                t_init = m_emb0.weight.mean(0)  # [D]
                init_toks = ["ALL"]
            else:  # using special tokens
                from ..helper import TokenizerHelper
                init_ids = [TokenizerHelper.do_subtok(s, toker, ensure_continuing=True)[-1] for s in _att_ctok_inits]
                init_toks = toker.convert_ids_to_tokens(init_ids)
                t_init = m_emb0.weight[init_ids].mean(0)  # [D]
        # prepare ids
        id_cur = toker.vocab_size  # next id to assign
        id_cur0 = id_cur
        ctok_ids = []  # [Level] * [Num]
        for level in range(att_conf.att_ctok_num_of_level):
            _ctok_num = att_conf.att_ctok_nums[level]
            if att_conf.att_ctok_level_diff:  # use different tokens
                ctok_ids.append(list(range(id_cur, id_cur+_ctok_num)))
                id_cur += _ctok_num
            else:  # use same tok at the same level
                ctok_ids.append([id_cur] * _ctok_num)
                id_cur += 1
        with BK.no_grad():
            model.resize_token_embeddings(id_cur, pad_to_multiple_of=32)
            model.get_input_embeddings().weight[id_cur0:id_cur, :] = t_init  # init as the same one!
        BK.setattr_borrow(model, 'CTOK_IDS', ctok_ids)
        zlog(f"Add special tokens: {ctok_ids}, init from {init_toks}")
        # --

    # note: similar to MyEngine.do_prepare_forw_kwargs
    def do_prepare_forw_kwargs(self, input_ids, attention_mask, max_length=None, model=None, toker=None, **input_kwargs):
        # get seq length constraint
        _max_len = Constants.INT_PRAC_MAX if max_length is None else max_length
        my_cache = model.my_cache
        t_ids, t_masks = input_ids, attention_mask
        # read things from and update my_cache
        _past_idx = my_cache.past_length  # note: not exactly accurate ...
        _max_len = max(1, _max_len - _past_idx)  # minus previous idx
        _curr_len = t_ids.shape[-1]  # length of the current input
        if t_masks.shape[-1] > _curr_len:
            t_masks = t_masks[..., -_curr_len:]  # process input masks, make it compatible for both input/full masks!
        if _curr_len > _max_len:  # seq's length too large
            zwarn(f"Seq-length seems to be larger than allowed, simply truncate! -> {_curr_len} >= {_max_len}")
            t_ids, t_masks = t_ids[..., -_max_len:], t_masks[..., -_max_len:]
        # --
        forw_dict = my_cache.prepare_and_update_inputs(t_ids, t_masks, ctok_ids=model.CTOK_IDS, pad_token_id=toker.pad_token_id, **input_kwargs)
        return forw_dict

# --
# My Attention

class MyAttentionConf(Conf):
    def __init__(self):
        # the ones from llama
        self.hidden_size = 1024
        self.num_attention_heads = 8
        self.num_key_value_heads = None
        self.max_position_embeddings = 4096
        self.rope_theta = 10000.
        self.rotary_pct = 1.
        self.attention_dropout = 0.0
        self.attention_bias = False
        self.sliding_window = None  # None means not using!
        # some more
        self._orig_cls_type = ""  # filled from the outside!
        self.att_use_rope = True  # whether use rope?
        self.att_func = "sdpa"  # plain/flash2/sdpa
        # --
        # more advanced ones (for compressions)
        self.att_recencies = [0]  # [level] <=0 means attending to all
        # self.att_sink_zero = False  # zero for sink tokens (both kv)? (no specifying for now)
        self.att_ctok_nums = [0]  # [level] compressing tokens for each level (L0 means sink tokens)
        self.att_ctok_level_diff = True  # if multiple ctoks at one level, will they get different embeddings?
        self.att_ctok_inits = ["\n", " ", ".", ",", ":", ";"]  # cotk embed init: take the average of what (empty means all)
        self.att_chunk_size = Constants.INT_PRAC_MAX  # chunk the query for efficiency
        self.att_use_full_posi = False  # include special tokens as full position ids
        self.att_isel_thresh = 0.7  # do index-select if ratio is under this
        self.att_shrink_thresh = 0.75  # do cache-shrinking if ratio is under this
        # for segment sampling
        self.sample_align_batch = False  # whether sampling the same along the batches
        self.sample_seg_ranges = [[16, 16]]  # [level-1] [Left, Right] segment sizes to sample

    @property
    def att_ctok_num_of_level(self):
        return len(self.att_recencies)  # number of compression levels (including the input one as L0)

    @property
    def att_sink_num(self):
        return self.att_ctok_nums[0]  # for simplicity

class MyRope(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000., scaling_factor=1.0):
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

def rotate_half(x):
    # Rotates half the hidden dims of the input.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # qk: [bs, ..., L, ..., D], cos/sin: [bs, L, D]
    _dim = -2
    cos = cos.unsqueeze(_dim)
    sin = sin.unsqueeze(_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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

# --
# helper for causal mask
def _prep_mask(t_input, t_mask):
    # first prepare it as: 0=nope, 1=valid
    size_bs, size_len = list(t_input.shape)[:2]  # [bs, Lc]
    # first prepare the causal ones for the input
    ret = torch.ones([size_len, t_mask.shape[-1]], dtype=t_mask.dtype, device=t_mask.device)  # [Lc, Lp+Lc]
    t_arange = torch.arange(size_len).to(t_mask.device)  # [Lc]
    c_mask = (t_arange <= t_arange.unsqueeze(-1)).to(t_mask)  # [Lc, Lc]
    ret[..., -size_len:] *= c_mask
    # then apply input mask
    ret = (ret * t_mask.unsqueeze(-2)).unsqueeze(-3)  # [bs, 1, Lc, Lp+Lc]
    # final converting: use float!
    final_ret = torch.zeros(ret.shape, device=t_input.device).float()  # note: simply put it as float here!
    # note: no need to mask if the full seq is NIL
    final_ret.masked_fill_(((ret <= 0) & (ret.sum(-1, keepdims=True) > 0)), torch.finfo(final_ret.dtype).min)  # [bs, 1, Lc, Lp+Lc]
    return final_ret
# --

# TODO(+N): haven't yet done detailed debugging/running!
"""
# Some explanations:
1. About the states:
    - past_length: sum of the past original input lengths (including original pads, but not extended ctok ones)
    - overall_cache: states to be maintained during multiple forwards, including (multi-level) positions and masks for the kv caches
        - t_mask [bs, Lp]: all past mask
        - t_level_posi [bs, Lp, level+1]: all past position ids (for all levels +1)
        - t_level [bs, Lp]: all past level information
        - t_next_posi [bs, level+1]: next position ID for all levels + 1=special_full_posi
        - t_next_count [bs, level+1]: how many previous tokens already get this position ID
    - layered_caches: mainly for actual kv caches
        - layer_idx -> {'k', 'v'} [bs, Lp, H', Dh]: kv caches (for all tokens, not transposed here for easier selecting)
    - tmp_cache: tmp pre-calculated ones which are useful for this one forward (such as some selecting indexes and att_masks)
        - t_revidxes [bs, L_original_input]: for reverse indexing
        - input_info/
            - kv_idx [bs, piece*Lkv]: selecting indexes for all the current query pieces
            - kv_att_mask [bs*piece, chunk_size, Lkv]: attention mask
    - *Note: currently caches are all internally stored (and refreshed with explicit functions), not using external ones (use_cache=False & past_key_value=None)
2. About the process:
2.1: MyAttentionModifier.new_forward
    - Call do_prepare_forw_kwargs to prepare inputs
    - Call reset_cache to clear cache: always clear tmp ones; clear kv and overall ones according to the flag "no_reset_cache"
    - Further call MyCache.prepare_and_update_inputs (which does the most work) inside do_prepare_forw_kwargs
    - Finally call MyCache.reset_cache to do some resetting (and maybe cache shrinking)
2.2: MyCache.prepare_and_update_inputs
    - Expand batch size if needed (might be useful for scenarios with common prefixes)
    - Call MyCache.prepare_positions to obtain position and ctok-mark information for the original inputs [2.2.1]
    - Call MyCache.prepare_extended_inputs to prepare extended inputs [2.2.2]
    - Call MyCache.update_with_inputs to prepare extra information for this forward [2.2.3]
    - Prepare new inputs for the forward, note that there is a flag of "att_use_full_posi" indicating which position_ids we are using to calculate rope (0:level-0 or -1:full-idx, see later "full_position_ids" for full-idx)
2.2.1: MyCache.prepare_positions
    - The input has the shape of [bs, Lc], this function will prepare positions ids (for each level) and ctok marks (for each level)
    - If there are no "input_marks" provided from the outside, we will segment things by random sampling according to the configs (sample_seg_ranges)
    - Each level has its own position ids, there is one final full_position_ids (t_posi_full) that counts all the tokens (no matter original input or inserted ctoks), this will be calculated in the next step (prepare_extended_inputs)
    - The outputs will still have the shape of [bs, Lc]
2.2.2: MyCache.prepare_extended_inputs
    - This function will obtain the full extended inputs (by inserting the ctoks)
    - Basically we first add ctoks to each of the input token and then do "mask2idx" selecting according to the marks prepared in the previous step
    - Each of the original inputs will get an extended version, "t_input" [bs, Lc] -> "te_input" [bs, Lce=Lc+Sum(ctoks)]
    - In addition to "te_input" and "te_mask", we get another tensor of "te_level", indicating each token's level: -2 means invalid, -1 means sink tokens, 0 means level-0 (original valid input), 1 means level-1 ctok, ...
    - te_level_posi[..., -1] (which is full_position_ids) are calculated here
2.2.3: MyCache.update_with_inputs
    - This function prepares various kinds of indexes and masks for later attention calculation
    - First t_next_posi/t_next_count are calculated for position calculation in future forwards
    - There is a config specifying the chunk size of the query; if the input seq len is smaller than this, then the chunk_size will be set to it (just one chunk)
    - After chunked, the input will be from [bs, Lce] to [bs, piece, chunk_size]
    - For each chunk, the max and min attention positions (t_max_posi, t_min_posi) are calculated (max=Max_position, min=Min_position-recency+1)
    - For each level, we finally prepare local attention indexes ("kv_idx") and attention masks ("kv_att_mask") for all the chunks. Sometimes, we do not need the indexes for re-selecting if it is not necessary (currently if there is only one chunk or the selecting size is close to the full KV size). If using the index, then the positions and masks for the caches are also selected with this index. 
    - Finally, the attention masks are calculated by considering three things: 1) causal masking (by checking "full_position_ids"), 2) item masking (by checking the KV cache's mask), 3) local_window/recency (by checking position ids of the corresponding level)
2.3: MyAttention.forward
    - Firstly, qkv are transposed slightly later to make things easier to index
    - Concat new kv caches, so each kv cache will have the size of [bs, Lpast+Lnew, H, Dh]
    - Chunk the input and reshape, so finally the query will have the shape of [bs*piece, H, chunk_size, Dh]
    - Prepare KVs/masks for each level and concat them, this involves using "kv_idx" to index the full cache and using "kv_att_mask" as the attention mask, so finally the KV will have the shape of [bs*piece, H, Lkv, Dh]
    - Perform the attention as before, and then do some reshaping/transposing and finish
3. Misc:
3.1: Some of the util functions are heavily utilized:
    - BK.mask2idx: return (padded) corresponding indexes according to a mask tensor
    - BK.gather_first_dims: gather only for some of the first dims (choosing all for the remaining ones)
3.2: TODOs
    - This implementation is mostly trading memory cost (need to prepare KVs for each chunk) for local attention, any ways to further reduce this cost?
    - More efficient implementations to avoid the potential O(N^2) steps on index selecting (maybe by further chunking the KV cache?) (see "TODO(+N!)", though these are light-weight since only on indexing)
    - Better positional encodings?
    - ...
"""

class MyCache:
    def __init__(self, conf: MyAttentionModifierConf):
        self.conf = conf
        # --
        self.past_length = 0  # past length = sum(len(orig-input_ids))
        self.overall_cache = {}  # cache for the full seq
        self.layered_caches = {}  # cache for each of the mods: key -> {}
        self.tmp_cache = {}  # k->v (non-persistent/tmp ones)
        self.mods = {}  # idx -> Module
        # flags
        self._no_update = False

    @contextmanager
    def env_no_update(self, no_update: bool):
        _old_flag = self._no_update
        self._no_update = no_update
        try:
            yield None
        finally:
            self._no_update = _old_flag

    @property
    def has_prev(self):
        return self.past_length > 0

    @property
    def num_of_registered_mods(self):
        return len(self.layered_caches)

    def copy(self):
        # --
        def _shallow_copy(_v):  # allow nesting
            if isinstance(_v, list):
                return [_shallow_copy(z) for z in _v]
            elif isinstance(_v, dict):
                return {a: _shallow_copy(z) for a, z in _v.items()}
            else:
                return _v
        # --
        ret = MyCache(self.conf)  # sharing conf is fine!
        ret.past_length = self.past_length
        ret.overall_cache = _shallow_copy(self.overall_cache)
        ret.layered_caches = _shallow_copy(self.layered_caches)
        # note: no copy for tmp ones!
        ret.mods = self.mods.copy()  # shallow copy
        return ret

    def register_mod(self, mod):
        if mod.layer_idx is None:
            layer_idx = self.num_of_registered_mods
            assert layer_idx not in self.layered_caches
            self.layered_caches[layer_idx] = {}
            mod.layer_idx = layer_idx
        else:
            assert mod.layer_idx in self.layered_caches
        self.mods[mod.layer_idx] = mod
        mod.my_cache = MyCacheLayeredView(self, mod.layer_idx)  # return a view for each module

    def reset_cache(self, full_reset=True):
        if full_reset:
            self.past_length = 0
            self.overall_cache.clear()
            for v in self.layered_caches.values():
                v.clear()
        else:
            att_conf = self.conf.att_conf
            if any(z>0 for z in att_conf.att_recencies) and self.has_prev:  # try shrinking
                t_mask, t_level_posi, t_level, t_next_posi = [self.overall_cache[z] for z in ["t_mask", "t_level_posi", "t_level", "t_next_posi"]]
                t_valid = (t_mask > 0)  # [bs, Lp]
                for _li, _rr in enumerate(att_conf.att_recencies):
                    if _rr > 0:  # recency at each level
                        t_valid &= ((t_next_posi[..., _li:_li+1] < t_level_posi[..., _li] + _rr) | (t_level != _li))
                _s_sel, _s_mask = BK.mask2idx(t_valid, pad=0)  # [bs, Lps]
                if _s_sel.shape[-1] / t_valid.shape[-1] < att_conf.att_shrink_thresh:  # do shrink!!
                    self.overall_cache.update(t_mask=_s_mask, t_level_posi=BK.gather_first_dims(t_level_posi, _s_sel, dim=1), t_level=BK.masked_select_with_pad_by_idxes(t_level, _s_sel, _s_mask, pad=-2))
                    for one_d in self.layered_caches.values():
                        for _k in list(one_d.keys()):
                            one_d[_k] = BK.gather_first_dims(one_d[_k], _s_sel, dim=1)
                    if ZDEBUG():
                        zwarn(f"Shrink cache from {t_valid.shape} to {_s_sel.shape}")
        self.tmp_cache.clear()  # always reset tmp ones!

    def get_cache(self, layer_idx: Optional[int] = None, tmp=False):
        if layer_idx is None:
            return self.tmp_cache if tmp else self.overall_cache
        else:
            return self.layered_caches[layer_idx]  # must be there

    def get(self, key: str, layer_idx: Optional[int] = None, tmp=False, df=None):
        _cache = self.get_cache(layer_idx, tmp=tmp)
        return _cache.get(key, df)

    def update(self, _m=None, layer_idx: Optional[int] = None, tmp=False, concat_dim: int = None, **kwargs):
        _ret = {}
        if _m is None:
            _m = {}
        _m.update(kwargs)
        _cache = self.get_cache(layer_idx, tmp=tmp)
        if len(_cache) == 0 or concat_dim is None:  # directly update
            if not self._no_update:  # update self?
                _cache.update(_m)
            _ret = _m
        else:  # concat if there are existing ones
            for _key, _value in _m.items():
                _value0 = _cache.get(_key)
                _ret[_key] = torch.concat([_value0, _value], dim=concat_dim) if _value0 is not None else _value
                if not self._no_update:  # update self?
                    _cache[_key] = _ret[_key]
        return _ret  # return the updated ones

    # for beam reordering
    def rearrange_cache(self, beam_idx):
        # --
        def _rearrange(_v):  # allow nesting
            if isinstance(_v, list):
                return [_rearrange(z) for z in _v]
            elif isinstance(_v, dict):
                return {k: _rearrange(z) for k, z in _v.items()}
            elif _v is None:
                return None
            else:
                return _v.index_select(0, beam_idx.to(_v.device))
        # --
        # note: no need to care about tmp_cache (since they are forward-specific)!
        for _cc in [self.overall_cache] + list(self.layered_caches.values()):
            for k in list(_cc.keys()):
                _cc[k] = _rearrange(_cc[k])
        # --

    # --
    # prepare inputs
    def prepare_and_update_inputs(self, input_ids, attention_mask, ctok_ids, pad_token_id: int, input_marks=None, **input_kwargs):
        att_conf = self.conf.att_conf
        _cache_overall, _cache_tmp = self.overall_cache, self.tmp_cache
        _level = att_conf.att_ctok_num_of_level
        # extend bs if needed
        if self.has_prev:  # has prev
            _t = _cache_overall["t_mask"]
            cache_bs, new_bs = _t.shape[0], input_ids.shape[0]
            if cache_bs != new_bs:
                assert new_bs > cache_bs and new_bs % cache_bs == 0, f"Bad sizes for (auto) batch expansion: {cache_bs} {new_bs}"
                _reidx = sum([[z] * (new_bs // cache_bs) for z in range(cache_bs)], [])
                t_beam_idx = torch.as_tensor(_reidx).to(_t.device)  # [new_bs] for indexing
                self.rearrange_cache(t_beam_idx)
        # formalize input mask
        attention_mask = (attention_mask > 0).long()  # [bs, Lc]
        # prepare positions and marks: [bs, Lc]
        orig_t_next_posi, orig_t_next_count, orig_add_sink = _cache_overall.get("t_next_posi"), _cache_overall.get("t_next_count"), (not self.has_prev)
        level_posi, level_mark, _sample_sizes = self.prepare_positions(
            input_ids, attention_mask, t_next_posi=orig_t_next_posi, t_next_count=orig_t_next_count, level=_level,
            input_marks=input_marks, sample_seg_ranges=att_conf.sample_seg_ranges, sample_align_batch=att_conf.sample_align_batch)
        # prepare extended inputs with ctoks: [bs, Lce]
        te_input, te_mask, te_level_posi, te_level, t_revidxes = self.prepare_extended_inputs(input_ids, attention_mask, level_posi, level_mark, ctok_ids=ctok_ids, add_sink=(not self.has_prev), att_chunk_size=att_conf.att_chunk_size, pad_token_id=pad_token_id)  # [bs, Lce]
        # update cache
        _cache_tmp["t_revidxes"] = t_revidxes  # [bs, L]
        _cache_tmp["sample_sizes"] = _sample_sizes
        self.update_with_inputs(te_input, te_mask, te_level_posi, te_level, ctok_ids=ctok_ids)
        self.past_length += input_ids.shape[-1]  # add original length
        # --
        # update kwargs
        # note: avoid ALL-PAD seq (which may cause NAN)
        te_mask_forw = _cache_overall["t_mask"].clone()  # [bs, Lp+Lce]
        te_mask_forw[te_mask_forw.sum(-1) <= 0] = 1  # simply make them all 1 for inputs!
        _posi_idx = -1 if att_conf.att_use_full_posi else 0
        ret = {'input_ids': te_input, 'attention_mask': te_mask_forw, 'position_ids': te_level_posi[..., _posi_idx], 'use_cache': False, 'return_dict': True}
        for k, v in input_kwargs.items():
            if k not in ret:  # no need the over-written ones
                if BK.is_tensor(v):  # warning for extra tensor inputs
                    zwarn_once(f"Ignore extra input: {k} = {v.shape}")
                ret[k] = v  # still put them in
        # --
        if ZDEBUG():
            self._zdebug_check_inputs(input_ids, attention_mask, te_input, te_mask_forw, orig_t_next_posi, orig_t_next_count, _cache_overall, _cache_tmp, ctok_ids=ctok_ids, add_sink=orig_add_sink, recencies=att_conf.att_recencies)
        # --
        return ret

    # prepare position-ids and ctok-marks (segmentation) for new inputs (for all the levels)
    # input: [bs, Lc] x2, [bs, level] x2 (for each level, None means 0), List[level-1] of [bs, Lc] -> *[bs, Lc]
    @staticmethod
    def prepare_positions(t_input, t_mask, t_next_posi, t_next_count,
                          level: int, input_marks, sample_seg_ranges: List, sample_align_batch: bool):
        _device = t_input.device
        _bs, _len = BK.get_shape(t_input)  # [bs, Lc]
        _level = level
        if input_marks is None:
            input_marks = [None] * (_level - 1)
        # prepare
        if t_next_posi is None:  # starting from 0
            # note: the addition level for level_posi is a special one indicating full idx (including special tokens)
            t_next_posi = BK.full([_bs, _level+1], 0, dtype=BK.int64, device=_device)  # [bs, level+1]
            t_next_count = t_next_posi.clone()  # [bs, level]
        # get L0 position ids
        t_cumsum = (t_mask.cumsum(-1) - 1).clamp(min=0)
        t_posi = t_next_posi[:, 0:1] + t_cumsum  # [bs, L], L0 position id
        ret_posi, ret_mark = [t_posi], [None]  # no ctok marks for L0
        # later levels
        _sample_sizes = []  # for easier debugging
        for _lidx in range(1, _level):
            _lidxM1 = _lidx - 1  # (segment over) previous level
            tl_mark = input_marks[_lidxM1]  # [bs, Lc], end-of-segment marks
            if tl_mark is None:  # note: simply randomly segment
                size_l, size_r = sample_seg_ranges[_lidxM1]  # sample-range [L, R], seg layer_L-1 to get layer_L
                max_num_seg = (_len + size_l - 1) // size_l  # maxN (max number of segments)
                t_size = BK.randint(size_l, size_r+1, size=[_bs, max_num_seg], device=_device)  # [bs, MaxN]
                if sample_align_batch:
                    t_size[1:] = t_size[0]  # use the same sizes to align batches
                _sample_sizes.append(t_size.clone())
                t_size[:, 0] = (t_size[:, 0] - t_next_count[:, _lidxM1+1]).clamp(min=1)  # subtract previous counts (stored at this level!) (at least one!)
                t_size_points = (t_size.cumsum(-1) - 1).clamp(min=0) + t_next_posi[:, _lidxM1:_lidx]  # [bs, maxN]
                # TODO(+N!): not efficient implementation O(L*L/size_l), but seems light here ...
                # note: check exact position match (assuming we always have continuous position ids!)
                t_eq = (ret_posi[-1].unsqueeze(-2) == t_size_points.unsqueeze(-1)) & (t_mask > 0).unsqueeze(-2)  # [bs, maxN, Lc]
                if ret_mark[-1] is not None:  # also require prev-level is a mark
                    t_eq = t_eq & (ret_mark[-1] > 0).unsqueeze(-2)
                tl_mark0 = BK.mark_first_one(t_eq, dim=-1, rightmost=True)  # [bs, maxN, Lc]
                tl_mark = (tl_mark0.sum(-2) > 0)  # [bs, Lc]
            else:
                _sample_sizes.append(None)
            # get new position ids
            tl_mark = tl_mark.long()  # [bs, Lc]
            tl_posi = t_next_posi[:, _lidx:_lidx+1] + (tl_mark.cumsum(-1) - tl_mark)  # [bs, Lc]
            ret_posi.append(tl_posi)
            ret_mark.append(tl_mark)
        t_posi_full = t_next_posi[:, -1:] + t_cumsum  # [bs, L], full position id
        ret_posi.append(t_posi_full)
        # to debug: pp (lambda rr,ii: BK.stack([z[ii] for z in rr], -1))(ret_posi+ret_mark[1:], 0).tolist()
        return ret_posi, ret_mark, _sample_sizes  # List[level+1] [bs, Lc], List[level] [bs, Lc]

    # add ctok inputs (get combined inputs)
    # input: [bs, Lc] x2, List[level+1?] [bs, Lc] x2 -> (Extended) [bs, Lce, ...]
    @staticmethod
    def prepare_extended_inputs(t_input, t_mask, level_posi: List, level_mark: List, ctok_ids: List, add_sink: bool, att_chunk_size: int, pad_token_id: int):
        _device = t_input.device
        _bs, _len = BK.get_shape(t_input)  # [bs, Lc]
        _level = len(level_posi) - 1
        assert len(level_mark) == _level and len(ctok_ids) == _level
        # --
        # combining inputs: full size FF = S?+1+sumC
        _len_sink = len(ctok_ids[0]) if add_sink else 0  # [S]
        tmp_input = BK.add_paddings(t_input.unsqueeze(-1), ((ctok_ids[0] if _len_sink else []), sum(ctok_ids[1:_level], [])))  # [bs, Lc, FF]
        tmps_posi = [z.unsqueeze(-1).expand_as(tmp_input) for z in level_posi]  # [bs, Lc, FF]
        # construct the valid levels
        tmp_level = []  # tmp sigs: -3=plain(m=1), -2=plain(m=0), -1=sink, 0=invalid, 1=L1, ...
        if _len_sink:
            _l_sink = BK.full_like(t_input.unsqueeze(-1), -1).expand(-1, -1, _len_sink)  # [bs, Lc, S]
            _l_sink[:, 1:, :] = 0  # only adding at the beginning
            tmp_level.append(_l_sink)
        tmp_level.append((- t_mask - 2).unsqueeze(-1))  # [bs, Lc, 1], -3/-2
        for _li, _lt in enumerate(level_mark[1:], 1):
            tmp_level.append((_li * _lt).unsqueeze(-1).expand(-1, -1, len(ctok_ids[_li])))  # [bs, Lc, Ci]
        tmp_level = BK.concat(tmp_level, -1)  # [bs, Lc, FF]
        t_tmp_input = tmp_input.view(_bs, -1)  # [bs, Lc*FF]
        t_tmps_posi = [z.contiguous().view_as(t_tmp_input) for z in tmps_posi]  # [bs, Lc*FF]
        t_tmp_level = tmp_level.view_as(t_tmp_input)  # [bs, Lc*FF]
        # select valid ones (original + new ctok)
        _t_sel, _t_val = BK.mask2idx((t_tmp_level != 0), pad=0)  # [bs, Lce], must have non-pad ones (since plain-invalid is -2 at this time)!
        te_input = BK.masked_select_with_pad_by_idxes(t_tmp_input, _t_sel, _t_val, pad=pad_token_id)  # [bs, Lce]
        te_level_posi = BK.stack([BK.masked_select_with_pad_by_idxes(z, _t_sel, _t_val, pad=-1) for z in t_tmps_posi], -1)  # [bs, Lce, level+1]
        te_level = BK.masked_select_with_pad_by_idxes(t_tmp_level, _t_sel, _t_val, pad=0)  # [bs, Lce]
        te_mask = ((te_level != 0) & (te_level != -2)).long()  # [bs, Lce], excluding all non-valid ones
        # include special tokens and rearrange full position ids
        te_level_posi[..., -1] = te_level_posi[..., :1, -1] + (te_mask.cumsum(-1) - 1).clamp(min=0)  # [bs, Lce]
        # --
        # some more modifications
        t_revidxes, _tmp_valid = BK.mask2idx(((te_level == -2) | (te_level == -3)), pad=0)  # [bs, L]
        if ZDEBUG():
            assert BK.get_shape(t_revidxes) == BK.get_shape(t_input) and (_tmp_valid == 1).all(), "Reverse must be full!"
        te_level[te_level == 0] = -2
        te_level[te_level == -3] = 0  # final sigs: -2=plain(m=0), -1=sink, 0=L0(plain(m=1)), 1=L1, ...
        # to debug: pp (lambda rr,ii: BK.concat([z[ii] for z in rr], -1))([te_input.unsqueeze(-1), te_mask.unsqueeze(-1), te_level.unsqueeze(-1), te_level_posi], 0)[:512].tolist()
        # --
        # pad according to chunking size (pad to the right, thus no changes to t_revidxes)
        _elen = te_input.shape[-1]
        if _elen > att_chunk_size and _elen % att_chunk_size != 0:
            _pad_num = att_chunk_size - (_elen % att_chunk_size)
            _elen2 = _elen + _pad_num
            _p_kwargs = {"dtype": BK.int64, "device": _device}
            _pte_input, _pte_mask, _pte_level_posi, _pte_level = BK.full([_bs, _elen2], pad_token_id, **_p_kwargs), BK.full([_bs, _elen2], 0, **_p_kwargs), BK.full([_bs, _elen2, _level+1], -1, **_p_kwargs), BK.full([_bs, _elen2], -2, **_p_kwargs)  # [bs, elen2, ...]
            _pte_input[:, :_elen] = te_input
            _pte_mask[:, :_elen] = te_mask
            _pte_level_posi[:, :_elen] = te_level_posi
            _pte_level[:, :_elen] = te_level
            te_input, te_mask, te_level_posi, te_level = _pte_input, _pte_mask, _pte_level_posi, _pte_level
        # --
        return te_input, te_mask, te_level_posi, te_level, t_revidxes

    # update cache with new inputs
    # input: [bs, Lce] ...
    def update_with_inputs(self, te_input, te_mask, te_level_posi, te_level, ctok_ids: List):
        att_conf = self.conf.att_conf
        _device = te_input.device
        _cache_overall, _cache_tmp = self.overall_cache, self.tmp_cache
        _level = att_conf.att_ctok_num_of_level
        _has_prev = self.has_prev
        # --
        # pre-calculated masks and indexes (note: assuming all the positions are monotonically increasing)
        # - chunk the current input
        _bs, _elen = BK.get_shape(te_input)  # shape of the extended inputs
        _chunk_size = min(att_conf.att_chunk_size, _elen)  # simply take _elen if smaller
        assert _elen % _chunk_size == 0, f"(Extended) input length here must be a multiplier of chunk size, but get {_elen} % {_chunk_size} != 0"
        _piece = _elen // _chunk_size
        # - max and min position for each chunk, valid keys should be min<= and <=max
        te_reshaped_level_posi = te_level_posi.view([_bs, _piece, _chunk_size, _level+1])  # [bs, piece, csize, level+1]
        te_reshaped_level = te_level.view([_bs, _piece, _chunk_size])  # [bs, piece, csize]
        if _piece > 1:
            t_max_posi, _ = te_reshaped_level_posi.max(-2)  # [bs, piece, level+1]
            t_min_posi, _ = BK.where((te_mask > 0).unsqueeze(-1), te_level_posi, Constants.INT_PRAC_MAX).view([_bs, _piece, _chunk_size, _level+1]).min(-2)
            t_min_posi = t_min_posi - BK.as_tensor(att_conf.att_recencies + [0]).to(t_min_posi) + 1  # [bs, piece, level+1]
        else:  # simply use all the kv, no further re-selecting to make things simpler
            t_max_posi = t_min_posi = None
        # --
        # - concat new caches masks and positions to the past (stored in overall cache)
        _one_input_info = {"t_mask": te_mask, "t_level_posi": te_level_posi, "t_level": te_level}
        for kk0, vv0 in _one_input_info.items():  # concat with the past
            _cache_overall[kk0] = vv0 if kk0 not in _cache_overall else BK.concat([_cache_overall[kk0], vv0], dim=1)  # [bs, Lp+Lce, ...]
        # --
        # - prepare local indexes and masks for each level
        _tmp_input_info = {
            "chunk_shape": [_bs, _piece, _chunk_size],  # chunked input shape
            "chunk_shapeR": [_bs*_piece, _chunk_size],  # reorganized chunked input shape
        }
        _cache_tmp["input_info"] = _tmp_input_info  # only useful for this one forward
        _full_mask, _full_level_posi, _full_level = [_cache_overall[z] for z in ["t_mask", "t_level_posi", "t_level"]]  # [bs, Lff, ...]
        _full_kv_len = int(_full_mask.shape[-1])  # Lff = Lp+Lce
        # TODO(+N!): not efficient implementation O(L/chunk_size * L_cache), but seems light here ...
        if _piece <= 1:  # no need to re-select if only one piece!
            _lsel_idx, _lsel_mask = None, None
        else:
            all_lsel_idx, all_lsel_mask = [], []  # [bs, piece, *Ls] selections for all the levels
            for _li in range(-1, _level):  # -1=sink, 0=L0(plain input), 1=L1, ...
                _li_recency = 0 if (_li == -1) else att_conf.att_recencies[_li]
                _li_lsel = (_full_mask > 0).unsqueeze(-2).expand(-1, _piece, -1)  # [bs, piece, Lff], ignore the invalid ones!
                _li_lsel = _li_lsel & (_full_level == _li).unsqueeze(-2)  # [bs, piece, Lff], selecting level!
                if _li_recency > 0:
                    _li_lposi = _full_level_posi[..., _li].unsqueeze(-2)  # [bs, 1, Lff]
                    _li_lselR = (t_min_posi[..., _li].unsqueeze(-1) <= _li_lposi) & (_li_lposi <= t_max_posi[..., _li].unsqueeze(-1))  # [bs, piece, Lff], local window
                    _li_lsel = _li_lsel & _li_lselR  # [bs, piece, Lff]
                _li_lsel_idx, _li_lsel_mask = BK.mask2idx(_li_lsel, pad=_full_kv_len)  # [bs, piece, Lsi], (pad a large value for sorting)
                all_lsel_idx.append(_li_lsel_idx)
                all_lsel_mask.append(_li_lsel_mask)
            # concat and sort all the kv indexes (except the invalid ones, there should not be any overlaps between levels)
            _lsel_idx0, _lsel_mask0 = BK.concat(all_lsel_idx, dim=-1), BK.concat(all_lsel_mask, dim=-1)  # [bs, piece, Lkv]
            _lsel_idx = _lsel_idx0.sort(dim=-1).values  # [bs, piece, Lkv]
            _lsel_mask = (_lsel_idx < _full_kv_len).long()  # [bs, piece, Lkv]
            _lsel_idx[_lsel_mask == 0] = 0  # reset special idx
            if ZDEBUG():
                zwarn(f"Concat inputs: {_lsel_idx.shape}={[z.shape for z in all_lsel_idx]}")
            _lsel_max_size = _lsel_mask.sum(-1).max().item()  # Lkv(shrinked)
            if _lsel_max_size < _lsel_idx.shape[-1]:
                _lsel_idx, _lsel_mask = _lsel_idx[..., :_lsel_max_size].contiguous(), _lsel_mask[..., :_lsel_max_size].contiguous()  # [bs, piece, Lkv]
            if _lsel_max_size >= _full_kv_len * att_conf.att_isel_thresh:  # simply no index-select since not differ too much!
                _lsel_idx, _lsel_mask = None, None
        # - prepare attention masks (causal att & mask & recency)
        if _lsel_idx is None:  # use all KV cache
            _lsel_idx_reshaped = None
            _lsel_mask = _full_mask.unsqueeze(1)  # [bs, 1, Lkv]
            _lsel_level_posi = _full_level_posi.unsqueeze(1)  # [bs, 1, Lkv, level+1]
            _lsel_level = _full_level.unsqueeze(1)  # [bs, 1, Lkv]
        else:
            _lsel_idx_reshaped = _lsel_idx.view(_bs, -1)  # [bs, piece*Lkv], convenient for gathering
            _lsel_level_posi = BK.gather_first_dims(_full_level_posi, _lsel_idx_reshaped, dim=1).view(BK.get_shape(_lsel_idx) + [-1])  # [bs, 1, Lkv, level+1]
            _lsel_level = _full_level.gather(-1, _lsel_idx_reshaped).view_as(_lsel_idx)  # [bs, piece, Lkv]
        _kv_att_mask = (te_reshaped_level_posi[..., -1].unsqueeze(-1) >= _lsel_level_posi[..., -1].unsqueeze(-2))  # [bs, piece, Lq, Lkv], causal
        _kv_att_mask &= (_lsel_mask.unsqueeze(-2) > 0)  # item mask
        for _li in range(0, _level):  # recency (no need to exclude -1=sink ones since they are always included!)
            _li_recency = att_conf.att_recencies[_li]
            if _li_recency > 0:  # recency at each level
                _kv_att_mask &= ((te_reshaped_level_posi[..., _li].unsqueeze(-1) < _lsel_level_posi[..., _li].unsqueeze(-2) + _li_recency) | (_lsel_level.unsqueeze(-2) != _li))
        _kv_att_mask_reshaped = _kv_att_mask.view(_bs*_piece, _chunk_size, -1)
        _tmp_input_info.update(kv_idx=_lsel_idx_reshaped, kv_att_mask=_kv_att_mask_reshaped)  # [bs, piece*Lkv], [bs*piece, Lq, Lkv]
        # --
        # update next posi/count [next first one's posi id & this id's count previously] & mask
        # -- (use full history to update; note: assuming recency covers all related info!!)
        t_next_posi, _ = _full_level_posi.max(-2)  # [bs, level+1], simply max one as previous position; note: no padding position will be larger than valid ones!
        _level_check = BK.as_tensor(list(range(_level + 1)), device=_device)
        _t_mark_hit = ((_full_level_posi == t_next_posi.unsqueeze(-2)) & (_full_mask > 0).unsqueeze(-1) & (_full_level.unsqueeze(-1) == _level_check))  # [bs, Lce, level+1]
        _t_mark_hit2 = (_t_mark_hit.sum(-2) > 0).long()  # [bs, level+1], any such hit!
        _t_mark_hit2[..., -1] = (te_mask.sum(-1) > 0).long()  # +1 for full_position_id if not full-invalid! (note: use current mask here!)
        t_next_posi += _t_mark_hit2  # already get an ending, +1 for the next!
        _t_next_hit = ((_full_level_posi == t_next_posi.unsqueeze(-2)) & (_full_mask > 0).unsqueeze(-1) & (_full_level.unsqueeze(-1) == (_level_check-1)))  # [bs, Lce, level+1], hit from prev level
        _level_ctok_num = BK.as_tensor([1] + ([1] if len(ctok_ids)>1 else []) + [len(z) for z in ctok_ids[1:_level-1]] + [1], device=_device)
        t_next_count = _t_next_hit.sum(-2) // _level_ctok_num  # [bs, level+1], hit count from prev level
        t_next_count[:, 0] = 0  # no hit for L0
        t_next_count[:, -1] = 0  # no hit for full_position_id
        _cache_overall.update(t_next_posi=t_next_posi, t_next_count=t_next_count)
        # --

    # -- note: for debugging!
    # check inputs by comparing against python-version
    def _zdebug_check_inputs(self, t0_input, t0_mask, te_input, te_mask, orig_t_next_posi, orig_t_next_count, cache_overall, cache_tmp, ctok_ids, add_sink, recencies):
        att_conf = self.conf.att_conf
        _level = att_conf.att_ctok_num_of_level
        _bs, _len0 = BK.get_shape(t0_input)
        _bs2, _lenE = BK.get_shape(te_input)
        te_mask = te_mask[:, -_lenE:]
        assert _bs == _bs2 and BK.get_shape(t0_mask) == BK.get_shape(t0_input) and BK.get_shape(te_mask) == BK.get_shape(te_input)
        kv_idx, kv_att_mask = cache_tmp["input_info"]["kv_idx"], cache_tmp["input_info"]["kv_att_mask"]  # [bs, piece, Lkv], [bs, piece*Lq, Lkv]
        _chunk_shape = cache_tmp["input_info"]["chunk_shape"]  # [bs, piece, Lq]
        if kv_idx is not None:
            kv_idx = kv_idx.view(_chunk_shape[:-1] + [-1])
        kv_att_mask = kv_att_mask.view([_chunk_shape[0], _chunk_shape[1]*_chunk_shape[2], -1])
        # -- prepare_positions & inputs
        for _bidx in range(_bs):
            # get inputs
            _ids = t0_input[_bidx][t0_mask[_bidx] > 0].tolist()  # [L]
            _next_posi = [0]*(_level+1) if orig_t_next_posi is None else orig_t_next_posi[_bidx].tolist()  # [level+1]
            _next_count = [0]*(_level+1) if orig_t_next_count is None else orig_t_next_count[_bidx].tolist()  # [level+1]
            # sample segs
            marks = []
            for _li in range(1, _level):
                _sizes = cache_tmp["sample_sizes"][_li-1][_bidx].tolist()
                _sizes[0] = max(1, _sizes[0] - _next_count[_li])  # at least one!
                _prev_idxes = list(range(len(_ids))) if _li == 1 else marks[-1]
                marks.append([])
                _pointer = 0
                for _one_size in _sizes:
                    _pointer = _pointer + _one_size
                    assert _pointer >= 0
                    if _pointer <= len(_prev_idxes):
                        marks[-1].append(_prev_idxes[_pointer-1])
                    else:
                        break
            # new ones
            _new_ids, _new_posi, _new_level = [], [], []
            if add_sink:  # add sink tokens
                _new_ids.extend(ctok_ids[0])
                for ii in range(len(ctok_ids[0])):
                    _new_posi.append(_next_posi.copy())
                    _next_posi[-1] += 1  # forward one
                    _next_count[-1] = 0
                _new_level.extend([-1] * len(ctok_ids[0]))
            marks_set = [None] + [set(z) for z in marks]
            for _one_ii, _one_token in enumerate(_ids):
                _new_ids.append(_one_token)
                _new_posi.append(_next_posi.copy())
                _next_posi[-1] += 1  # forward one
                _next_count[-1] = 0
                _new_level.append(0)
                _prev_level_end = True
                _up_levels = [0]
                for _li in range(1, _level):
                    if _one_ii in marks_set[_li]:  # add ctok
                        assert _prev_level_end, "Cannot get higher marks without lower ones!"
                        _new_ids.extend(ctok_ids[_li])
                        for ii in range(len(ctok_ids[_li])):
                            _new_posi.append(_next_posi.copy())
                            _next_posi[-1] += 1  # forward one
                            _next_count[-1] = 0
                        _new_level.extend([_li] * len(ctok_ids[_li]))
                        _up_levels.append(_li)
                    else:
                        _next_count[_li] += int(_prev_level_end)  # add only ending one of the previous level!
                        _prev_level_end = False
                for _li in _up_levels:  # update at the end since higher level endings still use the old ones!
                    _next_posi[_li] += 1  # forward one
                    _next_count[_li] = 0
            # check
            assert te_input[_bidx][te_mask[_bidx] > 0].tolist() == _new_ids
            assert cache_overall["t_level_posi"][_bidx, -_lenE:][te_mask[_bidx] > 0].tolist() == _new_posi
            assert cache_overall["t_level"][_bidx, -_lenE:][te_mask[_bidx] > 0].tolist() == _new_level
            assert cache_overall["t_next_posi"][_bidx].tolist() == _next_posi
            assert cache_overall["t_next_count"][_bidx].tolist() == _next_count
            # attention
            all_kv_posi, all_kv_level = cache_overall["t_level_posi"][_bidx].tolist(), cache_overall["t_level"][_bidx].tolist()
            _len_full = len(all_kv_level)
            _tmp_new_ids = list(reversed(_new_ids))
            for _qii in range(_lenE):  # check each token's attention (note: use padded seq!)
                if not te_mask[_bidx, _qii]:
                    continue  # skip invalid one
                assert _tmp_new_ids.pop() == te_input[_bidx, _qii].item()
                _qii_full = _qii + (_len_full - _lenE)  # position in full
                _should_hit_indexes = []
                for _kii in range(0, _qii_full+1):  # causal mask (until this one)
                    _k_level = all_kv_level[_kii]
                    if _k_level < -1: continue  # ignore invalid
                    if _k_level == -1:  # always add sink!
                        _should_hit_indexes.append(_kii)
                    elif recencies[_k_level] <= 0 or all_kv_posi[_qii_full][_k_level] - all_kv_posi[_kii][_k_level] < recencies[_k_level]:
                        _should_hit_indexes.append(_kii)
                _t_kv_idx = kv_idx[_bidx, _qii // _chunk_shape[-1]] if kv_idx is not None else BK.arange(_len_full).to(kv_att_mask.device)
                _actual_hit_indexes = _t_kv_idx[kv_att_mask[_bidx, _qii] > 0].tolist()
                assert _should_hit_indexes == _actual_hit_indexes
            assert len(_tmp_new_ids) == 0
        # --
        zlog("Passed _zdebug_check_inputs!")
    # --

class MyCacheLayeredView:
    def __init__(self, base_cache: MyCache, layer_idx: int):
        self.base_cache = base_cache
        self.layer_idx = layer_idx

    def get_cache(self):
        return self.base_cache.get_cache(layer_idx=self.layer_idx)

    def get(self, key: str, **kwargs):
        return self.base_cache.get(key, layer_idx=self.layer_idx, **kwargs)

    def update(self, *args, **kwargs):
        return self.base_cache.update(*args, layer_idx=self.layer_idx, **kwargs)

# --
class MyAttention(nn.Module):
    def __init__(self, config: MyAttentionConf, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.my_cache: MyCacheLayeredView = None  # to be assigned
        self._attn_func = getattr(self, f"_attn_{config.att_func}")
        zlog(f"Get attn_func = {self._attn_func} for MyAttention")
        # --
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.num_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        # init rope
        if config.att_use_rope:
            _rope_scale = 1.
            self.rotary_ndims = int(self.head_dim * config.rotary_pct)
            self.rotary_partial = (self.rotary_ndims != self.head_dim)
            self.rotary_emb = MyRope(self.rotary_ndims, base=config.rope_theta, scaling_factor=_rope_scale)
        else:
            self.rotary_emb = None

    def _apply_rope(self, query_states, key_states, position_ids):
        if self.rotary_partial:  # from modeling_gpt_neox
            _rdim = self.rotary_ndims
            query_rot = query_states[..., : _rdim]
            key_rot = key_states[..., : _rdim]
            query_pass = query_states[..., _rdim:]
            key_pass = key_states[..., _rdim:]
        else:
            query_rot, key_rot = query_states, key_states
            query_pass = key_pass = None
        cos_sin = self.my_cache.base_cache.get("cos_sin", tmp=True)
        if cos_sin is None:   # calculate and store tmp!
            cos_sin = self.rotary_emb(query_states, position_ids)
            self.my_cache.base_cache.update(cos_sin=cos_sin, tmp=True)
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
        if query_pass is not None:
            query = torch.cat((query, query_pass), dim=-1)
            key = torch.cat((key, key_pass), dim=-1)
        return query, key

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[MyCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # from gpt2
        layer_past: Optional[MyCache] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        att_conf = self.config
        _level = att_conf.att_ctok_num_of_level
        _tmp_input_info = self.my_cache.base_cache.get("input_info", tmp=True)  # pre-computed ones
        # --
        # prepare
        assert past_key_value is None and layer_past is None and (not use_cache), "No external cache for this mode!!"
        # assert position_ids is not None  # some modules (e.g., those with abs position) do not need this
        if self.config.sliding_window is not None:
            zwarn_once(f"Currently sliding_window={self.config.sliding_window} is ignored!")
        # --
        # projection
        bsz, q_len = list(hidden_states.shape)[:-1]  # [*, L]
        query_states = self.q_proj(hidden_states)  # [*, L, D]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # [*, H, L, Dh]
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # note: transpose later ...
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)  # [*, L, H, Dh]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)  # [*, L, H', Dh]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        # --
        # apply rope & concat past
        if self.rotary_emb is not None:
            query_states, key_states = self._apply_rope(query_states, key_states, position_ids)
        _new_kv = self.my_cache.update(k=key_states, v=value_states, concat_dim=1)  # [bs, Lp+L, H', Dh]
        _full_k, _full_v = _new_kv["k"], _new_kv["v"]
        # prepare attention
        _chunk_shape = _tmp_input_info["chunk_shape"]  # [bs, piece, chunk_size]
        _chunk_shapeR = _tmp_input_info["chunk_shapeR"]  # [bs*piece, chunk_size]
        _kv_idx = _tmp_input_info["kv_idx"]
        if _kv_idx is None:  # no need to re-index
            t_key, t_value = BK.unsqueeze_expand(_full_k, _chunk_shape[1], dim=1).contiguous(), BK.unsqueeze_expand(_full_v, _chunk_shape[1], dim=1).contiguous()  # [bs, piece, chunk_size, H', Dh]
        else:  # re-selecting!
            t_key, t_value = BK.gather_first_dims(_full_k, _kv_idx, dim=1), BK.gather_first_dims(_full_v, _kv_idx, dim=1)  # [bs, piece*Lkv, H', Dh]
        _one_shape = [_chunk_shapeR[0], -1, self.num_key_value_heads, self.head_dim]
        t_key, t_value = t_key.view(_one_shape), t_value.view(_one_shape)  # [bs*piece, Lkv, H', Dh]
        _kv_att_mask = _tmp_input_info["kv_att_mask"].unsqueeze(1)  # [bs*piece, 1, chunk_size, Lkv]
        _NEG = -10000.  # note: enough as mask and small enough for fp16!
        t_att_mask = BK.where((_kv_att_mask > 0), 0., _NEG).to(t_key)  # fill NEG value!
        # repeat kv group
        t_query = query_states.view(_chunk_shapeR + [self.num_heads, self.head_dim]).transpose(1, 2)  # [bs*piece, H, chunk_size, Dh]
        t_key = repeat_kv(t_key.transpose(1, 2), self.num_key_value_groups)  # [bs*piece, H, Sum_Lk, Dh]
        t_value = repeat_kv(t_value.transpose(1, 2), self.num_key_value_groups)
        # --
        # attention
        if ZDEBUG():
            zwarn_once(f"Calculate attention: hidden_shape={hidden_states.shape} attn_shape={t_att_mask.shape}")
        attn_drop_rate = self.attention_dropout if self.training else 0.
        attn_output, attn_weights = self._attn_func(t_query, t_key, t_value, t_att_mask, attn_drop_rate)  # [bs*piece, H, chunk_size, Dh]
        # final output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)  # [*, L, D]
        attn_output = self.o_proj(attn_output)
        assert not output_attentions, "No support for returning attn_weights!"
        attn_weights = None  # no support for att weight returning!
        # no return past_key_values!
        if self.config._orig_cls_type == "gpt2":
            final_ret = (attn_output, None) + ((attn_weights, ) if output_attentions else ())
        else:
            final_ret = attn_output, attn_weights, None
        return final_ret

    # --
    # attn implementations

    def _attn_plain(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)  # [*, H, Lq, Lk]
        if attention_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attn_drop_rate, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output, attn_weights

    def _attn_flash2(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float):
        from flash_attn import flash_attn_func
        # note: simply ignore attention_mask here, might be errored if having special attention_mask!!
        if attention_mask is not None:
            zwarn_once("Ignore attention_mask in _attn_flash2!")
        attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), dropout_p=attn_drop_rate, causal=self.is_causal)
        return attn_output.transpose(1, 2), None

    def _attn_sdpa(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float):
        if query_states.device.type == "cuda" and attention_mask is not None:  # to avoid potential sdpa bug?
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True, dropout_p=attn_drop_rate)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=attn_drop_rate)
        return attn_output, None
