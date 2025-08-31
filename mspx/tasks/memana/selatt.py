#

# selective attention

from functools import partial
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from mspx.nn import BK
from mspx.core import DataPadder
from mspx.utils import Conf, GET_ENV_VAR, GlobalObjects

# --
class SelAttConf(Conf):
    def __init__(self):
        self.enabled = False  # another outside flag?
        # how to score each group (dim=Lk)
        self.score_topk = 5.  # <=0 means take all
        # how to aggregate the scores
        self.aggr_dims = ""  # [layer,head,seq]
        self.aggr_topk = 5  # top-K for the aggregation
        self.aggr_momentum = 0.0  # aggregation momentum
        # how to select each group
        self.sel_topk = 0.  # select top-what group?

class SelAttHelper:
    def __init__(self, conf: SelAttConf):
        self.conf = conf
        # --
        self.current_enabled = False  # status
        self.current_context_pack = None
        # --
        # storages for group_attn
        self.group_attn_pass1 = []
        self.group_attn_pass2 = None
        # --

    def modify_model(self, model):
        if self.conf.enabled:  # no converting at all!
            BK.replace_module(model, (lambda m, p: p and m.__class__.__name__ == "MymAttention"), self.convert_module, inplace=True)

    def convert_module(self, module, path):
        assert hasattr(module, "_attn_func")
        BK.setattr_borrow(module, "_attn_func_old", getattr(module, "_attn_func"))  # store old one
        BK.setattr_borrow(module, "_attn_func", partial(self._attn_sel, module=module), assert_nonexist=False)  # new one

    def clear_group_attn(self):
        self.group_attn_pass1 = []
        self.group_attn_pass2 = None

    # --
    @contextmanager
    def env_use_sel_attn(self, enabled=True):
        old_v = self.current_enabled
        self.current_enabled = enabled
        try:
            yield None
        finally:
            self.current_enabled = old_v

    @contextmanager
    def env_use_context(self, inputs, **kwargs):
        # --
        assert len(inputs) == 1, "For simplicity assuming bs=1 for this mode!"
        _lpi = inputs[0]
        assert sum(len(z)>1 for z in _lpi.piece_lens) <= 1, "For simplicity assuming only one multi-piece"
        _max_item_id = max(_lpi.item_ids, default=0)
        _matched_idx = (torch.as_tensor(_lpi.item_ids).unsqueeze(-1) == torch.arange(1, _max_item_id+1))  # [L_ctx, Lm]
        self.current_context_pack = (_max_item_id, _matched_idx, kwargs)
        # --
        try:
            yield None
        finally:
            self.current_context_pack = None

    def finish_one_layer(self, module, query_states, key_states, value_states, attention_mask, attention_logit, attention_prob, attention_output):
        # --
        # part 1
        _amask = (attention_mask >= 0.)  # [*, H, Lq, Lk]
        _attabs = torch.where(_amask, attention_logit, 0.).float().abs().sum(-1) / _amask.float().sum(-1)
        GlobalObjects.get(f"_L{module.layer_idx:02d}_attabs", []).append(BK.get_value(_attabs))  # [*, H, Lq], average attention scale
        GlobalObjects.get(f"_L{module.layer_idx:02d}_attQ", []).append(BK.get_value(query_states.float().abs().mean(-1)))  # [*, H, Lq], average attention scale
        _l_attK, _l_attV = GlobalObjects.get(f"_L{module.layer_idx:02d}_attK", []), GlobalObjects.get(f"_L{module.layer_idx:02d}_attV", [])
        _len_query = query_states.shape[-2]
        if _l_attK:  # slicing
            _l_attK.append(BK.get_value(key_states[..., -_len_query:, :].float().abs().mean(-1)))
        else:  # all
            _l_attK.append(BK.get_value(key_states.float().abs().mean(-1)))  # [*, H, Lkv], average attention scale
        if _l_attV:  # slicing
            _l_attV.append(BK.get_value(value_states[..., -_len_query:, :].float().abs().mean(-1)))
        else:  # all
            _l_attV.append(BK.get_value(value_states.float().abs().mean(-1)))  # [*, H, Lkv], average attention scale
        # part 2
        _attent = - (torch.where(attention_prob>0, attention_prob.log(), 0.) * attention_prob).float().sum(-1)
        GlobalObjects.get(f"_L{module.layer_idx:02d}_attent", []).append(BK.get_value(_attent))  # [*, H, Lq], entropy
        # part 3
        _attout = attention_output.float().abs().mean(-1)
        GlobalObjects.get(f"_L{module.layer_idx:02d}_attout", []).append(BK.get_value(_attout))  # [*, H, Lq], L2 scale
        # part 4: L2-norm of Q K and cos-sim
        _l2Q, _l2K = (query_states.float()**2).sum(-1).sqrt(), (key_states.float()**2).sum(-1).sqrt()  # [*, H, L??]
        GlobalObjects.get(f"_L{module.layer_idx:02d}_l2Q", []).append(BK.get_value(_l2Q))  # [*, H, Lq], L2
        _l_l2K = GlobalObjects.get(f"_L{module.layer_idx:02d}_l2K", [])
        if _l_attK:  # slicing
            _l_l2K.append(BK.get_value(_l2K[..., -_len_query:]))
        else:  # all
            _l_l2K.append(BK.get_value(_l2K))  # [*, H, Lkv], average attention scale
        _lcos0 = attention_logit.float() / _l2K.unsqueeze(-2) / _l2Q.unsqueeze(-1)  # [*, H, Lq, Lk]
        _lcos = torch.where(_amask, _lcos0, 0.).sum(-1) / _amask.float().sum(-1)
        GlobalObjects.get(f"_L{module.layer_idx:02d}_attcos", []).append(BK.get_value(_lcos))  # [*, H, Lq], L2
        # breakpoint()
        # --

    def finish_one_inst(self):
        _full_keys = [z[2:] for z in GlobalObjects.keys() if z.startswith("_L")]
        _layers = sorted(set([z.rsplit("_", 1)[0] for z in _full_keys]))
        _keys = sorted(set([z.rsplit("_", 1)[1] for z in _full_keys]))
        # wrap up one inst for GlobalObjects
        for key in _keys:
            _ts = []
            _seqlens = None
            for layer in _layers:
                items = GlobalObjects.get(f"_L{layer}_{key}", ensure_existing=True)
                if _seqlens is None:
                    _seqlens = [z.shape[-1] for z in items]
                else:
                    assert _seqlens == [z.shape[-1] for z in items]
                _ts.append(np.concatenate(items, -1).mean(0))  # [head, L]
            _tt = np.stack(_ts, 0)  # [layer, head, Lseq]
            GlobalObjects.get(key, []).append((_seqlens, _tt))
        for kk in _full_keys:  # remove things
            GlobalObjects.pop(f"_L{kk}")
        # breakpoint()
        # --

    def finish_all_insts(self):
        # average over all insts
        ret = {}
        for kk, vvs in GlobalObjects.items():
            _max_len = max(_tt.shape[-1] for _, _tt in vvs)
            _shape0 = list(vvs[0][-1].shape)
            _shape0[-1] = _max_len
            _arr_sum, _arr_count = np.full(_shape0, 0., dtype=vvs[0][-1].dtype), np.full(_shape0, 0., dtype=vvs[0][-1].dtype)
            for _, _tt in vvs:
                _len = _tt.shape[-1]
                _arr_sum[..., :_len] += _tt
                _arr_count[..., :_len] += 1.
            ret[kk] = _arr_sum / _arr_count
        return ret

    def _attn_sel(self, query_states, key_states, value_states, attention_mask, attn_drop_rate: float, module=None):
        if not (self.conf.enabled and (self.current_enabled or GET_ENV_VAR("ZCOLVAL"))):
            return module._attn_func_old(query_states, key_states, value_states, attention_mask, attn_drop_rate)
        # --
        _conf = self.conf
        assert module.config.attn_logit_softcapping is None, "attn_logit_softcapping not implemented for this mode!"
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * module.scaling  # [*, H, Lq, Lk]
        if attention_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + attention_mask
        # --
        # softmax and masking
        if GET_ENV_VAR("ZCOLVAL"):
            attn_logits = attn_weights  # temp storage
        else:
            attn_logits = None
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [*, H, Lq, Lk]
        _no_change = False
        if self.current_enabled:  # enable selatt
            _max_item_id, _matched_idx, _extra_vals = self.current_context_pack
            _no_change = (_max_item_id > 1) and (self.group_attn_pass2 is None) and ("2pass" in _conf.aggr_dims)  # in this case, we do not actually store things
            if _max_item_id > 1 and _conf.sel_topk != 0:  # masking and recalculate
                _matched_idx = _matched_idx.to(attn_weights)  # [Lk', Lm]
                # =====
                if self.group_attn_pass2 is None:
                    GAP2_CHUNK_SIZE = int(GET_ENV_VAR("GAP2_CHUNK_SIZE", df=0))
                    if GAP2_CHUNK_SIZE > 0:
                        _group_scores = []
                        for q_start in range(0, attn_weights.shape[-2], GAP2_CHUNK_SIZE):
                            _gs = attn_weights[..., q_start:q_start+GAP2_CHUNK_SIZE, :len(_matched_idx)].unsqueeze(-1) * _matched_idx  # [*, H, LqC, Lk, Lm]
                            if _conf.score_topk > 0:  # note: take all if setup a large topk
                                _gs = _gs.topk(min(int(_conf.score_topk), _gs.shape[-2]), dim=-2)[0].sum(-2)  # [*, H, LqC, Lm]
                            else:  # otherwise use average
                                _gs = (_gs.float().sum(-2) / _matched_idx.sum(0)).to(_gs)  # [*, H, LqC, Lm]
                            _group_scores.append(_gs)
                        _group_score = torch.cat(_group_scores, -2)  # [*, H, Lq, Lm]
                    else:
                        _group_score = attn_weights[..., :len(_matched_idx)].unsqueeze(-1) * _matched_idx  # [*, H, Lq, Lk, Lm]
                        if _conf.score_topk > 0:  # note: take all if setup a large topk
                            _group_score = _group_score.topk(min(int(_conf.score_topk), _group_score.shape[-2]), dim=-2)[0].sum(-2)  # [*, H, Lq, Lm]
                        else:  # otherwise use average
                            _group_score = (_group_score.float().sum(-2) / _matched_idx.sum(0)).to(_group_score)  # [*, H, Lq, Lm]
                    # _normed_group_prob = _group_score / _group_score.sum(-1, keepdim=True)  # [*, H, Lq, Lm]
                    _normed_group_prob = _group_score  # [*, H, Lq, Lm], no norm!
                    # --
                    # aggr
                    if "2pass" in _conf.aggr_dims:  # need to handle this at outside
                        self.group_attn_pass1.append(_normed_group_prob)
                    elif _conf.aggr_dims:  # other dims?
                        assert "layer" not in _conf.aggr_dims, "Cannot handle layer within one att layer!"
                        _normed_group_prob = BK.aggr_tensor(_normed_group_prob, [i for i, n in zip([-3, -2], ["head", "seq"]) if n in _conf.aggr_dims], int(_conf.aggr_topk))
                else:
                    # note: directly use this, since it must be aggregated over layers
                    _layer_idx = 0 if self.group_attn_pass2.shape[1] <= 1 else module.layer_idx  # allow broadcasting
                    _normed_group_prob = self.group_attn_pass2[:, _layer_idx]
                # --
                # what if we force oracle selection
                # (note: only for debugging!) _group_score[..., _extra_vals["oracle_idxes"]] = 100.
                # l0, l1 = [int(z) for z in os.environ.get("ORACLE_LAYERS", "0,0").split(",")]
                # if module.layer_idx >= l0 and module.layer_idx < l1:
                #     _group_score[..., _extra_vals["oracle_idxes"]] = 100.
                # print(f"Layer-{module.layer_idx}: A={(_group_score / _group_score.sum(-1, keepdim=True)).flatten().topk(10)[0]} O={(_group_score[..., _extra_vals['oracle_idxes'][0][0]] / _group_score.sum(-1)).max()}")
                # breakpoint()
                # --
                if not _no_change:
                    _sel_topk = _conf.sel_topk
                    if _sel_topk < 1.:
                        _normed_group_logprob = (_normed_group_prob + 1e-5).log()  # [*, H, Lq, Lm]
                        _trg_group_prob = torch.softmax(_normed_group_logprob.float()/_sel_topk, -1).to(_normed_group_logprob)  # [*, H, Lq, Lm], sharper distribution
                        _trg_scale = _trg_group_prob / torch.matmul(attn_weights[..., :len(_matched_idx)], _matched_idx)  # [*, H, Lq, Lm], attention scale
                        _trg_scale2 = torch.matmul(_trg_scale, _matched_idx.transpose(0, 1))  # [*, H, Lq, Lk']
                        attn_weights[..., :_trg_scale2.shape[-1]] *= _trg_scale2
                        # breakpoint()
                    else:
                        _, _sel_group_idx = _normed_group_prob.topk(int(_sel_topk), dim=-1)  # [*, H, Lq, K]
                        _clear_mask = (_matched_idx.transpose(0, 1)[_sel_group_idx].sum(-2) <= 0) & (_matched_idx.sum(-1) > 0)  # [*, H, Lq, Lk']
                        attn_weights[..., :_clear_mask.shape[-1]] *= (~_clear_mask).to(attn_weights)  # clear things
                        attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).clamp(min=1e-5)  # renorm!
                        # breakpoint()
        # --
        if GET_ENV_VAR("ZCALC_ENTROPY") and not _no_change:  # calculate entropy
            _attent = - (torch.where(attn_weights>0, attn_weights.log(), 0.) * attn_weights).float().sum(-1).mean()  # []
            GlobalObjects.get(f"_L{module.layer_idx:02d}_ENTROPY", []).append(BK.get_value(_attent).reshape([1,1,1]))  # entropy
        # --
        attn_weights = nn.functional.dropout(attn_weights, p=attn_drop_rate, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)  # [*, H, Lq, D]
        # --
        if GET_ENV_VAR("ZCOLVAL") and not _no_change:  # store information for analysis
            # breakpoint()
            self.finish_one_layer(module, query_states, key_states, value_states, attention_mask, attn_logits, attn_weights, attn_output)
        # --
        return attn_output, attn_weights
    # --

    # --
    # replace KV cache
    @staticmethod
    def replace_kv_caches(cache, cache_ref, repl_types):
        if "K" in repl_types:
            cache.input_cache = cache_ref.input_cache.copy()  # also replace input cache!
            cache.key_cache = cache_ref.key_cache.copy()
        if "V" in repl_types:
            cache.value_cache = cache_ref.value_cache.copy()
    # --
