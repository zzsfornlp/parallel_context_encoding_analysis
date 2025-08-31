#

# helpers for generation

__all__ = [
    "GenConf", "do_generate"
]

import torch

from mspx.nn import BK
from mspx.utils import Conf, zlog, ZHelper, Timer, Constants

class GenConf(Conf):
    def __init__(self):
        self.gen_core = _GenerateConf()  # general ones
        self.stop_seqs = []  # List of str

class _GenerateConf(Conf):
    def __init__(self):
        self.do_sample = True
        self.num_beams = 1
        self.temperature = 1.0  # temperature
        self.top_k = 50
        self.top_p = 1.0
        self.max_new_tokens = 100  # new tokens to generate
        self.repetition_penalty = 1.0  # ~1.2 yields a good balance

# note: from lm-eval-harness
def stop_sequences_criteria(stop_sequences, toker):
    from transformers import StoppingCriteriaList, StoppingCriteria
    # --
    # Multi-token stopping criteria
    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""
        def __init__(self, sequence: str, toker):
            from .helper import TokenizerHelper
            self.sequence = sequence
            self.toker = toker
            self.seq_ids = TokenizerHelper.do_subtok(sequence, toker, ensure_continuing=True)  # check endings
            self.CHECK_MORE = 2  # check some more tokens!
            # states
            self.t_done = None
            self.t_ids = None

        def _init_seq(self, input_ids):
            import torch
            self.t_ids = torch.as_tensor(self.seq_ids).to(input_ids)  # [L]
            self.t_done = torch.zeros(len(input_ids)).to(input_ids)  # [bs]

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            if self.t_ids is None:
                self._init_seq(input_ids)
            _slen = len(self.t_ids) + self.CHECK_MORE
            # note: still need to check the original string, since there may be slight tokenization differences
            lookback_ids_batch = input_ids[:, -_slen:]  # [bs, L]
            lookback_strs_batch = self.toker.batch_decode(lookback_ids_batch)  # [bs]
            _matches = [z.endswith(self.sequence) for z in lookback_strs_batch]  # [bs]
            self.t_done = self.t_done + torch.as_tensor(_matches).to(self.t_done)
            ret = (self.t_done > 0).all().item()
            # breakpoint()
            return ret
    # --
    return StoppingCriteriaList([MultiTokenEOSCriteria(sequence, toker) for sequence in stop_sequences])

# --
def do_generate(batch_inputs, model, past_key_values, conf: GenConf, return_dict=False):
    # assert past_key_values is None, "Currently not supported!"
    toker = model.engine.toker
    stopping_criteria = stop_sequences_criteria(conf.stop_seqs, toker)
    generation_kwargs = conf.gen_core.to_dict(store_type=False)
    _t_inputs, _t_masks = model.engine.do_tokenize_prefixes(batch_inputs, add_special_tokens=model.engine.should_add_special_tokens(past_key_values is not None), prepare_tensor=True)
    # --
    # note: here inputs do not include previous tokens in past_key_values! But we need them to use transformer's generate!
    if past_key_values is not None and past_key_values.get_seq_length() > 0:
        _p_inputs, _p_masks = past_key_values.get_inputs(["input", "mask"])  # [*, Lp]
        _new_bs, _old_bs = _t_inputs.shape[0], _p_inputs.shape[0]
        if _new_bs > _old_bs:
            _p_inputs, _p_masks = _p_inputs.repeat_interleave(_new_bs//_old_bs, dim=0), _p_masks.repeat_interleave(_new_bs//_old_bs, dim=0)
        _t_inputs, _t_masks = torch.cat([_p_inputs, _t_inputs], dim=1), torch.cat([_p_masks, _t_masks], dim=1)  # [*, Lp+Lc]
    # --
    model_res = model.generate(input_ids=_t_inputs, attention_mask=_t_masks, stopping_criteria=stopping_criteria, pad_token_id=model.toker.pad_token_id, past_key_values=past_key_values, use_cache=True, **generation_kwargs)
    t_cont = model_res[:, _t_inputs.shape[-1]:]  # [bs, Cont]
    ret_conts = []
    for arr in BK.get_value(t_cont):
        s0 = model.toker.decode(arr)
        s = s0
        for term in conf.stop_seqs:
            if len(term) > 0:
                s = s.split(term)[0]
        ret_conts.append(s)
    # for one in ret_conts:
    #     print(one)
    # breakpoint()
    if return_dict:
        res = {"conts": ret_conts, "inputs": _t_inputs, "outputs": model_res, "input_length": _t_inputs.shape[-1], "output_length": model_res.shape[-1]}
        res["new_length"] = res["output_length"] - res["input_length"]
        return res
    else:
        return ret_conts
