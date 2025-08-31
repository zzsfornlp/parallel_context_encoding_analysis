#

# eval LM's PPL

from typing import List
import math
from collections import Counter

import numpy as np
import torch
import tqdm

from mspx.utils import Conf, init_everything, zopen_withwrapper, Timer, zlog, ZHelper, default_json_serializer, default_pickle_serializer, zwarn, Random, StatRecorder, GlobalObjects
from mspx.nn import BK
from mspx.core import MyEngineConf, MyEngine, MyDatasetConf, MyDataset, DataPadder, get_inst_info, RunnerConf, MyRunner, info_model, info_trainable_parameters, ListParInput

from .selatt import SelAttConf, SelAttHelper

class MainConf(Conf):
    def __init__(self):
        self.engine = MyEngineConf()  # model
        self.input_file = ""  # input eval file
        self.output_pkl = ""
        # self.inst_batch_size = 1  # how many insts inside one batch
        self.inst_eval_count = 100  # how many insts to eval
        # for LM eval
        self.test_batch_size = 4  # running batch size
        self.test_seg_size = 8192  # seg length
        self.test_step_size = 0  # LM-eval sliding window size
        self.test_context_split = [1, 1024, 1024]  # (split_num) + fix-size + actual-size
        self.test_context_fix_ratio = -1.0  # if >0., then replace fix size!
        # for splitting
        self.test_starting_prompt = ""  # prompt at the beginning of every seq
        self.test_starting_repeat = 1  # repeat the starting prompt?
        self.test_no_scontext = False  # simply discard those things!
        self.test_split_by_sep = False  # split based on sep tokens
        self.test_split_by_sep_random = False  # randomly shuffling for controlled exp
        self.test_split_step = -1  # >0 means abs step, <0 means chunk_size//abs(step)
        # parlist mode
        self.position_mode = "ME"  # order, {H,A,M} x {L,R,E}
        self.no_smask = False  # no application of smask
        self.shuffle_splits = False  # shuffle the split pieces
        self.sep_pass = True  # separate pass for contexts and later ones
        # --
        self.selatt = SelAttConf()  # for selective attention
        self.switch_kv = ""  # switch full-context for k or v??

def get_steps0(length: int, seg_size: int, step_size: int):
    ret = []
    _start, _end = 0, 0
    while _end < length:
        _last_end = _end  # last end
        _end = min(length, _start + seg_size)  # new end
        assert _start <= _last_end and _last_end < _end
        # ret.append((_start, _last_end, _end))  # [start, [new, end)
        ret.append((_start, _end))  # [start, [new, end)
        _start += step_size
    return ret

def get_steps1(length: int, seg_size: int, step_size: int):
    ret = []
    _start, _end = 0, min(length, seg_size)
    while True:
        ret.append((_start, _end))
        if _end >= length:
            break
        _start += step_size
        _end += step_size
        _end = min(_end, length)
    return ret

STARTING_PROMPTS = {
    "p1": "Given the following partial context, predicting the next sequence of words:\n\n",
    "p2": "\n\n",
}

# simple top-down greedy splitting
def split_contexts(ts, target: int, toker, do_random: bool):
    # --
    def _lookfor(_start, _end, _step, _list):
        while _start != _end and not _list[_start]:
            _start += _step
        return _start if _start != _end else None
    # --
    def _choose(_left, _right, _middle):
        if _left is None:
            return _right
        elif _right is None:
            return _left
        else:
            if abs(_left - _middle) < abs(_right - _middle):
                return _left
            else:
                return _right
    # --
    is_major_split = [(toker.decode([t]).endswith("\n\n")) for t in ts]
    is_minor_split = [(toker.decode([t]).endswith("\n")) for t in ts]
    pieces = [(0, len(ts))]
    # --
    _BUFFER = 3
    while len(pieces) < target:  # until reach our final goal
        largest_piece = max(pieces, key=(lambda x: (int(any(is_major_split[x[0]+_BUFFER:x[1]-_BUFFER])), x[1]-x[0])))  # (has_major, span_size)
        left, right = largest_piece  # [L, R)
        middle = (left + right) // 2
        # --
        valid_left, valid_right = left + _BUFFER, right - _BUFFER  # to avoid too small pieces
        assert middle > valid_left and middle < valid_right, "Piece is too small!"
        for split_flags in [is_major_split, is_minor_split]:  # check minor is no major
            cand_left, cand_right = _lookfor(middle, valid_left-1, -1, split_flags), _lookfor(middle, valid_right, 1, split_flags)
            chosen_idx = _choose(cand_left, cand_right, middle)
            if chosen_idx is not None:
                break
        if chosen_idx is None:
            chosen_idx = middle + 1  # no one can be found, simply choose the middle one
        else:
            while chosen_idx < valid_right and is_major_split[chosen_idx]:
                chosen_idx += 1
        # --
        pieces.remove(largest_piece)
        pieces.append((left, chosen_idx))
        pieces.append((chosen_idx, right))
    # --
    pieces.sort()
    if do_random:
        _gen = Random.get_generator()
        new_sizes = [b-a for a,b in pieces]
        _gen.shuffle(new_sizes)
        _new_start = 0
        _new_pieces = []
        for _size in new_sizes:
            _new_pieces.append((_new_start, _new_start + _size))
            _new_start += _size
        assert _new_start == len(ts)
        ret = [ts[a:b] for a, b in _new_pieces]
    else:
        ret = [ts[a:b] for a,b in pieces]
    # breakpoint()
    return ret

def run_test(model, selatt, data_iter, conf: MainConf):
    test_recorder = StatRecorder()
    toker = model.toker
    _pad_id = toker.pad_token_id
    all_inst_count = 0
    all_loss, all_count, all_count_words = 0., 0., 0.
    all_ratios = []
    _test_seg_size, _test_step_size = conf.test_seg_size, conf.test_step_size
    if _test_step_size <= 0:
        _test_step_size = conf.test_seg_size
    _eff_split_count, _eff_fix, _eff_actual = conf.test_context_split
    _eff_fix_ratio = conf.test_context_fix_ratio
    # for batch_insts in tqdm.tqdm(ZHelper.yield_batches(data_iter, conf.inst_batch_size)):
    _prompt_tokens = []
    _test_starting_prompt = STARTING_PROMPTS.get(conf.test_starting_prompt, conf.test_starting_prompt)
    if _test_starting_prompt:  # starting prompts
        for rii in range(conf.test_starting_repeat):  # only add bos for the first one!
            if isinstance(_test_starting_prompt, str):
                _prompt_tokens += toker(_test_starting_prompt, add_special_tokens=False)['input_ids']
            else:
                assert isinstance(_test_starting_prompt, list)
                _prompt_tokens += sum([toker(zz, add_special_tokens=False)['input_ids'] for ii,zz in enumerate(_test_starting_prompt)], [])
    zlog(f"Get {len(_prompt_tokens)} starting prompt tokens from <<< {_test_starting_prompt}x{conf.test_starting_repeat} ||| {_prompt_tokens} >>>")
    _gen = Random.get_generator()
    for batch_insts in tqdm.tqdm(data_iter):
        # create real insts
        one_stat = Counter()
        eval_seqs, eval_inputs = [], []
        for _inst in [batch_insts]:
            _tokens = toker.convert_tokens_to_ids(toker.tokenize(_inst["text"], add_special_tokens=False))
            _all_ranges = get_steps1(len(_tokens), _test_seg_size, _test_step_size)
            # assert get_steps0(len(_tokens), _test_seg_size, _test_step_size) == _all_ranges
            # zlog(f"{_inst['info']} {_all_ranges}")
            for a, b in _all_ranges:
                if b - a < _test_seg_size:  # note: simply discard the remaining ones
                    continue
                assert (b - a) > _eff_actual
                eval_seqs.append(_tokens[a:b])
            one_stat["inst"] += 1
            one_stat["token"] += len(_tokens)
            all_inst_count += 1
        one_stat["seq"] += len(eval_seqs)
        # make structured contexts
        pass2_inputs = []
        for one_ii, one_seq in enumerate(eval_seqs):
            _one_eff_fix = _eff_fix if _eff_fix_ratio <= 0. else int(_eff_fix_ratio*(len(one_seq)-_eff_actual))
            assert len(one_seq) > (_one_eff_fix + _eff_actual)
            _p0, _p1, _p2 = one_seq[:-(_one_eff_fix+_eff_actual)], one_seq[-(_one_eff_fix+_eff_actual):-_eff_actual], one_seq[-_eff_actual:]
            if conf.test_no_scontext:
                _p0s = []  # make it empty
            else:
                # --
                # split context
                if conf.test_split_by_sep:
                    _p0s = split_contexts(_p0, _eff_split_count, toker, conf.test_split_by_sep_random)
                else:
                    _p0s = list(ZHelper.yiled_chunks(_p0, count=_eff_split_count, step=conf.test_split_step))
                if conf.shuffle_splits:  # randomly shuffle the pieces
                    _gen.shuffle(_p0s)
                # --
                _p0s = tuple(_p0s)
            if conf.sep_pass:
                _one_input = ListParInput.create([_prompt_tokens, _p0s])
                pass2_inputs.append(_p1 + _p2)
            else:
                _one_input = ListParInput.create([_prompt_tokens, _p0s, _p1, _p2])
            eval_inputs.append(_one_input)
            # calculate ratios
            _ff_op = lambda x: (1+x)*x/2
            one_ratio = sum([_ff_op(len(z)) for z in _p0s]) / _ff_op(len(_p0))
            all_ratios.append(one_ratio)
            # breakpoint()
        # --
        # LM eval
        for ii in range(0, len(eval_inputs), conf.test_batch_size):
            # new forward
            _past_key_values = None
            _curr_inputs = eval_inputs[ii:ii+conf.test_batch_size]
            with selatt.env_use_context(_curr_inputs):
                if any(z.tok_ids for z in _curr_inputs):
                    res0 = model(_curr_inputs, no_smask=conf.no_smask)
                    _past_key_values = res0.past_key_values
                    if conf.switch_kv:
                        _full_res = model([ListParInput.create([z.tok_ids]) for z in _curr_inputs], no_smask=conf.no_smask)
                        SelAttHelper.replace_kv_caches(_past_key_values, _full_res.past_key_values, conf.switch_kv)  # replace KV cache!
                if pass2_inputs:
                    _pass2_inputs = pass2_inputs[ii:ii+conf.test_batch_size]
                    _t = torch.full([len(_pass2_inputs), max(len(z) for z in _pass2_inputs)], _pad_id)
                    for _one_pass2_input in _pass2_inputs:
                        _t[-len(_one_pass2_input):] = torch.as_tensor(_one_pass2_input)  # left padding!
                    with selatt.env_use_sel_attn():
                        one_res = model(_t, attention_mask=(_t!=_pad_id).long(), past_key_values=_past_key_values)
                else:
                    one_res = res0
            selatt.finish_one_inst()
            # calculate loss
            _t_logits = one_res.logits[..., -_eff_actual-1:-1, :]  # [*, T, V]
            _t_labels = one_res.past_key_values.get_inputs("input")[:, -_eff_actual:]  # [*, T]
            _nll = - _t_logits.log_softmax(-1, dtype=BK.float32).gather(-1, _t_labels.unsqueeze(-1)).squeeze(-1)  # [*, T]
            all_loss += _nll.sum().item()
            # all_count += ret_mask.sum().item()
            all_count += np.prod(list(_nll.shape)).item()  # note: assume no padding!
        # --
        test_recorder.record(one_stat)
        if all_inst_count >= conf.inst_eval_count:
            break
    # --
    _extra_res = selatt.finish_all_insts()
    _avg_entropy = _extra_res.get("ENTROPY", np.asarray([0.])).mean().item()
    if conf.output_pkl:
        default_pickle_serializer.to_file(_extra_res, conf.output_pkl)
    # --
    ret = test_recorder.summary("Final")
    log_ppl = all_loss / all_count if all_count > 0 else -1.
    ppl = math.exp(log_ppl)
    log2_ppl = math.log2(ppl)  # bit per unit
    _walpha = 1. if all_count_words == 0 else (all_count / all_count_words)
    zlog(f"#=====\nFinished LM testing with: {all_loss}/{all_count}={log_ppl:.4f} log2={log2_ppl:.4f} ENTROPY={_avg_entropy:.2f} PPL={ppl:.2f}\n" +
         # (f"\tRES-word: LPPL*{_walpha}={log_ppl*_walpha} PPL={math.exp(log_ppl*_walpha):.2f}\n" if conf.test_eval_wnum else "") +
         f"\tOther stat: {ZHelper.printd_str(ret, sep=' ')}\n" +
         f"\tOverall avg ratios = {np.asarray(all_ratios).mean() if all_ratios else -1.:.4f}")
    ret.update({'log_ppl': log_ppl, 'log2_ppl': log2_ppl, 'ppl': ppl, 'zres': -ppl})  # note: higher result is better
    # --
    return ret

def main():
    conf: MainConf = init_everything(MainConf())
    # --
    # model
    engine: MyEngine = conf.engine.make_node()  # get model & toker
    toker, model = engine.toker, engine.model
    zlog(f"Test with model: {info_model(model)} ***:\n{info_trainable_parameters(model).to_string()}")
    ListParInput.set_position_mode(conf.position_mode)
    selatt = SelAttHelper(conf.selatt)
    selatt.modify_model(model)
    # --
    # run!
    with Timer("eval", print_date=True), BK.inference_mode():
        _data_iter = default_json_serializer.yield_iter(conf.input_file)
        res = run_test(model, selatt, _data_iter, conf)
        zlog(f"Test-Info: {ZHelper.printd_str(res, sep=' ')}")
    # --

# python -mpdb -m mspx.tasks.memana.run_ppl
if __name__ == '__main__':
    main()
