#

# eval LLM with helmet

from typing import List
import math
import time
from collections import Counter, defaultdict

import torch
import numpy as np
import tqdm

from mspx.utils import Conf, init_everything, ZDEBUG, Timer, zlog, ZHelper, default_json_serializer, default_pickle_serializer, zwarn, Random, StatRecorder, GlobalObjects
from mspx.nn import BK
from mspx.core import MyEngineConf, MyEngine, MyDatasetConf, MyDataset, DataPadder, get_inst_info, RunnerConf, MyRunner, info_model, info_trainable_parameters, ListParInput

from .data import DATA_ITEMS, load_data
from .selatt import SelAttConf, SelAttHelper

class MainConf(Conf):
    def __init__(self):
        self.engine = MyEngineConf()  # model
        self.input_sig = ""  # look at data
        self.output_path = ""
        self.output_pkl = ""
        # self.inst_batch_size = 1  # how many insts inside one batch
        self.inst_eval_count = 100  # how many insts to eval
        # for LM eval
        # self.test_batch_size = 4  # running batch size
        # self.test_seg_size = 8192  # seg length
        # self.test_step_size = 0  # LM-eval sliding window size
        self.test_context_split = 1  # (split_num)
        self.test_context_fix_ratio = -1.0  # if >0., then adding context into the query
        # for splitting
        self.test_starting_prompt = ""  # prompt at the beginning of every seq
        self.test_starting_repeat = 1  # repeat the starting prompt?
        # self.test_no_scontext = False  # simply discard those things!
        # self.test_split_by_sep = False  # split based on sep tokens
        # self.test_split_by_sep_random = False  # randomly shuffling for controlled exp
        # self.test_split_step = -1  # >0 means abs step, <0 means chunk_size//abs(step)
        # parlist mode
        self.position_mode = "ME"  # order, {H,A,M} x {L,R,E}
        self.no_smask = False  # no application of smask
        # self.shuffle_splits = False  # shuffle the split pieces
        self.max_gen_length = 20
        self.sep_pass = True  # separate pass for contexts and later ones
        # --
        self.selatt = SelAttConf()  # for selective attention
        self.switch_kv = ""  # switch full-context for k or v??

STARTING_PROMPTS = {
    "p1": "Given the following contexts, answer the final question accordingly:\n\n",
    "p2": "\n\n",
}

def prepare_inst(inst, conf: MainConf, toker, starting_prompt_tokens):
    one_stat = Counter()
    _eff_split_count = conf.test_context_split  # ignore the later two!
    _eff_fix_ratio = conf.test_context_fix_ratio
    _tokenized_contexts = [toker.convert_tokens_to_ids(toker.tokenize(one, add_special_tokens=False)) for one in inst['context']]
    _tokenized_query = toker.convert_tokens_to_ids(toker.tokenize(inst['query'], add_special_tokens=False))
    if _eff_fix_ratio > 0.:
        _num_fix = int(len(_tokenized_contexts) * _eff_fix_ratio)
        _tokenized_query = sum(_tokenized_contexts[-_num_fix:], []) + _tokenized_query
        _tokenized_contexts = _tokenized_contexts[:-_num_fix]  # move from context into query
    # --
    each_bucket_count = [len(_tokenized_contexts) // _eff_split_count] * _eff_split_count
    for ii in range(len(_tokenized_contexts) % _eff_split_count):
        each_bucket_count[ii] += 1
    all_pieces = []
    idx_old2new = []  # map from old idx to new idx
    curr_ctx_idx = 0
    for one_count in each_bucket_count:
        idx_old2new.extend([len(all_pieces)] * one_count)
        all_pieces.append(sum(_tokenized_contexts[curr_ctx_idx:curr_ctx_idx+one_count], []))
        curr_ctx_idx += one_count
    assert len(_tokenized_contexts) == sum(each_bucket_count), "Number of pieces mismatched!"
    all_pieces_tuple = tuple(all_pieces)
    if conf.sep_pass:
        eval_input = ListParInput.create([starting_prompt_tokens, all_pieces_tuple])
        ret = {"context": eval_input, "query": _tokenized_query}
    else:
        eval_input = ListParInput.create([starting_prompt_tokens, all_pieces_tuple, _tokenized_query])
        ret = {"context": eval_input, "query": None}
    ret["oracle_idxes"] = [idx_old2new[z] for z in inst.get("oracle_idxes", []) if z<len(idx_old2new)]
    # --
    # stat
    _ff_op = lambda x: (1+x)*x/2
    one_ratio = sum([_ff_op(len(z)) for z in all_pieces_tuple]) / _ff_op(sum(len(z) for z in all_pieces_tuple))
    one_stat["inst"] += 1
    one_stat["inst_token_context"] += sum(len(z) for z in _tokenized_contexts)
    one_stat["inst_token_query"] += len(_tokenized_query)
    one_stat["inst_ratio"] += one_ratio
    # --
    return one_stat, ret

def forward_one_piece(model, selatt, t_input, past_key_values, conf, old_t_aggr):
    with selatt.env_use_sel_attn():
        _aggr_dims = selatt.conf.aggr_dims
        if "2pass" in _aggr_dims:
            # pass 1
            with past_key_values.with_frozen():  # first pass!
                model(t_input, past_key_values=past_key_values)
            t_probs = torch.stack(selatt.group_attn_pass1, 1)  # [*, Layer, Head, Seq, Lm]
            t_aggr = BK.aggr_tensor(t_probs, [i for i, n in zip([-4, -3, -2], ["layer", "head", "seq"]) if n in _aggr_dims], selatt.conf.aggr_topk)
            _aggr_momentum = selatt.conf.aggr_momentum
            old_t_aggr = (old_t_aggr.mean(-2, keepdim=True) * _aggr_momentum + t_aggr * (1.-_aggr_momentum)) if old_t_aggr is not None else t_aggr
            selatt.group_attn_pass2 = old_t_aggr
            # --
            if ZDEBUG():
                _shapeT = t_probs.shape
                for sidx in range(_shapeT[-2]):
                    _t_tmp_vals, _t_tmp_idxes = t_probs[..., sidx, :].flatten().topk(5)
                    _t_layer, _t_head, _t_which = _t_tmp_idxes // (_shapeT[-1]*_shapeT[-3]), (_t_tmp_idxes % (_shapeT[-1]*_shapeT[-3])) // _shapeT[-1], _t_tmp_idxes % _shapeT[-1]
                    zlog(f"Seq-{sidx}: {_t_tmp_vals.tolist()} {(_t_layer.tolist(), _t_head.tolist(), _t_which.tolist())}")
                breakpoint()  # p t_aggr.topk(5), selatt.current_context_pack
            # --
        res = model(t_input, past_key_values=past_key_values)
        selatt.clear_group_attn()
    return res, old_t_aggr

def greedy_decode(model, selatt, inputs, conf, inst, max_gen_length):
    old_t_aggr = None
    init_res = model([inputs["context"]], no_smask=conf.no_smask)
    if conf.switch_kv:
        _full_res = model([ListParInput.create([inputs["context"].tok_ids])], no_smask=conf.no_smask)
        SelAttHelper.replace_kv_caches(init_res.past_key_values, _full_res.past_key_values, conf.switch_kv)  # replace KV cache!
    _input_length = init_res.logits.shape[-2]
    if inputs["query"]:
        init_res, old_t_aggr = forward_one_piece(model, selatt, torch.as_tensor(inputs["query"]).unsqueeze(0), init_res.past_key_values, conf, old_t_aggr)
    past_key_values = init_res.past_key_values
    next_token_scores = init_res.logits[..., -1, :]  # [bs, V]
    all_output_tokens = []
    if max_gen_length is None:
        max_gen_length = conf.max_gen_length
    for step_idx in range(max_gen_length):
        next_tokens = torch.argmax(next_token_scores, dim=-1, keepdim=True)  # [bs, 1], decide next tokens
        all_output_tokens.append(next_tokens)
        next_res, old_t_aggr = forward_one_piece(model, selatt, next_tokens, past_key_values, conf, old_t_aggr)
        # prepare next
        past_key_values = next_res.past_key_values
        next_token_scores = next_res.logits[..., -1, :]  # [bs, V]
    t_output = torch.cat(all_output_tokens, -1)  # [bs, L]
    selatt.finish_one_inst()
    # --
    assert t_output.shape[0] == 1, "For simplicity use bs=1"
    answer = model.toker.decode(t_output[0], skip_special_tokens=True)
    full_answer = inst["query"].split("\n")[-1] + answer  # append sig simply for the last line
    ret = {"output": full_answer, "input_len": _input_length, "output_len": t_output.shape[-1]}
    return ret

def run_test(model, selatt, data_info, conf: MainConf, max_gen_length: int):
    test_recorder = StatRecorder()
    toker = model.toker
    _pad_id = toker.pad_token_id
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
    # --
    all_stat = Counter()
    metrics = defaultdict(list)
    results = []
    start_time = time.time()
    _all_data = data_info["data"]
    if conf.inst_eval_count > 0:
        _gen = Random.get_np_generator(seed=42)
        _gen.shuffle(_all_data)  # note: do shuffling first!!
        _all_data = _all_data[:conf.inst_eval_count]
    for idx, inst in enumerate(tqdm.tqdm(_all_data)):
        # prepare inputs!
        _stat, _input = prepare_inst(inst, conf, toker, _prompt_tokens)
        all_stat += _stat
        # run model
        # output = model.generate(**generate_kwargs)
        with selatt.env_use_context([_input["context"]], oracle_idxes=[_input["oracle_idxes"]]):
            output = greedy_decode(model, selatt, _input, conf, inst, max_gen_length=max_gen_length)
        if output is None:
            zwarn(f"skipping example {idx + 1} because the model returned None")
            continue
        # metric
        mets, others = data_info['post_process'](output, inst)
        output.update({**others, **mets})
        for k, v in mets.items():
            metrics[k].append(v)
        metrics["input_len"].append(output["input_len"])
        metrics["output_len"].append(output["output_len"])
        result = {**inst, **output}
        result.pop("context", None)
        result.pop("input_ids", None)
        results.append(result)
    # --
    end_time = time.time()
    mem_usage = sum([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
    zlog(f"Memory usage: {mem_usage/1000**3:.02f} GB")
    zlog(f"Throughput: {len(results) / (end_time - start_time):.02f} samples/s")
    # --
    _extra_res = selatt.finish_all_insts()
    _avg_entropy = _extra_res.get("ENTROPY", np.asarray([0.])).mean().item()
    if conf.output_pkl:
        default_pickle_serializer.to_file(_extra_res, conf.output_pkl)
    # --
    # average metric
    averaged_metrics = {"ENTROPY": _avg_entropy}
    averaged_metrics.update({k: np.mean(v).item()*(100 if "_len" not in k else 1) for k, v in metrics.items()})
    for k in list(all_stat.keys()):
        if k.startswith("inst_"):
            all_stat[k+"_AVG"] = all_stat[k] / all_stat["inst"]
    zlog("Overall stat: " + ZHelper.printd_str(all_stat, sep=' '))
    zlog("Averaged metrics: " + ZHelper.printd_str(averaged_metrics, sep=' '))
    zlog(f"Final results are {averaged_metrics}")  # print final results
    output = {
        "data": results,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "memory_usage": mem_usage,
        "throughput": len(results) / (end_time - start_time),
    }
    if conf.output_path:
        default_json_serializer.to_file(output, conf.output_path)
    # --
    return averaged_metrics

# --
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
        _data_items = DATA_ITEMS[conf.input_sig]
        _all_res = {}
        for vv in _data_items:
            zlog(f"Start to run for {vv}")
            data_info = load_data(vv["dataset"], vv["path"])
            _dataset = vv['dataset']
            res = run_test(model, selatt, data_info, conf, max_gen_length=vv["max_gen_length"])
            _all_res[_dataset] = res
            zlog(f"Test-Info for {vv}: {ZHelper.printd_str(res, sep=' ')}")
        zlog(f"Test-Info: {ZHelper.printd_str(res, sep=' ')}")
    # --

# python -mpdb -m mspx.tasks.memana.run_helmet
if __name__ == '__main__':
    main()
