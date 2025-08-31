#

# utils for dataset

__all__ = [
    "MyDatasetConf", "MyDataset",
    "get_inst_info", "set_inst_info",
]

import os
import numpy as np
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm

from mspx.nn import BK
from mspx.utils import Conf, zlog, default_json_serializer, default_pickle_serializer, Random, Configurable, auto_mkdir, ZHelper, zopen, zglob, MathHelper

def _simple_collate_fn(x): return x

_KEY_INFO = 'info'

def get_inst_info(inst, key, df=None):
    if _KEY_INFO not in inst:
        inst[_KEY_INFO] = {}
    return inst[_KEY_INFO].get(key, df)

def set_inst_info(inst, key, val):
    if _KEY_INFO not in inst:
        inst[_KEY_INFO] = {}
    inst[_KEY_INFO][key] = val

class MyDatasetConf(Conf):
    def __init__(self, name=''):
        # basic ones
        self.name = name  # data name
        self.sample_weight = 1.  # data sampling weight
        self.input_paths = []  # input paths
        self.input_type = "AUTO"  # input data type (AUTO means judging by suffix)
        self.sharding = 'A'  # whether sharding: A(all), F(file), others=nope
        # for batching
        self.f_bsize = '1'  # T=token, 1=inst
        self.batch_size = 8  # batch size per device
        self.seg_size = 1024  # sequence segment size
        # misc
        self.mbatch_num = 100  # how many insts to batch as a group (for processing and shuffling)
        self.shuffle_time = 0  # whether shuffling
        self.shuffle_seed = 1377  # shuffling seed
        self.print_data_progress = False
        self.max_read_num = -1  # max number of items to read (or tokens for bin mode) [total of all ranks]
        self.binary_random_batch = False  # randomly put batches
        self.binary_dtype = ""  # specifying dtype for binary file
        # for dataset (from external)
        self.d_rank = 0
        self.d_world_size = 1
        self.d_toker = None

    @property
    def has_input_paths(self):
        return bool(self.input_paths)

@MyDatasetConf.conf_rd()
class MyDataset(Configurable, IterableDataset):
    def __init__(self, conf: MyDatasetConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyDatasetConf = self.conf
        self.toker = conf.d_toker
        # --
        self.f_bsize = {"T": (lambda x: MathHelper.up_div(len(x['input_ids']), conf.seg_size)), "1": (lambda x: 1)}.get(conf.f_bsize)
        if self.f_bsize is None:
            self.f_bsize = ZHelper.eval_ff(conf.f_bsize, default_args="x")
        # --

    def __repr__(self):
        return f"MyDataset(name={self.name},sample_weight={self.sample_weight})"

    @property
    def name(self):
        return self.conf.name

    @property
    def sample_weight(self):
        return self.conf.sample_weight

    # --
    # interfaces

    def __iter__(self):
        return self.yield_data()

    def get_dataloader(self, **kwargs):
        from torch.utils.data import DataLoader
        # note: already batched in data yielding
        ret = DataLoader(self, batch_size=None, collate_fn=_simple_collate_fn, **kwargs)
        ret._batch_info = self.get_batch_info()  # store info for future usage
        return ret

    # generator for the data stream
    def yield_data(self):
        # prepare source
        _input_paths, _input_type, _shard_inside_file = self._prepare_input_files()
        # yielding with different types
        _ff = getattr(self, f"yield_data_{_input_type}")
        yield from _ff(_input_paths, _shard_inside_file)

    # --
    # yielding

    def yield_data_plain(self, input_paths, sharding: bool):
        conf: MyDatasetConf = self.conf
        _rank, _world_size = conf.d_rank, conf.d_world_size
        # --
        inst_yielder = self._yield_processed_data(input_paths, sharding)
        _this_max_read_num = MathHelper.up_div(max(0, conf.max_read_num), _world_size)  # max read insts per device
        if _this_max_read_num > 0:  # yield with truncating
            inst_yielder = ZHelper.yield_with_truncating(inst_yielder, _this_max_read_num)
        zlog(f"Read [PLAIN] data from {input_paths} [sharding={sharding}] [this_maxR={_this_max_read_num}]")
        # --
        # batching
        _gen = Random.get_np_generator(conf.shuffle_seed+_rank)  # each rank has different seed!
        _bsize, _mbatch = conf.batch_size, conf.mbatch_num
        cur_bs, cur_insts = 0, []
        count_inst, count_b = 0, 0  # counts
        for idx, inst in enumerate(inst_yielder):
            # --
            # check full
            if cur_bs >= _mbatch * _bsize:
                for _ in range(conf.shuffle_time):
                    _gen.shuffle(cur_insts)
                all_batches = list(ZHelper.yield_batches(cur_insts, _bsize, (lambda x: x[1])))
                last_b = sum(z[1] for z in all_batches[-1])
                if last_b >= _bsize:  # the last one is also enough!
                    cur_bs, cur_insts = 0, []
                else:  # keep the small ones to the next round
                    cur_bs, cur_insts = last_b, all_batches.pop()
                for one_b in all_batches:
                    count_b += 1
                    yield [z[0] for z in one_b]  # yield the inst itself
            # --
            # get insts
            one_insts = [inst]
            for one_idx, one_inst in enumerate(one_insts):
                count_inst += 1
                # one_inst['_idx'] = (idx, one_idx)  # instance idx!
                set_inst_info(one_inst, '_idx', (idx, one_idx))  # instance idx!
                one_bs = self._get_inst_size(one_inst)
                cur_bs += one_bs
                cur_insts.append((one_inst, one_bs))
        # --
        for one_b in ZHelper.yield_batches(cur_insts, _bsize, (lambda x: x[1])):
            count_b += 1
            yield [z[0] for z in one_b]  # yield the inst itself
        # --
        zlog(f"Finished plain data yielding; counts={count_inst}/{count_b}.")

    def yield_data_bin(self, input_paths, sharding: bool):
        conf: MyDatasetConf = self.conf
        _rank, _world_size = conf.d_rank, conf.d_world_size
        # --
        _bsize, _mbatch, _ssize = conf.batch_size, conf.mbatch_num, conf.seg_size
        _toker = self.toker
        _pad_id = _toker.pad_token_id
        cast_dtype = np.int64
        count_b, count_t = 0, 0
        _gen = Random.get_np_generator(conf.shuffle_seed+_rank)  # each rank has different seed!
        _this_max_read_num = MathHelper.up_div(conf.max_read_num, _world_size * len(input_paths))
        zlog(f"Read [BINARY] data from {input_paths} [sharding={sharding}] [this_maxR={_this_max_read_num/(1024**2)}M x {len(input_paths)}]")
        for _path in input_paths:
            dtype = getattr(np, conf.binary_dtype) if conf.binary_dtype else ZHelper.get_np_type_from_name(_path, _toker.vocab_size)
            arr = np.memmap(_path, dtype=dtype, mode='r')
            # --
            len_arr = len(arr) if _this_max_read_num <= 0 else min(len(arr), _this_max_read_num)
            num_seg_all = MathHelper.up_div(len_arr, _ssize)  # total number of segments: (total = num_seg_all * _ssize)
            # split for ranks
            if sharding:
                _nseg_rank = MathHelper.up_div(num_seg_all, _world_size)  # number of segments per rank
                sidx_start = _rank * _nseg_rank
                sidx_end = min(num_seg_all, sidx_start + _nseg_rank)  # current rank seg-idx
            else:
                sidx_start, sidx_end = 0, num_seg_all
            # yielding
            _curr_num_seg_rank = sidx_end - sidx_start
            zlog(f"Read binary data from {_path} with ({arr.shape},{arr.dtype}) ->"
                 f" rank={_rank}/{_world_size}, seg={_ssize}: [{sidx_start}({sidx_start*_ssize}), {sidx_end}({sidx_end*_ssize})) = {_curr_num_seg_rank}({_curr_num_seg_rank*_ssize})"
                 f" total_batches = {_curr_num_seg_rank} / {_bsize} = {_curr_num_seg_rank/_bsize:.2f}")  # print information
            progress_bar = tqdm(total=_curr_num_seg_rank, desc=f"Iter file {_path}", disable=not (BK.is_local_main_process() and conf.print_data_progress))
            # --
            def _yield_sidxes():
                if conf.binary_random_batch:
                    for sidx_curr_start in range(sidx_start, sidx_end, _bsize * _mbatch):  # mbatch for shuffling
                        curr_sidxes = list(range(sidx_curr_start, min(sidx_curr_start + _bsize * _mbatch, sidx_end)))
                        for _ in range(conf.shuffle_time):
                            _gen.shuffle(curr_sidxes)
                        yield from ZHelper.yield_batches(curr_sidxes, _bsize)
                else:
                    _num_seg_batch = _curr_num_seg_rank // _bsize  # number of segs per batch
                    for _seg_offset in range(_num_seg_batch):  # note: this might ignore some of the batches, but it will be <bsize, so nvm ...
                        _s_idxes = [(sidx_start+_seg_offset+ii*_num_seg_batch) for ii in range(_bsize)]
                        yield _s_idxes
            # --
            for _one_s_idxes in _yield_sidxes():
                ret = np.full([_bsize, _ssize], _pad_id, dtype=dtype)  # [bs, L]
                ret_mask = np.full_like(ret, 0)
                for ii, ss in enumerate(_one_s_idxes):
                    aa = ss * _ssize  # switch to real idx
                    bb = min(len_arr, aa + _ssize)
                    _assign_arr = arr[aa:bb]
                    ret[ii, :len(_assign_arr)] = _assign_arr
                    ret_mask[ii, :len(_assign_arr)] = 1
                # --
                # note: all pad_id might raise NAN problem!
                # ret = ret[~((ret==_pad_id).all(-1))]  # note: this will lead to different shaped inputs for torch.compile!
                # note: -> let prep_mask handle, btw, another guard with extra_valid_mask
                yield_ids, yield_masks = (BK.as_tensor(ret.astype(cast_dtype)), BK.as_tensor(ret_mask.astype(cast_dtype)))
                # yield yield_ids, yield_masks
                yield {"input_ids": yield_ids, "attention_mask": yield_masks}
                progress_bar.update(len(_one_s_idxes))
                count_b += 1
                count_t += yield_ids.numel()
            # --
        # --
        zlog(f"Finished binary data yielding; counts={count_t}/{count_b}.")

    # --
    # utils

    def get_batch_info(self):
        conf: MyDatasetConf = self.conf
        return {"batch_size": conf.batch_size, "seg_size": conf.seg_size}

    # prepare input files and sharding info
    def _prepare_input_files(self):
        conf = self.conf
        # --
        _rank, _world_size = conf.d_rank, conf.d_world_size
        _shard_inside_file = False  # whether doing sharding inside a file?
        _input_paths = zglob(conf.input_paths, err_act='err')
        _input_type = conf.input_type
        if _input_type.lower() in ["", "auto"]:  # get suffix
            _input_type = ZHelper.get_check_all([z.rsplit('.', 1)[-1] for z in _input_paths], '')
        _input_type = {"json": "plain", "jsonl": "plain", "pkl": "plain", "bin": "bin"}.get(_input_type, _input_type)
        # sharding
        if conf.sharding == 'F':
            _input_paths = [z for i, z in enumerate(_input_paths) if i%_world_size == _rank]
        elif conf.sharding == 'A':
            _shard_inside_file = True  # sharding at instance level?
        zlog(f"Prepare data [rank={_rank}/{_world_size},type={_input_type}] from {_input_paths} [sharding={_shard_inside_file}]")
        return _input_paths, _input_type, _shard_inside_file

    # process data for plain mode
    def _yield_processed_data(self, input_paths, sharding: bool):
        conf = self.conf
        _rank, _world_size = conf.d_rank, conf.d_world_size
        # --
        def _inst_yielder():
            _idx = 0
            for path in input_paths:
                if path.endswith("json") or path.endswith("jsonl"):
                    _serializer = default_json_serializer
                else:
                    assert path.endswith("pkl")
                    _serializer = default_pickle_serializer
                for inst in _serializer.yield_iter(path):
                    if not sharding or _idx % _world_size == _rank:
                        yield inst
                    _idx += 1
        # --
        def _instp_yielder():
            for _insts in ZHelper.yield_batches(_inst_yielder(), conf.mbatch_num):
                _batch2 = self._process_insts(_insts)
                yield from _batch2
        # --
        yield from _instp_yielder()

    def _process_insts(self, insts):
        _add_special_tokens = False
        toker = self.toker
        if toker and len(insts) > 0 and 'input_ids' not in insts[0]:
            if "messages" in insts[0]:  # special mode
                for one_inst in insts:
                    ones = one_inst["messages"]
                    curr_text, curr_ids = "", []
                    _prev_idx = 0
                    ignore_ranges = []
                    for one_message in ones:
                        curr_text = curr_text + one_message["content"]
                        curr_ids = toker(curr_text, add_special_tokens=_add_special_tokens)["input_ids"]
                        if one_message["role"] != "assistant":
                            ignore_ranges.append((_prev_idx, len(curr_ids)))
                        _prev_idx = len(curr_ids)
                    for a, b in ignore_ranges:
                        for ii in range(a, b):
                            curr_ids[ii] = - curr_ids[ii]
                    one_inst["input_ids"] = curr_ids
            else:
                if 'text' in insts[0]:
                    _inputs = None
                    all_texts = [z['text'] for z in insts]
                else:
                    _inputs = toker([z['input'] for z in insts], add_special_tokens=_add_special_tokens)
                    all_texts = [z['input'] + z['output'] for z in insts]
                res = toker(all_texts, add_special_tokens=_add_special_tokens)
                for idx, inst in enumerate(insts):
                    set_inst_info(inst, 'num_words', len(all_texts[idx].split()))
                    # del inst['text']  # note: simply no DEL!
                    inst['input_ids'] = res['input_ids'][idx]
                    if _inputs is not None:
                        for ii in range(len(_inputs[idx])):  # mark as negative for inputs!
                            inst['input_ids'][ii] = - inst['input_ids'][ii]
        else:
            if len(insts) > 0:
                assert 'input_ids' in insts[0]
        return insts

    def _get_inst_size(self, inst):
        return self.f_bsize(inst)
