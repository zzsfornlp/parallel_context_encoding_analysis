#

# some helper functions

__all__ = [
    "DataPadder", "TrainingProgressRecord", "SVConf", "ScheduledValue",
    "info_trainable_parameters", "info_model",
    "TokenizerHelper",
]

from typing import List, Sequence
import numpy as np
from collections import Counter
from mspx.utils import zlog, ZObject, Registrable, ZResult, Conf, ZHelper

# --
# data padding: from lists to np.arr
class DataPadder:
    # a helper function to turn a list of lengths to MaskArr
    @staticmethod
    def len2mask(lengths: List[int]):
        ret = np.zeros((len(lengths), max(lengths)))
        for ii, ll in enumerate(lengths):
            ret[ii][:ll] = 1.
        return ret

    # for 2d case
    @staticmethod
    def batch_2d(inputs: Sequence[Sequence], pad_val, max_len=None, dtype=None, ret_mask=False, ret_tensor=False, pad_left=False):
        bs = len(inputs)
        # if max_len is None:
        #     max_len = max((len(z) for z in inputs), default=1)
        _actual_max_len = max((len(z) for z in inputs), default=1)
        max_len = _actual_max_len if max_len is None else min(max_len, _actual_max_len)
        if dtype is None:  # guess dtype
            if isinstance(pad_val, int):
                dtype = np.int64
            elif isinstance(pad_val, float):
                dtype = np.float32
        arr = np.full([bs, max_len], pad_val, dtype=dtype)
        if ret_mask:
            arr_m = np.zeros([bs, max_len], dtype=np.int64)
        else:
            arr_m = None
        for ii, vv in enumerate(inputs):
            _ll = min(len(vv), max_len)
            if pad_left:
                arr[ii, -_ll:] = vv[:_ll]
                if ret_mask:
                    arr_m[ii, -_ll:] = 1.
            else:
                arr[ii, :_ll] = vv[:_ll]
                if ret_mask:
                    arr_m[ii, :_ll] = 1.
        if ret_tensor:
            import torch
            return torch.as_tensor(arr), torch.as_tensor(arr_m) if ret_mask else None
        else:
            return arr, arr_m
        # --

    # for 3d case
    @staticmethod
    def batch_3d(inputs: Sequence[Sequence[Sequence]], pad_val,
                 max_len1=None, max_len2=None, dtype=None, ret_mask=False, ret_tensor=False):
        bs = len(inputs)
        # if max_len1 is None:
        #     max_len1 = max((len(z) for z in inputs), default=1)
        # if max_len2 is None:
        #     max_len2 = max((len(b) for a in inputs for b in a), default=1)
        _actual_max_len1 = max((len(z) for z in inputs), default=1)
        _actual_max_len2 = max((len(b) for a in inputs for b in a), default=1)
        max_len1 = _actual_max_len1 if max_len1 is None else min(max_len1, _actual_max_len1)
        max_len2 = _actual_max_len2 if max_len2 is None else min(max_len2, _actual_max_len2)
        arr = np.full([bs, max_len1, max_len2], pad_val, dtype=dtype)
        if ret_mask:
            arr_m = np.zeros([bs, max_len1, max_len2], dtype=np.int64)
        else:
            arr_m = None
        for ii1, vv1 in enumerate(inputs):
            for ii2, vv2 in enumerate(vv1):
                _ll = min(len(vv2), max_len2)
                arr[ii1, ii2, :_ll] = vv2[:_ll]
                if ret_mask:
                    arr_m[ii1, ii2, :_ll] = 1.
        if ret_tensor:
            import torch
            return torch.as_tensor(arr), torch.as_tensor(arr_m) if ret_mask else None
        else:
            return arr, arr_m
        # --

# --
# training process

# Record of the training process
@Registrable.rd('tpr')
class TrainingProgressRecord(ZObject):
    def __init__(self):
        super().__init__()
        # idxes (counts: how many has past up to now!)
        self.eidx = 0  # epochs: increase end-of-epoch
        self.uidx = 0  # updates: increase after updating
        self.fidx = 0  # forwards: increase after forward
        self.iidx = 0  # instances: increase for number of instances
        self.cidx = 0  # checkpoints: increase end-of-validate(check)
        # idx counter (from which dataset?)
        self.iidx_counter = {}
        self.fidx_counter = {}
        self.uidx_counter = {}
        # checkpoints: len==self.cidx
        self.chp_names: List[str] = []
        self.train_results: List[ZResult] = []
        self.dev_results: List[ZResult] = []
        # all with dev results
        # track best point
        self.overall_best_result = ZResult()  # overall best, regardless of 'record_best' or not
        self.overall_best_point = -1
        self.best_result = ZResult()  # recorded best
        self.best_point = -1  # recorded best point

    def current_suffix(self, brief=False):
        # 4 digits should be enough
        if brief:  # only step/update info!
            sname = f".u{self.uidx:06d}"
        else:
            sname = f".c{self.cidx:03d}-e{self.eidx}-u{self.uidx}-f{self.fidx}"
        return sname

    def info_best(self):
        if self.best_point < 0:
            return [-1, "None", ZResult()]
        else:
            return [self.best_point, self.chp_names[self.best_point], self.best_result]

    def info_overall_best(self):
        if self.overall_best_point < 0:
            return [-1, "None", ZResult()]
        else:
            return [self.overall_best_point, self.chp_names[self.overall_best_point], self.overall_best_result]

    # simple updates for plain idxes
    def update_eidx(self, d: int):
        self.eidx += d

    def update_fidx(self, d: int, dname=''):
        self.fidx += d
        self.fidx_counter[dname] = self.fidx_counter.get(dname, 0) + d

    def update_iidx(self, d: int, dname=''):
        self.iidx += d
        self.iidx_counter[dname] = self.iidx_counter.get(dname, 0) + d

    def update_uidx(self, d: int, dname=''):
        self.uidx += d
        self.uidx_counter[dname] = self.uidx_counter.get(dname, 0) + d

    # special update at checkpoint: no_bad means no recording bad, patience is used for anneal
    def update_checkpoint(self, train_result: ZResult, dev_result: ZResult, record_best=True):
        sname = self.current_suffix()
        train_result = ZResult() if train_result is None else train_result
        dev_result = ZResult() if dev_result is None else dev_result
        # ----
        if_overall_best = if_best = False
        # --
        # for overall best
        if float(dev_result) > float(self.overall_best_result):
            self.overall_best_result = dev_result
            self.overall_best_point = self.cidx
            if_overall_best = True
        # --
        # for recorded best
        if float(dev_result) > float(self.best_result):
            if record_best:
                self.best_result = dev_result
                self.best_point = self.cidx
                if_best = True
        # --
        # record others
        self.chp_names.append(sname)
        self.train_results.append(train_result)
        self.dev_results.append(dev_result)
        self.cidx += 1
        return if_overall_best, if_best

# =====
# scheduled values

class SVConf(Conf):
    def __init__(self):
        self.val = 0.  # basic value
        # how to schedule the value
        self.which_idx = "cidx"  # count steps on which: aidx, eidx, iidx, uidx
        self.val_range = [0., 1.]  # [min, max]
        self.ff = "1."  # i as 'idx': lambda i: ...
        # --
        # transform on idx
        self.idx_bias = 0
        self.idx_scale = 1.0
        self.idx_int = False  # ensure transformed idx is int?
        # --

# note: SV should be stateless, that is, its value can be decided by obj at one step!
class ScheduledValue:
    def __init__(self, conf: SVConf, name: str = None):
        self.conf = conf
        self.name = name
        self._bv = conf.val  # base val
        self._minv, self._maxv = conf.val_range
        self.cur_val: float = None
        _ff = conf.ff.strip()
        self.changeable = "i" in _ff  # involving "i"
        self.ff = ZHelper.eval_ff(_ff, 'i')
        # -- init
        self._set(0)
        zlog(f"Init scheduled value {self.name} as {self.cur_val} (changeable={self.changeable}).")

    @property
    def value(self): return self.cur_val
    def __repr__(self): return "SV-%s=%s" % (self.name, self.cur_val)
    def __float__(self): return float(self.cur_val)
    def __int__(self): return int(self.cur_val)

    def transform_idx(self, idx: int):
        _conf = self.conf
        v = max(0, idx - _conf.idx_bias) / _conf.idx_scale
        if _conf.idx_int:
            v = int(v)
        return v

    def _set(self, the_idx: int):
        _conf = self.conf
        # --
        new_idx = self.transform_idx(the_idx)
        vv = self.ff(new_idx)  # ff
        vv = min(max(self._minv, vv), self._maxv)  # clamp
        vv = self._bv * vv  # base val
        # --
        old_val = self.cur_val
        self.cur_val = vv
        return old_val, self.cur_val

    # adjust at checkpoint
    def adjust_at_ckp(self, sname: str, obj: object, extra_info: str = ""):
        the_idx = getattr(obj, self.conf.which_idx)
        old_val, new_val = self._set(the_idx)
        if self.cur_val != old_val:
            zlog(f"Change scheduled value {self.name}({extra_info}) at {sname}: {old_val} => {self.cur_val}.")
        else:
            zlog(f"Keep scheduled value {self.name}({extra_info}) at {sname} as {self.cur_val}.")

# check model info
def info_trainable_parameters(model):
    import pandas as pd
    from collections import defaultdict
    # --
    def _get_key(_name):
        _fields = _name.split(".")
        while len(_fields) > 0 and _fields[-1] in ['weight', 'bias']:
            _fields.pop()
        return _fields[-1] if len(_fields)>0 else 'UNK'
    # --
    info_all, info_trainable = defaultdict(int), defaultdict(int)
    for param_name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        # --
        keys = ['ALL', _get_key(param_name)]
        for kk in keys:
            info_all[kk] += num_params
            if param.requires_grad:
                info_trainable[kk] += num_params
    # --
    all_keys = sorted(info_all.keys(), key=(lambda x: -info_all[x]))
    data = pd.DataFrame({'all': [info_all[z] for z in all_keys], 'trainable': [info_trainable[z] for z in all_keys]}, index=all_keys)
    data['perc'] = data['trainable'] / data['all']
    return data

def info_model(model):
    num_param = 0
    device_count, dtype_count = Counter(), Counter()
    for param_name, param in model.named_parameters():
        num_param += 1
        device, dtype = str(param.device), str(param.dtype)
        device_count[device] += 1
        dtype_count[dtype] += 1
    return {"model_type": str(type(model)), "num_param_item": num_param, "device": device_count, "dtype": dtype_count}

# --
class TokenizerHelper:
    @staticmethod
    def do_subtok(s: str, toker, convert_id=True, ensure_continuing=False):
        _toks0 = []
        if ensure_continuing:
            if str.isalpha(s[0]):
                s = " " + s  # add space before!
            else:  # mostly punctuations!
                s = "X" + s
                _toks0 = toker.tokenize("X")  # need some marks
        _toks = toker.tokenize(s)
        if _toks[:len(_toks0)] == _toks0:
            _toks = _toks[len(_toks0):]  # get rid of the prefix
        while len(_toks) > 0 and _toks[0] in ['▁', 'Ġ']:  # for llama and gpt2
            _toks = _toks[1:]
        # in some cases, there can be empty strings -> put the original word
        if len(_toks) == 0:
            _toks = [s]
        if convert_id:  # conver to ids
            ret = toker.convert_tokens_to_ids(_toks)
        else:
            ret = _toks
        return ret
