#


__all__ = [
    "MyCheckerModifierConf", "MyCheckerModifier"
]

import re
from typing import Optional, Tuple, List, Sequence
from collections import OrderedDict, defaultdict
import math
from functools import partial

from mspx.nn import BK
from mspx.utils import zwarn_once, Conf, zwarn, Constants, zlog, ZDEBUG
from ._base import BaseModifierConf, BaseModifier

# --
class MyCheckerModifierConf(BaseModifierConf):
    def __init__(self):
        super().__init__()
        self.check_module_names = [""]  # modules that contain any of these names

@MyCheckerModifierConf.conf_rd()
class MyCheckerModifier(BaseModifier):
    def __init__(self, conf: MyCheckerModifierConf):
        super().__init__(conf)
        self.stat = defaultdict(OrderedDict)

    def modify_model(self, model, toker):
        _mkeys = self.conf.check_module_names
        BK.replace_module(model, (lambda m, p: (model is not m) and any(z in m.__class__.__name__ for z in _mkeys)), (lambda m, p: self.modify_module(m, p)), inplace=True)

    # --
    # modify one module to store extra information to check
    def modify_module(self, module, path):
        # setup a wrapper function
        path_copied = path.copy()
        BK.setattr_borrow(module, '_orig_nocheck_forward', module.forward)
        BK.setattr_borrow(module, 'forward', partial(self._new_forward, module=module, path=path_copied), assert_nonexist=False)

    def _new_forward(self, *args, module=None, path=None, **kwargs):
        orig_results = module._orig_nocheck_forward(*args, **kwargs)
        results = (orig_results, ) if not isinstance(orig_results, Sequence) else orig_results
        for ii, vv in enumerate(results):
            if BK.is_tensor(vv):
                _path = ".".join(path) + f"_{ii}"
                _one_stat = {"std": vv.std(-1).mean().item()}  # add one checking!
                for kk, ss in _one_stat.items():
                    if _path not in self.stat[kk]:
                        self.stat[kk][_path] = []
                    self.stat[kk][_path].append(ss)
        return orig_results
    # --

    def summarize_stat(self):
        for one_name, one_stats in self.stat.items():
            zlog(f"#--\nSummarize stat for {one_name}")
            for one_path, one_stat in one_stats.items():
                _sum, _len = sum(one_stat), len(one_stat)
                zlog(f"{one_path}/{one_name} = {_sum}/{_len}={_sum/max(1.,_len):.4f}")
