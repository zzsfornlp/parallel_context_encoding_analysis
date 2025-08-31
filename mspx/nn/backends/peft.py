#

# helpers for peft modules
# note: again modify the module's method to make things easier

from typing import List
from contextlib import contextmanager
from functools import partial
import math

import torch
from torch import nn

from mspx.utils import Random, Conf, ZHelper, zlog
from .bktr import replace_module, setattr_borrow, get_first_device

# helper to wrap (reuse) a module!
class ModuleWrapper(nn.Module):
    def __init__(self, base_mod, base_borrow=True, base_detach=True):
        super().__init__()
        if base_borrow:
            setattr_borrow(self, "base_mod", base_mod)  # simply borrowing
        else:
            self.base_mod = base_mod
        self.base_detach = base_detach

    def forward(self, *args, **kwargs):
        ret = self.base_mod.forward(*args, **kwargs)
        if self.base_detach:
            ret = ret.detach()
        return ret

class PeftConf(Conf):
    def __init__(self):
        self.peft_param_prefix = "peft_"
        self.peft_method = "lora"  # nope/lora/...
        # lora
        self.lora_r = 8  # lora rank
        self.lora_alpha = 8.  # scale
        self.lora_dropout = 0.  # dropout

# note: use a specific module to separate parameters!
class MyPeftMod(nn.Module):
    def __init__(self, conf: PeftConf, base_mod: nn.Module):
        super().__init__()
        self.conf = conf
        self.change_module(base_mod)  # store related info!
        self.peft_enabled = True
        # --

    def change_module(self, mod):
        conf: PeftConf = self.conf
        # --
        if conf.peft_method == "nope":
            return mod  # nothing changed here!
        elif conf.peft_method == "lora":
            self.setup_lora(mod)
        # --
        setattr_borrow(mod, '_orig_nopeft_forward', mod.forward)
        mod.peft_mod = self  # store self!
        setattr_borrow(mod, 'forward', partial(self.new_forward, mod=mod), assert_nonexist=False)  # setup a wrapper function
        return mod
        # --

    def new_forward(self, *args, mod=None, **kwargs):
        t_output = mod._orig_nopeft_forward(*args, **kwargs)
        t_delta = mod.peft_mod(*args, **kwargs)  # call this since it may be fsdp wrapped
        if t_delta is not None:
            t_output = t_output + t_delta
        # --
        return t_output

    # need the forward to gather stuffs!
    def forward(self, *args, **kwargs):
        conf: PeftConf = self.conf
        if not self.peft_enabled:
            return None
        if conf.peft_method == "nope":
            t_delta = None  # no need to do anything!
        elif conf.peft_method == "lora":
            t_delta = self.forward_lora(*args, **kwargs)
        else:
            raise NotImplementedError(f"UNK peft_method = {conf.peft_method}")
        return t_delta

    # --
    # detailed helpers

    def setup_lora(self, mod):
        conf: PeftConf = self.conf
        # --
        raw_mod = mod.base_mod if isinstance(mod, ModuleWrapper) else mod
        if isinstance(raw_mod, nn.Linear):
            _device = raw_mod.weight.device
            shape_weight = raw_mod.weight.shape  # [output, input]
            lora_A, lora_B = nn.Linear(shape_weight[1], conf.lora_r, bias=False).to(_device), nn.Linear(conf.lora_r, shape_weight[0], bias=False).to(_device)
            lora_dropout = nn.Dropout(conf.lora_dropout) if conf.lora_dropout > 0. else (lambda x: x)
            lora_scale = conf.lora_alpha / conf.lora_r
            with torch.no_grad():  # init
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)
            # note: set to self!
            for k, v in zip(["lora_A", "lora_B", "lora_dropout", "lora_scale"], [lora_A, lora_B, lora_dropout, lora_scale]):
                k2 = conf.peft_param_prefix + k
                assert not hasattr(self, k2)
                setattr(self, k2, v)  # add new params (to the upper mod)!
        else:
            raise NotImplementedError(f"Not supported for module {type(raw_mod)}")

    def forward_lora(self, input):
        conf: PeftConf = self.conf
        # --
        lora_A, lora_B, lora_dropout, lora_scale = [getattr(self, conf.peft_param_prefix + k) for k in ["lora_A", "lora_B", "lora_dropout", "lora_scale"]]
        t_input = input.to(lora_A.weight.dtype)
        t_delta = lora_B(lora_A(lora_dropout(t_input))) * lora_scale
        t_delta = t_delta.to(input.dtype)
        return t_delta

class PeftHelper:
    def __init__(self, conf: PeftConf):
        self.conf = conf
        self.peft_enabled = True  # for all the new peft modules added in this instance
        self.all_peft_mods = []

    def get_peft_model(self, model, target_names: List[str]):
        if target_names:
            replace_module(model, (lambda m, p: ZHelper.match_name(".".join(p), target_names)), (lambda m, p: self.change_module(m)), inplace=True)
        return model

    def set_peft_enabled(self, enabled=True):
        self.peft_enabled = enabled
        for peft_mod in self.all_peft_mods:
            peft_mod.peft_enabled = enabled
        zlog(f"Set peft_mods {self.all_peft_mods} enabled = {enabled}")

    @contextmanager
    def env_set_peft_enabled(self, enabled=True):
        old_v = self.peft_enabled
        self.set_peft_enabled(enabled)
        try:
            yield None
        finally:
            self.set_peft_enabled(old_v)

    def change_module(self, mod):
        peft_mod = MyPeftMod(self.conf, mod)  # modifications are inplace!
        self.all_peft_mods.append(peft_mod)
        return mod
