#

__all__ = [
    "BaseModifierConf", "BaseModifier", "get_mod_confs",
]

from collections import OrderedDict
from mspx.utils import Conf, Configurable, ZHelper, zwarn

class BaseModifierConf(Conf):
    def __init__(self):
        pass

@BaseModifierConf.conf_rd()
class BaseModifier(Configurable):
    def __init__(self, conf: BaseModifierConf):
        super().__init__(conf)

    # modify model (inplace)
    def modify_model(self, model, toker):
        pass

# useful for conf
def get_mod_confs(conf: Conf, s: str):
    # --
    def _valid_cls(ss, vv):
        return isinstance(vv, type) and issubclass(vv, BaseModifierConf) and (vv is not BaseModifierConf) and (not ss.startswith("_"))
    # --
    def _subconf_ff(name):
        base_mod = ".".join(__name__.split(".")[:-1])
        cls = ZHelper.find_item_in_module(f"{base_mod}.{name}", _valid_cls, only_one=True)
        return cls()
    # --
    ret = conf.get_subconf(s, ff=_subconf_ff, name_prefix="mod_")
    return ret
