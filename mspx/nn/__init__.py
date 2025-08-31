#

# for nn

from .backends import BK, NIConf

# --
def init(cc: NIConf, **kwargs):
    BK.init(cc, **kwargs)
    pass
