#

__all__ = [
    "MyCache", "ListParInput",
]

from typing import Union, Optional, Dict, Any
from contextlib import contextmanager
import torch
from transformers.cache_utils import Cache, DynamicCache, HybridCache
from mspx.utils import zlog, zwarn

# note: adapted from DynamicCache
class MyCache(Cache):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache = {}  # layer_idx -> T
        self.value_cache = {}
        # other useful values
        # input/mask: [bs, L], posi: [bs, 1]
        self.input_cache = {}  # name -> previous ones
        self._frozen = {"input": False, "states": False}  # no updating for (input, states)

    @contextmanager
    def with_frozen(self, frozen_input=True, frozen_states=True):
        _old_vals = self._frozen.copy()
        self._frozen = {"input": frozen_input, "states": frozen_states}
        try:
            yield None
        finally:
            self._frozen = _old_vals

    @property
    def t_mask(self):  # an usually one
        return self.input_cache.get("mask")

    @property
    def t_input(self):  # an usually one
        return self.input_cache.get("input")

    def get_inputs(self, keys):
        if isinstance(keys, str):
            return self.input_cache.get(keys)
        else:
            return [self.input_cache.get(k) for k in keys]

    def update_inputs(self, dim=None, **kwargs):
        new_kvs = {}
        if dim is None:  # no concat, directly update
            new_kvs.update(kwargs)
        else:
            for k, v in kwargs:
                if k not in self.input_cache:
                    new_kvs[k] = v
                else:
                    new_kvs[k] = torch.cat([self.input_cache[k], v], dim=dim)
        # --
        if not self._frozen["input"]:  # real update
            self.input_cache.update(new_kvs)
        # --
        return new_kvs

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None):
        new_key_cache, new_value_cache = {}, {}
        # --
        if layer_idx not in self.key_cache:
            new_key_cache[layer_idx] = key_states
        else:
            new_key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        if layer_idx not in self.value_cache:
            new_value_cache[layer_idx] = value_states
        else:
            new_value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        # --
        if not self._frozen["states"]:  # real update
            self.key_cache.update(new_key_cache)
            self.value_cache.update(new_value_cache)
        # --
        return new_key_cache[layer_idx], new_value_cache[layer_idx]

    # return the full seq length!
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_size(-1)

    def get_max_length(self) -> Optional[int]:
        return None

    def get_seq_size(self, dim: int):
        if self.t_mask is None:
            return 0
        else:
            return self.t_mask.shape[dim]

    # reorder or simply repeat
    def reorder_cache(self, beam_idx: torch.LongTensor = None, repeats=None, dim=0):
        assert (beam_idx is None) != (repeats is None), "Nothing get specified!"
        for dd in [self.key_cache, self.value_cache, self.input_cache]:
            if beam_idx is not None:
                new_dd = {kk: vv.index_select(dim, beam_idx.to(vv.device)) for kk, vv in dd.items()}
            else:
                new_dd = {kk: vv.repeat_interleave(repeats, dim=dim) for kk, vv in dd.items()}
            dd.update(new_dd)

    @classmethod
    def from_other_cache(cls, cache):
        ret = MyCache()
        if isinstance(cache, MyCache):
            ret.key_cache = cache.key_cache.copy()
            ret.value_cache = cache.value_cache.copy()
            ret.input_cache = cache.input_cache.copy()
        elif isinstance(cache, DynamicCache):  # note: input_cache missed!
            ret.key_cache = {i: v for i, v in enumerate(cache.key_cache)}
            ret.value_cache = {i: v for i, v in enumerate(cache.value_cache)}
        elif isinstance(cache, HybridCache):  # note: simply ignore things, this seems to be specific to gemma2!
            zwarn("Simply ignore the stuffs in HybridCache!!")
        elif isinstance(cache, tuple):
            cache0 = DynamicCache.from_legacy_cache(cache)
            ret = cls.from_other_cache(cache0)
        else:
            raise RuntimeError(f"UNK cache type {type(cache)}")
        return ret

    def clone(self):
        return self.__class__.from_other_cache(self)

    def expand_batch_size_as(self, t_input, dim=0):
        # cache_bs, new_bs = self.get_seq_size(dim), t_input.shape[dim]
        cache_bs = self.get_seq_size(dim)
        new_bs = t_input if isinstance(t_input, int) else t_input.shape[dim]
        if cache_bs > 0 and cache_bs != new_bs:
            assert new_bs > cache_bs and new_bs % cache_bs == 0, f"Bad sizes for batch expansion: {cache_bs} {new_bs}"
            # _reidx = sum([[z] * (new_bs // cache_bs) for z in range(cache_bs)], [])
            # t_reidx = torch.as_tensor(_reidx)
            # cached_vals.reorder_cache(t_reidx)  # reorder!
            self.reorder_cache(repeats=(new_bs//cache_bs), dim=dim)  # simply repeat!

# --
# note: currently simply assuming a sequence of set
class ListParInput:
    POSITION_MODE = "UNK"

    @staticmethod
    def set_position_mode(mode):
        ListParInput.POSITION_MODE = mode
        zlog(f"Set ListParInput.POSITION_MODE to {mode}")

    def __init__(self):
        self.tok_ids = []  # concatenated tokens
        self.piece_ids = []  # piece ids
        self.item_ids = []  # item ids inside one piece
        self.piece_lens = []  # List[List[int]], lengths of each piece

    def __repr__(self):
        return f"ListParInput(L={len(self.tok_ids)}): pieces={self.piece_lens}"

    def add_piece(self, piece):
        # note: piece should be List[int] or Tuple(List[int])
        ps = (piece, ) if isinstance(piece, list) else piece
        assert isinstance(ps, tuple) and all(isinstance(z, list) for z in ps)
        _last_pid = self.piece_ids[-1] if len(self.piece_ids) else -1
        _new_pid = _last_pid + 1
        _item_offset = int(len(ps) > 1)  # starting from 1 if we have multiple pieces
        for ii, one_item in enumerate(ps):
            self.tok_ids.extend(one_item)
            self.piece_ids.extend([_new_pid] * len(one_item))
            self.item_ids.extend([ii + _item_offset] * len(one_item))
        self.piece_lens.append([len(z) for z in ps])  # as the input order!
        # --

    @classmethod
    def create(cls, pieces):
        ret = cls()
        for p in pieces:
            ret.add_piece(p)
        return ret

    def get_positions(self):
        _mode = ListParInput.POSITION_MODE
        ret_posi, ret_next = [], None
        cur_posi = 0.
        mode0, mode1 = _mode[0], _mode[1:]
        for lens in self.piece_lens:
            if _mode == "order":
                _piece_size = sum(lens)
                ret_posi.extend([cur_posi+i for i in range(_piece_size)])
            else:
                if lens:
                    if mode0 == 'H':  # harmonic mean
                        _piece_size = len(lens) / sum(1./max(z, 1) for z in lens)
                    elif mode0 == 'A':  # average
                        _piece_size = sum(lens) / len(lens)
                    elif mode0 == 'M':  # max
                        _piece_size = max(lens)
                    else:
                        raise NotImplementedError(f"UNK mode0 with {_mode}")
                else:  # nothing
                    _piece_size = 0
                if mode1 == 'L':  # left
                    for one_len in lens:
                        ret_posi.extend([cur_posi+i for i in range(one_len)])
                elif mode1 == 'R':  # right
                    for one_len in lens:
                        _off = _piece_size - one_len
                        ret_posi.extend([cur_posi+i+_off for i in range(one_len)])
                elif mode1 == 'E':  # evenly spread
                    for one_len in lens:
                        _step = _piece_size / max(1, one_len)
                        ret_posi.extend([cur_posi+i*_step for i in range(one_len)])
                else:
                    raise NotImplementedError(f"UNK mode1 with {_mode}")
            cur_posi += _piece_size
            ret_next = cur_posi
        # breakpoint()
        return ret_posi, ret_next
