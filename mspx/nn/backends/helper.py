#

__all__ = [
    "mask2idx", "mask2idx_v2", "mask2idx_v4", "masked_select_with_pad", "masked_select_with_pad_by_idxes",
    "add_paddings", "gather_with_negative_idxes", "gather_first_dims", "extend_idxes", "unsqueeze_expand", "mark_first_one",
    "aggr_tensor",
]

import numpy as np
import torch


# note: for mask->idx: 1) argsort, 2) pad 1s + nonzero, 3) loop, 4) flatten and scatter, => still v2 is the fastest!
# the inputs should be 1. or 0. (float); [*, L] -> [*, max-count]
def mask2idx_v4(t_mask, pad=0, dim=-1, max_count=None):
    _device = t_mask.device
    input_shape = t_mask.shape  # [*, L]
    # --
    # judge zero-shape
    if t_mask.numel() == 0:
        _shape = list(input_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape).to(t_mask)  # [*, 1, *], put an one here!
        return zz.long(), zz
    # --
    assert dim == -1, "For simplicity, just handle the dim==-1 case ..."
    mask_b = (t_mask.long() > 0).view([-1, input_shape[-1]])  # [**, L]
    mask_l = mask_b.long()  # [**, L]
    # note: must be careful with given max_count, might cause out-of-index or implicit errors!!
    if max_count is None:  # if no information given, calculate it!
        max_count = mask_l.sum(-1).max().item()
    ret_numel = len(mask_l) * max_count
    tmp_arange = torch.arange(input_shape[-1], device=_device).unsqueeze(0).expand(len(mask_l), -1)  # [**, L]
    tmp_ret_idxes, tmp_valid_mask = \
        torch.full([ret_numel], pad, dtype=torch.long, device=_device), torch.full([ret_numel], 0, dtype=torch.long, device=_device)
    tmp_global_indexes = (mask_l.cumsum(-1) - 1).clamp(min=0) + torch.arange(len(mask_l), device=_device).unsqueeze(-1) * max_count  # [**, L]
    tmp_global_indexes_valid = tmp_global_indexes[mask_b]  # [***]
    tmp_ret_idxes[tmp_global_indexes_valid] = tmp_arange[mask_b]  # assign only valid ones
    tmp_valid_mask[tmp_global_indexes_valid] = 1
    # reshape back
    ret_shape = list(input_shape)
    ret_shape[dim] = max_count
    ret_idxes, valid_mask = tmp_ret_idxes.view(ret_shape), tmp_valid_mask.view(ret_shape)
    return ret_idxes, valid_mask

def mask2idx_v2(t_mask, pad=0, dim=-1):
    mask_shape = t_mask.shape  # [*, L, *]
    # --
    # judge zero-shape
    if t_mask.numel() == 0:
        _shape = list(mask_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape).to(t_mask)  # [*, 1, *], put an one here!
        return zz.long(), zz
    # --
    mask_f = t_mask.long()  # [*, L, *]
    # get max counts
    counts = mask_f.sum(dim=dim, keepdims=True)  # [*, 1, *]
    # max_count = max(1, int(counts.max().item()))  # M (at least one)
    max_count = int(counts.max().item())
    if max_count <= 0:
        _shape = list(mask_shape)
        _shape[dim] = 0
        zz = torch.zeros(_shape).to(t_mask)  # [*, 0, *], put zero here!
        return zz.long(), zz
    padding_counts = max_count - counts  # [*, 1, *]
    max_padding_count = int(padding_counts.max().item())  # int, the max count of padding
    # pad and concat
    _arange_idx = torch.arange(max_padding_count).to(padding_counts)  # [mp]
    to_expand_dim = (-dim-1) if dim<0 else (len(mask_shape)-1-dim)
    pad_t = (_arange_idx.view([max_padding_count]+[1]*to_expand_dim) < padding_counts).to(mask_f)  # [*, mp, *]
    concat_t = torch.cat([mask_f, pad_t], dim)  # [*, L+mp, *]
    # nonzero and extract
    final_shape = list(mask_shape)
    final_shape[dim] = max_count
    if dim != -1 or dim != len(mask_shape) - 1:
        final_shape = final_shape[:dim] + final_shape[dim:][1:] + [max_count]
        _p0 = list(range(len(mask_shape)))  # [0, N)
        _p1 = _p0[:dim] + _p0[dim:][1:] + [dim]
        _p2 = _p0[:dim] + [-1] + [z-1 for z in _p0[dim:][1:]]
        ret_idxes = concat_t.permute(_p1).nonzero(as_tuple=False)[:, -1].view(final_shape).permute(_p2)
    else:
        ret_idxes = concat_t.nonzero(as_tuple=False)[:, dim].view(final_shape)  # [*, M, *]
    # get valid mask and set pad for invalid ones
    max_len = mask_shape[dim]  # L
    valid_mask = (ret_idxes < max_len).long()  # [*, M, *]
    ret_idxes[valid_mask<=0] = pad
    return ret_idxes, valid_mask

# note: still using v2 since it seems to be better
mask2idx = mask2idx_v2

# masked select
def masked_select_with_pad(t_val, t_mask, pad, dim=-1):
    sel_idxes, sel_valid = mask2idx(t_mask, pad=0, dim=dim)  # [..., S, ...]
    ret = masked_select_with_pad_by_idxes(t_val, sel_idxes, sel_valid, pad, dim=dim)
    return ret, sel_valid

def masked_select_with_pad_by_idxes(t_val, sel_idxes, sel_valid, pad, dim=-1):
    ret0 = t_val.gather(dim, sel_idxes)  # ...
    ret = torch.where((sel_valid > 0), ret0, pad)  # replace invalid ones!
    return ret

# an easier tool to add paddings
def add_paddings(t, paddings=(None, None), dim=-1):
    pad0, pad1 = paddings
    ndim_t = len(list(t.shape))
    if pad0 is not None and len(pad0):
        pad0 = torch.as_tensor(pad0).to(t)
        _shape0 = list(pad0.shape)
        _shape0 = [1] * (ndim_t - len(_shape0)) + _shape0  # add pre dims
    else:
        pad0 = None
    if pad1 is not None and len(pad1):
        pad1 = torch.as_tensor(pad1).to(t)
        _shape1 = list(pad1.shape)
        _shape1 = [1] * (ndim_t - len(_shape1)) + _shape1  # add pre dims
    else:
        pad1 = None
    shape_expand = list(t.shape)
    shape_expand[dim] = -1  # extend shape
    ts = []
    if pad0 is not None:
        ts.append(pad0.view(_shape0).expand(shape_expand))
    ts.append(t)
    if pad1 is not None:
        ts.append(pad1.view(_shape1).expand(shape_expand))
    if len(ts) > 1:
        ret = torch.concat(ts, dim=dim)
    else:
        ret = t
    return ret

# gather with negative idxes
def gather_with_negative_idxes(input, dim, index):
    full_dim = input.shape[dim]
    index = torch.where((index >= 0), index, (full_dim + index))
    ret = torch.gather(input, dim, index)
    return ret

# special gather for the first several dims
# t: [s1, s2, ..., sn-1, sn, ...]; idx: [s1, s2, ..., sn-1, k]
def gather_first_dims(t_val, t_idxes, dim=-1):
    t_shape = list(t_val.shape)  # [..., L, ...]
    if dim < 0:
        dim = len(t_shape) + dim
    idx_shape = list(t_idxes.shape)
    assert t_shape[:dim] == idx_shape[:-1]
    # flatten and index select
    t_shape0, t_shape1 = t_shape[:dim+1], t_shape[dim+1:]
    flatten_t = t_val.view([np.prod(t_shape0).item()] + t_shape1)  # [s1*...*sn, ...]
    basis_t = torch.arange(np.prod(t_shape0[:-1]), device=t_idxes.device).view(t_shape0[:-1]) * t_shape0[-1]  # [s1, ..., sn-1]
    basis_t = (basis_t.unsqueeze(-1) + t_idxes).view(-1)  # [*]
    output_t0 = torch.index_select(flatten_t, dim=0, index=basis_t)  # [*, ...]
    return output_t0.view(idx_shape + t_shape1)

# extend idxes
def extend_idxes(t_idxes, v: int, dim=-1, add_dim=False):
    if add_dim:
        t_idxes = t_idxes.unsqueeze(dim)
    _shape = list(t_idxes.shape)  # [..., 1, ...]
    _shape[dim] = v
    ret = torch.zeros(_shape).to(t_idxes.device)
    ret.scatter_(dim, t_idxes, 1.)
    return ret

# similar to repeat and repeat_interleave
def unsqueeze_expand(t, alpha: int, dim: int, interleave=False, reshape=False):
    assert alpha > 0, "Alpha cannot be negative!"
    _shape = list(t.shape)  # [..., L, ...]
    if dim < 0:  # make it positive
        dim = len(_shape) + dim
    _dim0 = dim
    _shape2 = [-1] * (len(_shape)+1)
    _dim2 = dim+1 if interleave else dim  # [..., *L, A, ...] or [..., A, *L, ...]
    _shape2[_dim2] = alpha
    ret = t.unsqueeze(_dim2).expand(_shape2)
    if reshape:
        _shape[_dim0] = _shape[_dim0] * alpha
        ret = ret.reshape(_shape)  # [..., L*A, ...]
    return ret

# get the first one from the left
def mark_first_one(t_input, dim=-1, rightmost=False):
    vv = (t_input > 0)  # make things bool
    if rightmost:
        vv = torch.flip(vv, dims=[dim])  # flip left/right
    cc = (vv.cumsum(dim=dim) > 0)
    # now we only need to find the switch of 0 -> 1
    _len = cc.shape[dim]
    ret0 = cc.narrow(dim, 0, 1)  # [..., 1, ...]
    ret1 = (~ cc.narrow(dim, 0, _len-1) & cc.narrow(dim, 1, _len-1))  # [..., _len-1, ...]
    ret = torch.concat([ret0, ret1], dim=dim)
    if rightmost:
        ret = torch.flip(ret, dims=[dim])
    return ret.long()

# helper function
def aggr_tensor(t, aggr_dims, aggr_topk: int):
    _shape = list(t.shape)
    if not isinstance(aggr_dims, list):
        assert isinstance(aggr_dims, int)
        aggr_dims = [aggr_dims]
    aggr_dims = [(z if z>=0 else len(_shape)+z) for z in aggr_dims]
    # first transpose all the aggr dims to the end
    assert len(aggr_dims) == len(set(aggr_dims)) and all((z>=0 and z<len(_shape)) for z in aggr_dims), f"Strange aggr_dims {aggr_dims}"
    all_dims = list(range(len(_shape)))
    non_aggr_dims = [z for z in all_dims if z not in aggr_dims]  # [non_aggr..., aggr...]
    t_permute = torch.permute(t, non_aggr_dims + aggr_dims).reshape([_shape[z] for z in non_aggr_dims] + [-1])  # [non_aggr..., -1]
    # get topk vals
    if aggr_topk > 0:
        t_aggr_score = t_permute.topk(min(aggr_topk, t_permute.shape[-1]), dim=-1)[0].sum(-1)  # [non_aggr...]
    else:  # simply average
        t_aggr_score = t_permute.mean(-1)
    # permute back
    t_aggr_score2 = t_aggr_score.view([_shape[z] for z in non_aggr_dims] + [1] * len(aggr_dims))  # [non_aggr..., 1...]
    _rev_idx = sorted([(vv, ii) for ii, vv in enumerate(non_aggr_dims + aggr_dims)])
    t_ret = torch.permute(t_aggr_score2, [z[1] for z in _rev_idx])
    return t_ret
