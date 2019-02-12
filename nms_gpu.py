from __future__ import absolute_import
import torch
from _ext import nms


def sort_by_column(t, series_id, dim=0, descending=True):
    assert t.dim() == 2 and dim in [0, 1]
    series_id = int(series_id)
    assert series_id >= 0
    if dim == 0:
        assert series_id < t.size(1)
        _, ind = torch.sort(t[:, series_id], descending=descending)
        return t[ind], ind
    else:
        assert series_id < t.size(0)
        _, ind = torch.sort(t[series_id], descending=descending)
        return t[:, ind], ind


def nms_gpu(dets, thresh):
    # sort by score (descending)
    dets, ind = sort_by_column(dets, 2, dim=0, descending=True)

    keep = dets.new(dets.size(0), 1).zero_().int()
    num_out = dets.new(1).zero_().int()
    nms.nms_cuda(keep, dets, num_out, thresh)
    keep = keep[:num_out[0], 0].long()
    return ind[keep]


# if __name__=="__main__":
#     a = torch.Tensor(
#         [[0.0785, 1.5267, -0.8521, 0.4065],
#          [0.1598, 0.0788, -0.0745, -1.2700],
#          [1.2208, 1.0722, -0.7064, 1.2564],
#          [0.0669, -0.2318, -0.8229, -0.9280]]
#     )
#     print(sort_by_column(a, 3))