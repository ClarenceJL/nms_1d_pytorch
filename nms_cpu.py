from __future__ import absolute_import
import numpy as np
import torch


def nms_cpu(dets, thresh):
    dets = dets.detach().numpy()
    start = dets[:, 0]
    end = dets[:, 1]
    scores = dets[:, 2]

    length = end - start + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        start_ub = np.maximum(start[i], start[order[1:]])
        end_lb = np.minimum(end[i], end[order[1:]])

        inter = np.maximum(0.0, end_lb - start_ub + 1)

        iou = inter / (length[i] + length[order[1:]] - inter)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]   # skip the overlapping segments

    return torch.IntTensor(keep)
