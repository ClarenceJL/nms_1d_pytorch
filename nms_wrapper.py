import torch
if torch.cuda.is_available():
    from nms_gpu import nms_gpu
from nms_cpu import nms_cpu


def nms(dets, thresh, use_gpu=True):
    if dets.shape[0] == 0:
        return []

    if use_gpu:
        return nms_gpu(dets, thresh)
    else:
        return nms_cpu(dets, thresh)
