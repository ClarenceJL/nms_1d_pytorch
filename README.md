# nms_1d_pytorch
1-dim non-maximum suppression (NMS) in PyTorch, supporting both CPU and CUDA

* Revised from the 2-dim NMS implementation by [jwyang](https://github.com/jwyang) in [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/master/lib/model/nms) with only minor changes.
* Fix bugs in nms_gpu.py so that it behaves the same as num_cpu when input proposals are not sorted
