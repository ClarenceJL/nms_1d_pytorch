import torch
from nms_wrapper import nms

if __name__ == "__main__":
    # create a series of proposals
    centers = torch.empty(100).uniform_(0, 64)
    lengths = torch.clamp(6 + torch.randn(100) * 6, 2, 10)
    starts = torch.clamp(centers - lengths / 2.0, 0, 64)
    ends = torch.clamp(centers + lengths / 2.0, 0, 64)
    scores = torch.empty(100).uniform_(0, 1)
    props = torch.stack((starts, ends, scores), dim=-1)

    result_cpu = nms(props, 0.5, use_gpu=False)
    print(result_cpu.size(0))
    print(result_cpu)

    if torch.cuda.is_available():
        props = props.cuda()
        result_gpu = nms(props, 0.5, use_gpu=True)
        print(result_gpu.size(0))
        print(result_gpu)

