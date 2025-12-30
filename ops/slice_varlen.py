import torch
from torch import Tensor


def slice_varlen(x_joint: Tensor, cu_joint: Tensor, cu0: Tensor):
    return slice_varlen_ref(x_joint, cu_joint, cu0)


def slice_varlen_ref(x_joint: Tensor, cu_joint: Tensor, cu0: Tensor):
    joint_sizes = cu_joint.diff().tolist()
    sizes0 = cu0.diff().tolist()
    return torch.cat([x[:size] for x, size in zip(x_joint.split(joint_sizes), sizes0)], dim=0)
