import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.testing import do_bench


@triton.jit
def _memcpy_2d(in_ptr, out_ptr, length, BLOCK_L: tl.constexpr, D: tl.constexpr):
    l_offsets = tl.arange(0, BLOCK_L)[:, None]
    d_offsets = tl.arange(0, D)[None, :]

    for i in range(tl.cdiv(length, BLOCK_L)):
        mask = l_offsets < length
        data = tl.load(in_ptr + l_offsets * D + d_offsets, mask=mask)
        tl.store(out_ptr + l_offsets * D + d_offsets, data, mask=mask)
        l_offsets += BLOCK_L


@triton.jit
def _kernel(
    x0_ptr,  # [L0, D]
    x1_ptr,  # [L1, D]
    cu0_ptr,  # [B+1]
    cu1_ptr,  # [B+1]
    out_ptr,  # [L0+L1, D]
    D: tl.constexpr,
    BLOCK_L: tl.constexpr = 32,  # we might want to pick this dynamically
):
    batch_id = tl.program_id(0)

    start0 = tl.load(cu0_ptr + batch_id)
    start1 = tl.load(cu1_ptr + batch_id)

    l0 = tl.load(cu0_ptr + batch_id + 1) - start0
    l1 = tl.load(cu1_ptr + batch_id + 1) - start1

    x0_ptr += start0 * D
    x1_ptr += start1 * D
    out_ptr += (start0 + start1) * D

    _memcpy_2d(x0_ptr, out_ptr, l0, BLOCK_L, D)

    out_ptr += l0 * D
    _memcpy_2d(x1_ptr, out_ptr, l1, BLOCK_L, D)


def merge_varlen(x0: Tensor, x1: Tensor, cu0: Tensor, cu1: Tensor):
    # we intentionally use cu (cumulative sequence length) as the auxiliary data
    # for merging because varlen attention also uses it.
    assert x0.is_contiguous()
    assert x1.is_contiguous()
    assert cu0.is_contiguous()
    assert cu1.is_contiguous()
    L0, D = x0.shape
    L1, _ = x1.shape
    B = cu0.shape[0] - 1

    out = x0.new_empty(L0 + L1, D)
    _kernel[(B,)](x0, x1, cu0, cu1, out, D)
    return out


def merge_varlen_ref(x0: Tensor, x1: Tensor, cu0: Tensor, cu1: Tensor):
    x0_list = x0.split(cu0.diff().tolist())
    x1_list = x1.split(cu1.diff().tolist())
    return torch.cat([x for pair in zip(x0_list, x1_list) for x in pair], dim=0)


def generate_test_data(B: int, dim: int, min_length: int, max_length: int, dtype: torch.dtype = torch.bfloat16):
    x0_list = []
    x1_list = []
    cu0_list = [0]
    cu1_list = [0]

    for _ in range(B):
        l0 = torch.randint(5, 30, size=(1,)).item()
        l1 = torch.randint(5, 30, size=(1,)).item()
        x0_list.append(torch.randn(l0, dim, dtype=dtype, device="cuda"))
        x1_list.append(torch.randn(l1, dim, dtype=dtype, device="cuda"))
        cu0_list.append(cu0_list[-1] + l0)
        cu1_list.append(cu1_list[-1] + l1)

    x0 = torch.cat(x0_list, dim=0)
    x1 = torch.cat(x1_list, dim=0)
    cu0 = torch.tensor(cu0_list, device="cuda")
    cu1 = torch.tensor(cu1_list, device="cuda")
    return x0, x1, cu0, cu1


if __name__ == "__main__":
    import statistics

    B = 128
    dim = 256
    min_length = 10
    max_length = 200

    x0, x1, cu0, cu1 = generate_test_data(B, dim, min_length, max_length)
    out_ref = merge_varlen_ref(x0, x1, cu0, cu1)
    out = merge_varlen(x0, x1, cu0, cu1)
    assert (out == out_ref).all()

    def benchmark(f, *args):
        latency_us = do_bench(lambda: f(*args), warmup=50, rep=100, return_mode="median") * 1e3
        membw = (x0.nbytes + x1.nbytes + cu0.nbytes + cu1.nbytes + out_ref.nbytes) / (latency_us * 1e-6) * 1e-9
        return latency_us, membw

    data_ref = []
    data = []

    for _ in range(10):
        x0, x1, cu0, cu1 = generate_test_data(B, dim, min_length, max_length)
        data_ref.append(benchmark(merge_varlen_ref, x0, x1, cu0, cu1))
        data.append(benchmark(merge_varlen, x0, x1, cu0, cu1))

    def get_stat(data: list):
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        return f"{mean:6.2f} ± {std:5.2f}"

    latency_ref, membw_ref = zip(*data_ref)
    print(f"Reference: {get_stat(latency_ref)} us, {get_stat(membw_ref)} GB/s")

    latency, membw = zip(*data)
    print(f"Triton:    {get_stat(latency)} us, {get_stat(membw)} GB/s")
