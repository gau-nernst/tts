import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from triton.testing import do_bench


@triton.jit
def _memcpy(in_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)

    for _ in range(tl.cdiv(size, BLOCK_SIZE)):
        mask = offs < size
        data = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, data, mask)
        offs += BLOCK_SIZE


@triton.jit
def _kernel(
    embd0_ptr,  # [L0, D_embed]
    embd1_ptr,  # [L1, D_embed]
    rope0_ptr,  # [L0, D_rope]
    rope1_ptr,  # [L1, D_rope]
    cu0_ptr,  # [B+1]
    cu1_ptr,  # [B+1]
    embd_out_ptr,  # [L0+L1, D_embed]
    rope_out_ptr,  # [L0+L1, D_rope]
    cu_out_ptr,  # [B+1]
    D_embed: tl.constexpr,
    D_rope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 4096,
):
    batch_id = tl.program_id(0)

    start0 = tl.load(cu0_ptr + batch_id)
    start1 = tl.load(cu1_ptr + batch_id)

    end0 = tl.load(cu0_ptr + batch_id + 1)
    end1 = tl.load(cu1_ptr + batch_id + 1)

    start_out = start0 + start1

    embd0_ptr += start0 * D_embed
    embd1_ptr += start1 * D_embed
    embd_out_ptr += start_out * D_embed

    rope0_ptr += start0 * D_rope
    rope1_ptr += start1 * D_rope
    rope_out_ptr += start_out * D_rope

    l0 = end0 - start0
    l1 = end1 - start1

    # handle cumsum
    tl.store(cu_out_ptr + batch_id, start_out)
    if batch_id == tl.num_programs(0) - 1:
        tl.store(cu_out_ptr + batch_id + 1, end0 + end1)

    _memcpy(embd0_ptr, embd_out_ptr, l0 * D_embed, BLOCK_SIZE)
    _memcpy(rope0_ptr, rope_out_ptr, l0 * D_rope, BLOCK_SIZE)

    embd_out_ptr += l0 * D_embed
    rope_out_ptr += l0 * D_rope
    _memcpy(embd1_ptr, embd_out_ptr, l1 * D_embed, BLOCK_SIZE)
    _memcpy(rope1_ptr, rope_out_ptr, l1 * D_rope, BLOCK_SIZE)


def merge_varlen(embd0: Tensor, embd1: Tensor, rope0: Tensor, rope1: Tensor, cu0: Tensor, cu1: Tensor):
    # we intentionally use cu (cumulative sequence length) as the auxiliary data
    # for merging because varlen attention also uses it.
    assert embd0.is_contiguous()
    assert embd1.is_contiguous()
    assert rope0.is_contiguous()
    assert rope1.is_contiguous()
    assert cu0.is_contiguous()
    assert cu1.is_contiguous()
    L0, D_embed = embd0.shape
    L1, D_rope = rope1.shape
    B = cu0.shape[0] - 1

    embd_out = embd0.new_empty(L0 + L1, D_embed)
    rope_out = rope0.new_empty(L0 + L1, D_rope)
    cu_out = torch.empty_like(cu0)

    _kernel[(B,)](embd0, embd1, rope0, rope1, cu0, cu1, embd_out, rope_out, cu_out, D_embed, D_rope)

    return embd_out, rope_out, cu_out


def merge_varlen_ref(
    embd0: Tensor, embd1: Tensor, rope0: Tensor, rope1: Tensor, cu0: Tensor, cu1: Tensor, sizes0=None, sizes1=None
):
    if sizes0 is None:
        sizes0 = cu0.diff().tolist()
    if sizes1 is None:
        sizes1 = cu1.diff().tolist()

    embd = torch.cat([x for pair in zip(embd0.split(sizes0), embd1.split(sizes1)) for x in pair], dim=0)
    rope = torch.cat([x for pair in zip(rope0.split(sizes0), rope1.split(sizes1)) for x in pair], dim=0)
    return embd, rope, cu0 + cu1


def generate_test_data(B: int, D_embed: int, min_length: int, max_length: int, dtype: torch.dtype = torch.bfloat16):
    D_rope = 128

    sizes0 = torch.randint(min_length, max_length, size=(B,), device="cuda")
    sizes1 = torch.randint(min_length, max_length, size=(B,), device="cuda")

    L0 = sizes0.sum().item()
    L1 = sizes1.sum().item()

    embd0 = torch.randn(L0, D_embed, dtype=dtype, device="cuda")
    embd1 = torch.randn(L1, D_embed, dtype=dtype, device="cuda")

    rope0 = torch.randn(L0, D_rope, dtype=dtype, device="cuda")
    rope1 = torch.randn(L1, D_rope, dtype=dtype, device="cuda")

    cu0 = F.pad(sizes0.cumsum(dim=0), (1, 0))
    cu1 = F.pad(sizes1.cumsum(dim=0), (1, 0))

    return embd0, embd1, rope0, rope1, cu0, cu1


if __name__ == "__main__":
    import statistics

    # on sufficiently large shape, our kernel is actually slower...
    B = 128
    dim = 256
    min_length = 10
    max_length = 200

    args = generate_test_data(B, dim, min_length, max_length)
    out_ref = merge_varlen_ref(*args)
    out = merge_varlen(*args)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)

    def benchmark(f, *args):
        out_list = f(*args)
        latency_us = do_bench(lambda: f(*args), warmup=50, rep=100, return_mode="median") * 1e3

        in_bytes = sum(getattr(x, "nbytes", 0) for x in args)
        out_bytes = sum(x.nbytes for x in out_list)
        membw = (in_bytes + out_bytes) / (latency_us * 1e-6) * 1e-9
        return latency_us, membw

    data_ref = []
    data = []

    for _ in range(10):
        *args, cu0, cu1 = generate_test_data(B, dim, min_length, max_length)
        sizes0 = cu0.diff().tolist()
        sizes1 = cu1.diff().tolist()
        data_ref.append(benchmark(merge_varlen_ref, *args, cu0, cu1))
        data.append(benchmark(merge_varlen, *args, cu0, cu1))

    def get_stat(data: list):
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        return f"{mean:6.2f} ± {std:5.2f}"

    latency_ref, membw_ref = zip(*data_ref)
    print(f"Reference: {get_stat(latency_ref)} us, {get_stat(membw_ref)} GB/s")

    latency, membw = zip(*data)
    print(f"Triton:    {get_stat(latency)} us, {get_stat(membw)} GB/s")
