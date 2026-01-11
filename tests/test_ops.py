import torch


def test_merge_varlen():
    from ops.merge_varlen_op import generate_test_data, merge_varlen_forward, merge_varlen_ref

    B = 10
    D = 32

    embd0, embd1, rope0, rope1, cu0, cu1 = generate_test_data(B, D, 5, 30)

    embd, rope, cu = merge_varlen_forward(embd0, embd1, rope0, rope1, cu0, cu1)
    embd_ref, rope_ref, cu_ref = merge_varlen_ref(embd0, embd1, rope0, rope1, cu0, cu1)

    assert (embd == embd_ref).all()
    assert (rope == rope_ref).all()
    assert (cu == cu_ref).all()
