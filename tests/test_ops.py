def test_merge_varlen():
    from ops.merge_varlen import generate_test_data, merge_varlen, merge_varlen_ref

    B = 10
    D = 32

    x0, x1, cu0, cu1 = generate_test_data(B, D, 5, 30)

    actual = merge_varlen(x0, x1, cu0, cu1)
    expected = merge_varlen_ref(x0, x1, cu0, cu1)
    assert (actual == expected).all()  # exact match
