import pytest

from tokenizer import DOT, GRAVE, HOOK, TILDE, tokenize_text


@pytest.mark.parametrize(
    "input,expected",
    [
        ("Xin chào!", "Xin chao" + GRAVE + "!"),
        ("ở TP.HCM. Học Đà Nẵng", "ơ" + HOOK + " TP.HCM. Hoc" + DOT + " Đa" + GRAVE + " Năng" + TILDE),
    ],
)
def test_tokenizer(input: str, expected: str):
    output = tokenize_text(input)
    assert tuple(output) == tuple(expected)
