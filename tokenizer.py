import re
import unicodedata

# Tone marks (NFD)
# https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-7/#G19663
ACUTE = "\u0301"  # sắc
GRAVE = "\u0300"  # huyền
HOOK = "\u0309"  # hỏi
TILDE = "\u0303"  # ngã
DOT = "\u0323"  # nặng
TONES = {ACUTE, GRAVE, HOOK, TILDE, DOT}


def decompose_tone(ch: str):
    # assume input is NFC
    assert len(ch) == 1

    # convert the character to decomposed form
    # when we iterate over this, it becomes [o,  ̂,  ́]
    nfd = unicodedata.normalize("NFD", ch)
    out = []
    tone = None
    for c in nfd:
        if c in TONES:
            assert tone is None, list(nfd)  # there should be only 1 tone
            tone = c
        else:
            out.append(c)

    if tone is not None:
        # rebuild NFC character without tone
        ch = unicodedata.normalize("NFC", "".join(out))

    return ch, tone


def tokenize_text(text: str):
    output = []

    # Split on non-characters i.e. spaces and punctuations
    # keep them as tokens
    words = re.split(r"(\W+)", text)

    for word in words:
        # convert to precomposed form
        word = unicodedata.normalize("NFC", word)
        tone = None

        for ch in word:
            ch, new_tone = decompose_tone(ch)
            if new_tone is not None:
                assert tone is None, word
                tone = new_tone
            output.append(ch)

        if tone is not None:
            output.append(tone)

    return output


print(tokenize_text("Xin chào!"))
# tokenize_text("123 Xin chào! TP.HCM thật đẹp. Trung-ương")
