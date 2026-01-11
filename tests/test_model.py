import pytest
import torch

from model import AttnAuxData, Model, ModelConfig


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_model_smoke(weight_dtype: torch.dtype):
    cfg = ModelConfig(256, 512, 256, 128, 2, 2, 10, 2048, (64, 64))
    with torch.device("cuda"):
        model = Model(cfg).to(weight_dtype)

    audio_lens = torch.randint(20, 400, size=(10,)).tolist()
    text_lens = torch.randint(5, 100, size=(10,)).tolist()

    audio_aux = AttnAuxData.from_size_list(audio_lens, device="cuda")
    text_aux = AttnAuxData.from_size_list(text_lens, device="cuda")

    audio = torch.randn(sum(audio_lens), cfg.in_dim, device="cuda")
    time = torch.tensor([0.5], device="cuda")
    text_tokens = torch.randint(cfg.vocab_size, size=(sum(text_lens),), device="cuda")

    out = model(audio, audio_aux, time, text_tokens, text_aux)
    assert out.shape == audio.shape

    out.sum().backward()
    for p in model.parameters():
        assert p.grad is not None
