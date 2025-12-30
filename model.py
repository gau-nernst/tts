# inspired by Z-Image arch
# TODO: weight init, mixed-precision

import dataclasses
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.varlen import varlen_attn

import ops


class AttnAuxData(NamedTuple):
    pos_ids: Tensor | None  # for RoPE
    cu: Tensor  # for varlen_attn
    max_length: int

    @staticmethod
    def from_size_list(size_list: list[int], device: torch.types.Device = None):
        pos_ids_list = []
        cu_list = [0]
        max_length = 0

        for size in size_list:
            pos_ids_list.append(torch.arange(size, device=device))
            cu_list.append(cu_list[-1] + size)
            max_length = max(max_length, size)

        return AttnAuxData(
            pos_ids=torch.cat(pos_ids_list, dim=0),
            cu=torch.tensor(cu_list, dtype=torch.int32, device=device),
            max_length=max_length,
        )


def timestep_embedding(t: Tensor, dim: int, max_period: float = 1e4, time_factor: float = 1e3) -> Tensor:
    t = time_factor * t.float()
    half = dim // 2
    omega = torch.exp(-math.log(max_period) / half * torch.arange(half, dtype=torch.float32, device=t.device))
    freqs = omega * t.unsqueeze(-1)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def compute_rope(pos_ids: Tensor, dim: int, theta: float) -> Tensor:
    # initial computations in fp64
    omega = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=pos_ids.device) / dim))
    freqs = (pos_ids[:, None] * omega).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def compute_nd_rope(pos_ids: Tensor, dims: tuple[int, ...], theta: float) -> Tensor:
    # TODO: investigate the effect of theta. Z-Image uses theta=256
    rope_list = [compute_rope(pos_ids[..., i], dim, theta=theta) for i, dim in enumerate(dims)]
    return torch.cat(rope_list, dim=-1)


def apply_rope(x: Tensor, rope: Tensor):
    dtype = rope.dtype.to_real()
    x_ = torch.view_as_complex(x.to(dtype).unflatten(-1, (-1, 2)))  # [..., L, nH, D/2]
    out = torch.view_as_real(x_ * rope.unsqueeze(-2)).flatten(-2)  # [..., L, nH, D]
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, dim: int, qk_norm: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(128, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(128, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: Tensor, rope: Tensor, aux_data: AttnAuxData) -> Tensor:
        q, k, v = self.qkv(x).unflatten(-1, (-1, 128)).chunk(3, dim=-2)  # each is (..., L, nH, D)
        q = apply_rope(self.q_norm(q), rope)
        k = apply_rope(self.k_norm(k), rope)

        cu = aux_data.cu
        max_length = aux_data.max_length
        o = varlen_attn(q, k, v, cu, cu, max_length, max_length)
        return self.o(o.flatten(-2))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def modulate(x: Tensor, scale: Tensor, eps: float = 1e-6):
    out = F.rms_norm(x.float(), x.shape[-1:], eps=eps)
    out = out * scale.unsqueeze(-2)
    return out.to(x.dtype)


class Block(nn.Module):
    def __init__(self, dim: int, mod_dim: int, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.attn = Attention(dim, eps=eps)
        self.mlp = MLP(dim, hidden_dim)
        self.eps = eps

        if mod_dim > 0:  # with modulation
            self.modulation = nn.Linear(mod_dim, dim * 4)
        else:
            self.modulation = None
            self.attn_norm = nn.RMSNorm(dim, eps=eps)
            self.mlp_norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x: Tensor, rope: Tensor, aux_data: AttnAuxData, mod_input: Tensor | None = None) -> Tensor:
        if self.modulation is not None:
            attn_scale, attn_gate, mlp_scale, mlp_gate = self.modulation(mod_input).float().chunk(4, dim=-1)

            res = self.attn(modulate(x, 1.0 + attn_scale, eps=self.eps), rope, aux_data)
            x = x + modulate(res, attn_gate.tanh(), eps=self.eps)

            res = self.mlp(modulate(x, 1.0 + mlp_scale, eps=self.eps))
            x = x + modulate(res, mlp_gate.tanh(), eps=self.eps)

        else:
            x = x + self.attn(self.attn_norm(x), rope, aux_data)
            x = x + self.mlp(self.mlp_norm(x))

        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, mod_dim: int, out_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(mod_dim, dim))
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x: Tensor, mod_input: Tensor) -> Tensor:
        scale = self.modulation(mod_input)
        x = modulate(x, 1.0 + scale.float(), eps=1e-6)
        return self.linear(x)


@dataclasses.dataclass
class ModelConfig:
    dim: int
    in_dim: int
    t_dim: int
    vocab_size: int
    text_layers: int
    audio_layers: int
    joint_layers: int
    mlp_dim: int
    rope_dims: tuple[int, ...]


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_embed = nn.Sequential(nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, cfg.t_dim))

        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.text_layers = nn.ModuleList([Block(cfg.dim, 0, cfg.mlp_dim) for _ in range(cfg.text_layers)])

        self.audio_embed = nn.Linear(cfg.in_dim, cfg.dim)
        self.audio_layers = nn.ModuleList([Block(cfg.dim, cfg.t_dim, cfg.mlp_dim) for _ in range(cfg.audio_layers)])

        self.joint_layers = nn.ModuleList([Block(cfg.dim, cfg.t_dim, cfg.mlp_dim) for _ in range(cfg.joint_layers)])
        self.output = FinalLayer(cfg.dim, cfg.t_dim, cfg.in_dim)

    def forward(
        self,
        audio: Tensor,
        audio_aux: AttnAuxData,
        time: Tensor,
        text_tokens: Tensor,
        text_aux: AttnAuxData,
    ) -> Tensor:
        time = timestep_embedding(time.squeeze(), 256).to(self.time_embed[0].weight.dtype)
        time = self.time_embed(time)

        # text-only processing
        text = self.text_embed(text_tokens)
        text_pos_ids = F.pad(text_aux.pos_ids.unsqueeze(-1), (0, 1))  # set 2nd RoPE dim to 0
        text_rope = compute_nd_rope(text_pos_ids, self.cfg.rope_dims, theta=1e4)
        for layer in self.text_layers:
            text = layer(text, text_rope, text_aux)

        # audio-only processing
        audio = self.audio_embed(audio)
        audio_pos_ids = F.pad(audio_aux.pos_ids.unsqueeze(-1), (1, 0))  # set 1st RoPE dim to 0
        audio_rope = compute_nd_rope(audio_pos_ids, self.cfg.rope_dims, theta=1e4)
        for layer in self.audio_layers:
            audio = layer(audio, audio_rope, audio_aux, time)

        # merge audio and text varlen sequences
        joint, joint_rope, joint_cu = ops.merge_varlen(
            audio,
            text,
            torch.view_as_real(audio_rope).flatten(-2),
            torch.view_as_real(text_rope).flatten(-2),
            audio_aux.cu,
            text_aux.cu,
        )
        joint_rope = torch.view_as_complex(joint_rope.unflatten(-1, (-1, 2)))
        joint_aux = AttnAuxData(None, joint_cu, audio_aux.max_length + text_aux.max_length)

        # joint processing
        for layer in self.joint_layers:
            joint = layer(joint, joint_rope, joint_aux, time)

        audio = ops.slice_varlen(joint, joint_cu, audio_aux.cu)
        return self.output(audio, time)
