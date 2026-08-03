"""Wraps and runs the v1 FlashAttention Metal kernel."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

SHADERS_PATH_ = Path(__file__).resolve().parent.parent / "src"

KERNEL_ = mx.fast.metal_kernel(
    name="flash_attention_mlx_v1",
    input_names=["q", "k", "v"],
    output_names=["out"],
    source=(SHADERS_PATH_ / "flashattention_mlx.metal").read_text(),
    header=(SHADERS_PATH_ / "flashattention.metal").read_text(),
)


def _mlx_attention(
    q: mx.array, k: mx.array, v: mx.array, causal: bool = True
) -> mx.array:
    B, H, T, D = q.shape
    tile = 32 if D <= 64 else 16
    if T % tile != 0:
        raise ValueError(f"sequence {T} must be a multiple of {tile}")

    keys = 32 if D <= 64 else 16
    shared = tile * tile + 3 * tile * D
    return KERNEL_(
        inputs=[q, k, v],
        template=[
            ("kVersion", 1),
            ("kDim", D),
            ("kSeq", T),
            ("kBatch", B),
            ("kHeads", H),
            ("kCausal", causal),
            ("kTile", tile),
            ("kShared", shared),
            ("kKTile", keys),
        ],
        grid=(32 * (T // tile), H, B),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


class TransformerBlock(nn.Module):
    def __init__(self, dims: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dims // heads
        self.qkv = nn.Linear(dims, 3 * dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dims, 4 * dims), nn.GELU(), nn.Linear(4 * dims, dims)
        )
        self.pre_attention_rmsnorm = nn.RMSNorm(dims)
        self.pre_mlp_rmsnorm = nn.RMSNorm(dims)

    def _attention(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        q, k, v = (
            t.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
            for t in mx.split(self.qkv(x), 3, axis=-1)
        )
        o = _mlx_attention(q, k, v)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, T, D))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self._attention(self.pre_attention_rmsnorm(x))
        x = x + self.mlp(self.pre_mlp_rmsnorm(x))
        return x


def main():
    B, T, D = 2, 128, 256
    x = mx.random.normal((B, T, D))
    out = TransformerBlock(dims=D, heads=D // 64)(x)
    mx.eval(out)
    print(out)


if __name__ == "__main__":
    main()
