# metal

[![ci](https://github.com/dirmeier/metal/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/metal/actions/workflows/ci.yaml)

> FlashAttention for Metal

`metal` implements three FlashAttention variants in Metal. All three implementations compute the same output, but differ in where the data lives: in registers, in threadgroup memory (SMEM, 'shared memory' on CUDA devices), or in device memory (DRAM, HBM on CUDA devices).

The repo is a proof-of-concept to learn Metal and contrast it with CUDA.
Comparing the 3 implementation with MLX's native implementation gives the following stats.

| kernel | ms | GFLOP/s |
|---|---|---|
| v1 (Q, K and O in SMEM; scores written to SMEM and read back; V from DRAM) | 266 | 65 |
| v2 (scores and O stay in registers; SMEM holds only Q and K) | 40 | 432 |
| v3 (K written transposed into SMEM; shared KV buffer) | 11 | 1611 |
| MLX's own attention | 10 | 1729 |

## Requirements

- `meson` >= 1.0 and `ninja`,
- `Xcode` with `metal` and `metallib`,
- `metal-cpp` is fetched automatically as a Meson subproject,
- `mlx` for the baseline and the custom-kernel path. For instance, build using [`macports`](https://www.macports.org/) via
  `sudo port install mlx`.

Verify the Metal toolchain with:

```sh
xcrun --find metal && xcrun --find metallib
```

## Usage

To build the static library and run the executable, call:

```sh
meson setup build
meson compile -C build
./build/src/flash-attention build/src/flashattention.metallib
```

## API

To build the static lib against your own source file (`main.cpp`),
call:

```sh
clang++ -std=c++23 -Iinclude -I/opt/local/include \
  -Isubprojects/metal-cpp-release-metal-cpp_macOS26.4_iOS26.4 \
  main.cpp build/src/libfa.a -L/opt/local/lib -lmlx \
  -framework Metal -framework Foundation -framework QuartzCore -o main
```

The metal-cpp headers come from a Meson subproject/

Q, K and V are tensors of shape `[batch, heads, sequence, head_dim]`
with the head dimension being either 64 or 128.

```cpp
#include <mlx/mlx.h>

#include "flashattention_mlx.hpp"

namespace mx = mlx::core;

const mx::Shape dims{2, 8, 512, 128};
mx::array q = mx::random::normal(dims);
mx::array k = mx::random::normal(dims);
mx::array v = mx::random::normal(dims);

MLXAttention attention;
mx::array o = attention(q, k, v, /*causal=*/true,
                        MLXAttention::Version::v3);
mx::eval(o);
```

## LLVM IR (AIR)

To produce AIR, Apple's LLVM IR, just call:

```sh
xcrun metal-objdump --disassemble build/src/flashattention.metallib \
  > flashattention_air_simdgroup.air.ll
grep -i simdgroup_matrix flashattention_air_simdgroup.air.ll
```

## Development

The project uses `uv` for `pre-commit` and `gitlint`.

```bash
python3 -m pre_commit install -t pre-commit -t commit-msg
python3 -m pre_commit run --all-files
```
