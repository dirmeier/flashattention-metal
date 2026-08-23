#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

struct FlashAttentionDims {
    uint batch;
    uint heads;
    uint seq;  ///< must be a multiple of the kernel's query tile
    uint dim;
    uint causal;
    float scale;
};

// Q, K, V and O are rank 4 tensors of fimensionality [B, H, S, D],
// One (b, h) pair owns a contiguous [S, D] slice.
//
// Inside a [S, D] slice, we split Q into multiple `blocks` along S
// such that each threadgroup receives one [T, D] tile of Q where T is a
// constant and T << S.
// The grid is hence dispatched as (`blocks`, H, B).
//
// Every kernel below names its slice:
// tgid.z is the batch, tgid.y the head, tgid.x the block.

constant constexpr uint kQBlock = 32;
constant constexpr uint kKBlock = 32;
constant constexpr uint kMaxDim = 128;
constant constexpr uint kFrag = 8;
/// number of threads per threadgroup (in this case a threadgroup has exactly one simdgroup of threads (32))
constant constexpr uint kThreads = 32;

constant constexpr uint kQTileWide = 32;
constant constexpr uint kSimdgroupsWide = kQTileWide / kFrag;
constant constexpr uint kThreadsWide = kSimdgroupsWide * 32;

constant constexpr uint kPad = 4;
// One threadgroup handles one block of [T, D] query rows.
//
// Since each query row needs the entire [S, D]-dimensional keys and values,
// they are iterated over, i.e., read form device memory (DRAM), instead of
// stored to SRAM. This implementation also reads the query to DRAM!
//
// The running output O, and the softmax max and sum stay in
// registers throughout. The score block is written to threadgroup memory (SMEM).
kernel void flash_attention(device const float* q [[buffer(0)]],
                       device const float* k [[buffer(1)]],
                       device const float* v [[buffer(2)]],
                       device float* out [[buffer(3)]],
                       constant FlashAttentionDims& dims [[buffer(4)]],
                       threadgroup float* shared [[threadgroup(0)]],
                       uint3 tgid [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    const uint q0 = tgid.x * kQBlock;
    if (q0 >= dims.seq) {
        return;
    }

    const uint stride = dims.seq * dims.dim;
    const uint slice = (tgid.z * dims.heads + tgid.y) * stride;
    device const float* qb = q + slice;
    device const float* kb = k + slice;
    device const float* vb = v + slice;
    device float* ob = out + slice;

    // one thread per query row
    const uint row = q0 + tid;
    const bool active = (tid < kQBlock) && (row < dims.seq);

     // kQBlock * kKBlock
    threadgroup float* scores = shared;

    float acc[kMaxDim];
    for (uint d = 0; d < dims.dim; ++d) {
        acc[d] = 0.0f;
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // key blocks entirely above the diagonal are never visited
    const uint key_limit = (dims.causal != 0u) ? min(q0 + kQBlock, dims.seq) : dims.seq;

    for (uint k0 = 0; k0 < key_limit; k0 += kKBlock) {
        const uint kn = min(kKBlock, dims.seq - k0);

        float block_max = -INFINITY;
        if (active) {
            for (uint j = 0; j < kn; ++j) {
                const uint key = k0 + j;
                float dot = 0.0f;
                if (dims.causal == 0u || key <= row) {
                    for (uint d = 0; d < dims.dim; ++d) {
                        dot += qb[row * dims.dim + d] * kb[key * dims.dim + d];
                    }
                    dot *= dims.scale;
                } else {
                    dot = -INFINITY;
                }
                scores[tid * kKBlock + j] = dot;
                block_max = max(block_max, dot);
            }
        }
        // this is __syncthreads in CUDA
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            const float new_max = max(running_max, block_max);
            const float alpha = isinf(running_max) ? 0.0f : exp(running_max - new_max);
            running_sum *= alpha;
            for (uint d = 0; d < dims.dim; ++d) {
                acc[d] *= alpha;
            }

            for (uint j = 0; j < kn; ++j) {
                const float s = scores[tid * kKBlock + j];
                const float p = isinf(s) ? 0.0f : exp(s - new_max);
                running_sum += p;
                for (uint d = 0; d < dims.dim; ++d) {
                    acc[d] += p * vb[(k0 + j) * dims.dim + d];
                }
            }
            running_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        for (uint d = 0; d < dims.dim; ++d) {
            ob[row * dims.dim + d] = acc[d] / running_sum;
        }
    }
}

// One threadgroup handles one block of [T, D] query rows.
// Compared to to `flash_attention`, both matrix products (QK^T and SV)
// now run on the simdgroup matrix engine.
//
// Q, K and the running output O are loaded to threadgroup memory (SMEM).
// V is still read from DRAM, because:
//  SMEM is 32 KB on Apple silicon: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
//  V1 uses the following memory:
//      scores:  T × T (scores)
//      O_smem:  T × D (output)
//      Q_smem:  T × D
//      K_smem:  T × D
//  Which means the SMEM would overflow if we would also moved V into it.
//
// The scores are written to SMEM from registers and read back vice versa.
// This is necessary because the softmax needs a max and a sum per row. However, the
// score block lives in `simdgroup_float8x8` fragments which are 8x8 pieces, each
// held across the simdgroup's 32 threads with two values each. Hence, a row's 8
// values sit in 4 different threads, so no thread can compute a row max on its
// own meaning we need to write to SMEM.
template <uint kTile>
void flash_attention_1_impl(device const float* q,
                            device const float* k,
                            device const float* v,
                            device float* out,
                            FlashAttentionDims dims,
                            threadgroup float* shared,
                            uint3 tgid,
                            uint tid,
                            uint threads) {
    const uint q0 = tgid.x * kTile;
    if (q0 >= dims.seq) {
        return;
    }

    const uint stride = dims.seq * dims.dim;
    const uint slice = (tgid.z * dims.heads + tgid.y) * stride;
    device const float* qb = q + slice;
    device const float* kb = k + slice;
    device const float* vb = v + slice;
    device float* ob = out + slice;

    const uint tile_elems = kTile * dims.dim;
    threadgroup float* scores = shared;                    // kTile * kTile
    threadgroup float* O_smem = scores + kTile * kTile;   // kTile * dim
    threadgroup float* Q_smem = O_smem + tile_elems;     // kTile * dim
    threadgroup float* K_smem = Q_smem + tile_elems;     // kTile * dim

    // A simdgroup is 32 threads wide regardless of the tile, so at kTile = 16
    // the upper threads help with the copy and sit out the row-wise softmax.
    const bool owns_row = tid < kTile;
    const uint row = q0 + tid;

    // Row-then-column rather than a flat index. A flat index needs a division
    // and a modulo by a runtime `dim` for every element, and this form also has
    // adjacent threads reading adjacent addresses.
    for (uint r = 0; r < kTile; ++r) {
        for (uint c = tid; c < dims.dim; c += threads) {
            Q_smem[r * dims.dim + c] = qb[(q0 + r) * dims.dim + c];
        }
    }
    if (owns_row) {
        for (uint d = 0; d < dims.dim; ++d) {
            O_smem[tid * dims.dim + d] = 0.0f;
        }
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint key_limit = (dims.causal != 0u) ? min(q0 + kTile, dims.seq) : dims.seq;

    for (uint k0 = 0; k0 < key_limit; k0 += kTile) {
        for (uint r = 0; r < kTile; ++r) {
            for (uint c = tid; c < dims.dim; c += threads) {
                K_smem[r * dims.dim + c] = kb[(k0 + r) * dims.dim + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q * K^T, now entirely out of threadgroup memory.
        for (uint i = 0; i < kTile / kFrag; ++i) {
            for (uint j = 0; j < kTile / kFrag; ++j) {
                simdgroup_float8x8 s_frag = simdgroup_float8x8(0.0f);
                for (uint d = 0; d < dims.dim; d += kFrag) {
                    simdgroup_float8x8 fq;
                    simdgroup_float8x8 fk;
                    simdgroup_load(fq, Q_smem + i * kFrag * dims.dim + d, dims.dim);
                    simdgroup_load(fk, K_smem + j * kFrag * dims.dim + d, dims.dim, ulong2(0),
                                   true);
                    simdgroup_multiply_accumulate(s_frag, fq, fk, s_frag);
                }
                simdgroup_store(s_frag, scores + i * kFrag * kTile + j * kFrag, kTile);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (owns_row) {
            float srow[kTile];
            float block_max = -INFINITY;
            for (uint j = 0; j < kTile; ++j) {
                const uint key = k0 + j;
                float s = scores[tid * kTile + j] * dims.scale;
                if (key >= dims.seq || (dims.causal != 0u && key > row)) {
                    s = -INFINITY;
                }
                srow[j] = s;
                block_max = max(block_max, s);
            }

            const float new_max = max(running_max, block_max);
            const float alpha = isinf(running_max) ? 0.0f : exp(running_max - new_max);
            running_sum *= alpha;
            for (uint d = 0; d < dims.dim; ++d) {
                O_smem[tid * dims.dim + d] *= alpha;
            }
            for (uint j = 0; j < kTile; ++j) {
                const float p = isinf(srow[j]) ? 0.0f : exp(srow[j] - new_max);
                running_sum += p;
                scores[tid * kTile + j] = p;
            }
            running_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + P * V. Only V still comes from device memory.
        for (uint i = 0; i < kTile / kFrag; ++i) {
            for (uint dd = 0; dd < dims.dim; dd += kFrag) {
                simdgroup_float8x8 o_frag;
                simdgroup_load(o_frag, O_smem + i * kFrag * dims.dim + dd, dims.dim);
                for (uint j = 0; j < kTile / kFrag; ++j) {
                    simdgroup_float8x8 fp;
                    simdgroup_float8x8 fv;
                    simdgroup_load(fp, scores + i * kFrag * kTile + j * kFrag, kTile);
                    simdgroup_load(fv, vb + (k0 + j * kFrag) * dims.dim + dd, dims.dim);
                    simdgroup_multiply_accumulate(o_frag, fp, fv, o_frag);
                }
                simdgroup_store(o_frag, O_smem + i * kFrag * dims.dim + dd, dims.dim);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (owns_row && row < dims.seq) {
        for (uint d = 0; d < dims.dim; ++d) {
            ob[row * dims.dim + d] = O_smem[tid * dims.dim + d] / running_sum;
        }
    }
}

// Builds on v1, but leaves the score tile in a simdgroup fragment.
//
// Uses `thread_elements` to access thread registers. As a result, can leave
// scores and outputs in the register:
//  SMEM holds only Q and K
//      Q_smem:  32 × D
//      K_smem:  K_rows × D
// which is 16 KB at D = 64 and 24 KB at D = 128. V is still read from DRAM.
//
// Similarly to `thread_elements`, the row max and sum can be obtained via
// another MSL function: `simd_shuffle_xor`.
//
// The head dimension is a template parameter so the fragment arrays can be
// indexed at compile time and stay in registers rather than spilling.
template <uint kDim, uint kKTile>
void flash_attention_2_impl(device const float* q,
                            device const float* k,
                            device const float* v,
                            device float* out,
                            FlashAttentionDims dims,
                            threadgroup float* shared,
                            uint3 tgid,
                            uint tid,
                            uint sgid,
                            uint lane) {
    const uint q0 = tgid.x * kQTileWide;
    if (q0 >= dims.seq) {
        return;
    }

    const uint stride = dims.seq * dims.dim;
    const uint slice = (tgid.z * dims.heads + tgid.y) * stride;
    device const float* qb = q + slice;
    device const float* kb = k + slice;
    device const float* vb = v + slice;
    device float* ob = out + slice;

    threadgroup float* Q_smem = shared;                  // kQTileWide * kDim
    threadgroup float* K_smem = Q_smem + kQTileWide * kDim;  // kKTile * kDim

    // each simdgroup owns one 8-row slice of the query block.
    const uint row_base = sgid * kFrag;
    const uint frag_row = 4u * (lane / 16u) + (lane % 8u) / 2u;
    const uint frag_col = 2u * (lane % 2u) + 4u * ((lane % 16u) / 8u);
    const uint my_row = q0 + row_base + frag_row;

    for (uint r = 0; r < kQTileWide; ++r) {
        for (uint c = tid; c < kDim; c += kThreadsWide) {
            Q_smem[r * kDim + c] = qb[(q0 + r) * kDim + c];
        }
    }

    simdgroup_float8x8 o_frag[kDim / kFrag];
    for (uint c = 0; c < kDim / kFrag; ++c) {
        o_frag[c] = simdgroup_float8x8(0.0f);
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint key_limit = (dims.causal != 0u) ? min(q0 + kQTileWide, dims.seq) : dims.seq;

    for (uint k0 = 0; k0 < key_limit; k0 += kKTile) {
        for (uint r = 0; r < kKTile; ++r) {
            for (uint c = tid; c < kDim; c += kThreadsWide) {
                K_smem[r * kDim + c] = kb[(k0 + r) * kDim + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s_frag[kKTile / kFrag];
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
            for (uint d = 0; d < kDim; d += kFrag) {
                simdgroup_float8x8 fq;
                simdgroup_float8x8 fk;
                simdgroup_load(fq, Q_smem + row_base * kDim + d, kDim);
                simdgroup_load(fk, K_smem + j * kFrag * kDim + d, kDim, ulong2(0), true);
                simdgroup_multiply_accumulate(acc, fq, fk, acc);
            }
            s_frag[j] = acc;
        }

        // scale and mask in place, and take this lane's maximum as we go.
        float block_max = -INFINITY;
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            thread auto& e = s_frag[j].thread_elements();
            for (uint i = 0; i < 2; ++i) {
                const uint key = k0 + j * kFrag + frag_col + i;
                float s = e[i] * dims.scale;
                if (key >= dims.seq || (dims.causal != 0u && key > my_row)) {
                    s = -INFINITY;
                }
                e[i] = s;
                block_max = max(block_max, s);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1u));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8u));

        const float new_max = max(running_max, block_max);
        const float alpha = isinf(running_max) ? 0.0f : exp(running_max - new_max);

        float block_sum = 0.0f;
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            thread auto& e = s_frag[j].thread_elements();
            for (uint i = 0; i < 2; ++i) {
                const float p = isinf(e[i]) ? 0.0f : exp(e[i] - new_max);
                e[i] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1u);
        block_sum += simd_shuffle_xor(block_sum, 8u);

        running_sum = running_sum * alpha + block_sum;
        running_max = new_max;

        // O = alpha * O + P * V
        for (uint c = 0; c < kDim / kFrag; ++c) {
            thread auto& oe = o_frag[c].thread_elements();
            oe[0] *= alpha;
            oe[1] *= alpha;
        }
        for (uint c = 0; c < kDim / kFrag; ++c) {
            for (uint j = 0; j < kKTile / kFrag; ++j) {
                simdgroup_float8x8 fv;
                simdgroup_load(fv, vb + (k0 + j * kFrag) * kDim + c * kFrag, kDim);
                simdgroup_multiply_accumulate(o_frag[c], s_frag[j], fv, o_frag[c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = 1.0f / running_sum;
    for (uint c = 0; c < kDim / kFrag; ++c) {
        thread auto& oe = o_frag[c].thread_elements();
        oe[0] *= inv;
        oe[1] *= inv;
        simdgroup_store(o_frag[c], ob + (q0 + row_base) * kDim + c * kFrag, kDim);
    }
}

kernel void flash_attention_2_d64(device const float* q [[buffer(0)]],
                                  device const float* k [[buffer(1)]],
                                  device const float* v [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  constant FlashAttentionDims& dims [[buffer(4)]],
                                  threadgroup float* shared [[threadgroup(0)]],
                                  uint3 tgid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    flash_attention_2_impl<64, 32>(q, k, v, out, dims, shared, tgid, tid, sgid, lane);
}

kernel void flash_attention_2_d128(device const float* q [[buffer(0)]],
                                   device const float* k [[buffer(1)]],
                                   device const float* v [[buffer(2)]],
                                   device float* out [[buffer(3)]],
                                   constant FlashAttentionDims& dims [[buffer(4)]],
                                   threadgroup float* shared [[threadgroup(0)]],
                                   uint3 tgid [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sgid [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    flash_attention_2_impl<128, 16>(q, k, v, out, dims, shared, tgid, tid, sgid, lane);
}


// Builds on v2, but adds features from Apple's implementation of FlashAttention from MLX
// https://github.com/ml-explore/mlx/blob/117188cd735299f396a08f7697a81759b1e0550b/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L70
//
// tl;dr: I just adapted what MLX is doing :] In short:
//  SMEM now also holds V. at no extra cost, because K and V share one buffer.
//  K is written, used for the score product, and then V is written over the top
//  of it. So SMEM is Q plus whichever of K and V is larger:
//      Q_smem:   32 × (D + 4)
//      KV_smem:  max(D × (K_rows + 4), K_rows × (D + 4))
//  K is written transposed to SMEM which explains the dimensionalty: K_rows × (D + 4)
template <uint kDim, uint kKTile>
void flash_attention_3_impl(device const float* q,
                            device const float* k,
                            device const float* v,
                            device float* out,
                            FlashAttentionDims dims,
                            threadgroup float* shared,
                            uint3 tgid,
                            uint tid,
                            uint sgid,
                            uint lane) {
    const uint q0 = tgid.x * kQTileWide;
    if (q0 >= dims.seq) {
        return;
    }

    const uint stride = dims.seq * dims.dim;
    const uint slice = (tgid.z * dims.heads + tgid.y) * stride;
    device const float* qb = q + slice;
    device const float* kb = k + slice;
    device const float* vb = v + slice;
    device float* ob = out + slice;

    constexpr uint kLDQ = kDim + kPad;    // query rows, padded
    constexpr uint kLDK = kKTile + kPad;  // K^T rows are depth, cols are keys
    constexpr uint kLDV = kDim + kPad;
    threadgroup float* Q_smem = shared;                   // kQTileWide * kLDQ
    threadgroup float* KV_smem = Q_smem + kQTileWide * kLDQ;  // K, then V over it

    const uint qid = lane / 4u;
    const uint sm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint sn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint row_base = sgid * kFrag;
    const uint my_row = q0 + row_base + sm;

    for (uint r = 0; r < kQTileWide; ++r) {
        for (uint c = tid; c < kDim; c += kThreadsWide) {
            Q_smem[r * kLDQ + c] = qb[(q0 + r) * kDim + c];
        }
    }

    simdgroup_float8x8 o_frag[kDim / kFrag];
    for (uint c = 0; c < kDim / kFrag; ++c) {
        o_frag[c] = simdgroup_float8x8(0.0f);
    }

    // Scores are carried in log2 space, as MLX does, so the softmax is exp2.
    const float scale2 = dims.scale * M_LOG2E_F;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    const uint key_limit = (dims.causal != 0u) ? min(q0 + kQTileWide, dims.seq) : dims.seq;

    for (uint k0 = 0; k0 < key_limit; k0 += kKTile) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // K transposed on the way in: KV_smem[depth][key].
        for (uint r = 0; r < kKTile; ++r) {
            for (uint c = tid; c < kDim; c += kThreadsWide) {
                KV_smem[c * kLDK + r] = kb[(k0 + r) * kDim + c];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s_frag[kKTile / kFrag];
#pragma clang loop unroll(full)
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            s_frag[j] = simdgroup_float8x8(0.0f);
        }
#pragma clang loop unroll(full)
        for (uint d = 0; d < kDim / kFrag; ++d) {
            simdgroup_float8x8 fq;
            simdgroup_load(fq, Q_smem + row_base * kLDQ + d * kFrag, kLDQ);
#pragma clang loop unroll(full)
            for (uint j = 0; j < kKTile / kFrag; ++j) {
                simdgroup_float8x8 fk;
                simdgroup_load(fk, KV_smem + d * kFrag * kLDK + j * kFrag, kLDK);
                    simdgroup_multiply_accumulate(s_frag[j], fq, fk, s_frag[j]);
            }
        }

        const bool edge = (k0 + kKTile > dims.seq) ||
                          (dims.causal != 0u && k0 + kKTile > q0 + row_base + 1u);
#pragma clang loop unroll(full)
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            thread auto& e = s_frag[j].thread_elements();
            e[0] *= scale2;
            e[1] *= scale2;
            if (edge) {
                for (uint i = 0; i < 2; ++i) {
                    const uint key = k0 + j * kFrag + sn + i;
                    if (key >= dims.seq || (dims.causal != 0u && key > my_row)) {
                        e[i] = -INFINITY;
                    }
                }
            }
        }

        // V goes into the same buffer K used
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < kKTile; ++r) {
            for (uint c = tid; c < kDim; c += kThreadsWide) {
                KV_smem[r * kLDV + c] = vb[(k0 + r) * kDim + c];
            }
        }

        float block_max = -INFINITY;
#pragma clang loop unroll(full)
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            thread auto& e = s_frag[j].thread_elements();
            block_max = max(block_max, max(e[0], e[1]));
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1u));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8u));

        const float new_max = max(running_max, block_max);
        const float factor = isinf(running_max) ? 0.0f : exp2(running_max - new_max);

        float block_sum = 0.0f;
#pragma clang loop unroll(full)
        for (uint j = 0; j < kKTile / kFrag; ++j) {
            thread auto& e = s_frag[j].thread_elements();
            e[0] = isinf(e[0]) ? 0.0f : exp2(e[0] - new_max);
            e[1] = isinf(e[1]) ? 0.0f : exp2(e[1] - new_max);
            block_sum += e[0] + e[1];
        }
        block_sum += simd_shuffle_xor(block_sum, 1u);
        block_sum += simd_shuffle_xor(block_sum, 8u);

        running_sum = running_sum * factor + block_sum;
        running_max = new_max;

#pragma clang loop unroll(full)
        for (uint c = 0; c < kDim / kFrag; ++c) {
            thread auto& oe = o_frag[c].thread_elements();
            oe[0] *= factor;
            oe[1] *= factor;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint c = 0; c < kDim / kFrag; ++c) {
#pragma clang loop unroll(full)
            for (uint j = 0; j < kKTile / kFrag; ++j) {
                    simdgroup_float8x8 fv;
                simdgroup_load(fv, KV_smem + j * kFrag * kLDV + c * kFrag, kLDV);
                    simdgroup_multiply_accumulate(o_frag[c], s_frag[j], fv, o_frag[c]);
            }
        }
    }

    const float inv = 1.0f / running_sum;
#pragma clang loop unroll(full)
    for (uint c = 0; c < kDim / kFrag; ++c) {
        thread auto& oe = o_frag[c].thread_elements();
        oe[0] *= inv;
        oe[1] *= inv;
        simdgroup_store(o_frag[c], ob + (q0 + row_base) * kDim + c * kFrag, kDim);
    }
}

kernel void flash_attention_3_d64(device const float* q [[buffer(0)]],
                                  device const float* k [[buffer(1)]],
                                  device const float* v [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  constant FlashAttentionDims& dims [[buffer(4)]],
                                  threadgroup float* shared [[threadgroup(0)]],
                                  uint3 tgid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sgid [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    flash_attention_3_impl<64, 32>(q, k, v, out, dims, shared, tgid, tid, sgid, lane);
}

kernel void flash_attention_3_d128(device const float* q [[buffer(0)]],
                                   device const float* k [[buffer(1)]],
                                   device const float* v [[buffer(2)]],
                                   device float* out [[buffer(3)]],
                                   constant FlashAttentionDims& dims [[buffer(4)]],
                                   threadgroup float* shared [[threadgroup(0)]],
                                   uint3 tgid [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sgid [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    flash_attention_3_impl<128, 16>(q, k, v, out, dims, shared, tgid, tid, sgid, lane);
}
kernel void flash_attention_1_d64(device const float* q [[buffer(0)]],
                                 device const float* k [[buffer(1)]],
                                 device const float* v [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant FlashAttentionDims& dims [[buffer(4)]],
                                 threadgroup float* shared [[threadgroup(0)]],
                                 uint3 tgid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    flash_attention_1_impl<32>(q, k, v, out, dims, shared, tgid, tid, kThreads);
}

kernel void flash_attention_1_d128(device const float* q [[buffer(0)]],
                                 device const float* k [[buffer(1)]],
                                 device const float* v [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant FlashAttentionDims& dims [[buffer(4)]],
                                 threadgroup float* shared [[threadgroup(0)]],
                                 uint3 tgid [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    flash_attention_1_impl<16>(q, k, v, out, dims, shared, tgid, tid, kThreads);
}
