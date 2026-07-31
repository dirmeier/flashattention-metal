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
