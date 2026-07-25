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
