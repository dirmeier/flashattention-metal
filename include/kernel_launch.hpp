#pragma once

/// launch geometry for the FlashAttention kernels.
namespace launch {

inline constexpr int kHeadDims[] = {64, 128};

/// key rows per iteration of the inner loop
constexpr int num_key_rows(int head_dim) { return head_dim <= 64 ? 32 : 16; }

/// query rows one threadgroup computes
constexpr int num_query_rows(int head_dim) { return head_dim <= 64 ? 32 : 16; }

/// number of threads per threadgroup
constexpr int num_threads() { return 32; }

/// bytes of threadgroup memory the kernel needs, for
/// `setThreadgroupMemoryLength`
constexpr int threadgroup_memory_bytes(int head_dim) {
  const int tile = num_query_rows(head_dim);
  // scores: tile * tile (QK^T)
  // each of queries/keys/outputs:  tile * head_dim;
  const int floats = tile * tile + 3 * tile * head_dim;
  return floats * static_cast<int>(sizeof(float));
}

}  // namespace launch
