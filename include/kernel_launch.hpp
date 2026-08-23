#pragma once

#include <algorithm>

/// launch geometry for the three FlashAttention kernels.
namespace launch {

inline constexpr int kPad = 4;
inline constexpr int kVersions[] = {1, 2, 3};
inline constexpr int kHeadDims[] = {64, 128};

/// key rows per iteration of the inner loop
constexpr int num_key_rows(int head_dim) { return head_dim <= 64 ? 32 : 16; }

/// query rows one threadgroup computes
constexpr int num_query_rows(int version, int head_dim) {
  return version == 1 ? (head_dim <= 64 ? 32 : 16) : 32;
}

/// number of threads per threadgroup
constexpr int num_threads(int version) { return version == 1 ? 32 : 128; }

/// bytes of threadgroup memory the kernel needs, for
/// `setThreadgroupMemoryLength`
constexpr int threadgroup_memory_bytes(int version, int head_dim) {
  const int tile = num_query_rows(version, head_dim);
  const int keys = num_key_rows(head_dim);
  int floats = 0;
  switch (version) {
    case 1:
      // scores: tile * tile (QK^T)
      // each of queries/keys/outputs:  tile * head_dim;
      floats = tile * tile + 3 * tile * head_dim;
      break;
    case 2:  // queries (32, fixed) and keys (accumulation in registers)
      floats = (32 + keys) * head_dim;
      break;
    default:  // queries, plus KV threadgroup memory sized for the larger of K,
              // V
      floats = 32 * (head_dim + kPad) +  // Q
               std::max(                 // take whatever is larger of K and V
                   head_dim * (keys + kPad),  // K
                   keys * (head_dim + kPad)   // V
               );
  }
  return floats * static_cast<int>(sizeof(float));
}

}  // namespace launch
