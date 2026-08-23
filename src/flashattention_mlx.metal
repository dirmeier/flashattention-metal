/// MLX custom kernel body

threadgroup float shared[kShared];

FlashAttentionDims dims;
dims.batch = kBatch;
dims.heads = kHeads;
dims.seq = kSeq;
dims.dim = kDim;
dims.causal = kCausal ? 1u : 0u;
dims.scale = 1.0f / sqrt(float(kDim));

const uint3 block = threadgroup_position_in_grid;

if constexpr (kVersion == 1) {
  flash_attention_1_impl<kTile>(q, k, v, out, dims, shared, block,
                                     thread_position_in_threadgroup.x, 32u);
} else if constexpr (kVersion == 2) {
  flash_attention_2_impl<kDim, kKTile>(
      q, k, v, out, dims, shared, block, thread_position_in_threadgroup.x,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
} else {
  flash_attention_3_impl<kDim, kKTile>(
      q, k, v, out, dims, shared, block, thread_position_in_threadgroup.x,
      simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
}
