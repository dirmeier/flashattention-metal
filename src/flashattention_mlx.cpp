#include "flashattention_mlx.hpp"

#include <mlx/mlx.h>

#include <format>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "embedded_shaders.hpp"
#include "kernel_launch.hpp"

namespace mx = mlx::core;

namespace {

using Version = MLXAttention::Version;
using TemplateArgs = std::vector<std::pair<std::string, mx::fast::TemplateArg>>;

/// our FlashAttention, compiled by MLX from the embedded shader source
mx::array custom_flash_attention(const mx::array& q, const mx::array& k,
                                 const mx::array& v, bool causal,
                                 Version version) {
  const int batch = q.shape(0);
  const int heads = q.shape(1);
  const int seq = q.shape(2);
  const int dim = q.shape(3);

  if (dim != 64 && dim != 128) {
    throw std::runtime_error(
        std::format("head dimension must be 64 or 128, got {}", dim));
  }

  const int tile = launch::num_query_rows(dim);
  const int threads = launch::num_threads();

  if (seq % tile != 0) {
    throw std::runtime_error(
        std::format("sequence length {} must be a multiple of {}", seq, tile));
  }

  const std::string name = std::format("flash_attention_mlx_v{}_d{}",
                                       std::to_underlying(version), dim);

  const auto kernel = mx::fast::metal_kernel(name, {"q", "k", "v"}, {"out"},
                                             shaders::kFlashAttentionMLXBody,
                                             shaders::kFlashAttentionHeader);

  const TemplateArgs templates{
      {"kVersion", std::to_underlying(version)},
      {"kDim", dim},
      {"kSeq", seq},
      {"kBatch", batch},
      {"kHeads", heads},
      {"kCausal", causal},
      {"kTile", tile},
      {"kShared",
       launch::threadgroup_memory_bytes(dim) / static_cast<int>(sizeof(float))},
      {"kKTile", launch::num_key_rows(dim)}};

  const auto grid = std::tuple{threads * (seq / tile), heads, batch};
  const auto threadgroup = std::tuple{threads, 1, 1};

  return kernel({q, k, v}, {q.shape()}, {q.dtype()}, grid, threadgroup,
                templates, std::nullopt, /*verbose=*/false, {})[0];
}

}  // namespace

bool MLXAttention::available() { return mx::metal::is_available(); }

mx::array MLXAttention::operator()(const mx::array& q, const mx::array& k,
                                   const mx::array& v, bool causal,
                                   Version version) {
  if (q.ndim() != 4) {
    throw std::runtime_error("expected rank 4 tensor");
  }
  return custom_flash_attention(q, k, v, causal, version);
}
