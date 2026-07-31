#include "flashattention.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string>

namespace {

struct FlashAttentionDims {
  uint32_t batch;
  uint32_t heads;
  uint32_t seq;
  uint32_t dim;
  uint32_t causal;
  float scale;
};

std::string kernel_name(int head_dim) {
  return std::format("flash_attention_1_d{}", head_dim);
}

constexpr std::size_t pipeline_slot(int head_dim) noexcept {
  return head_dim == launch::kHeadDims[0] ? 0 : 1;
}

}  // namespace

std::size_t FlashAttention::threadgroup_memory_bytes(std::size_t head_dim) {
  return launch::threadgroup_memory_bytes(static_cast<int>(head_dim));
}

FlashAttention::FlashAttention(const MetalDevice& device,
                               std::string_view metallib_path)
    : device_(device),
      library_(device.load_library(std::string(metallib_path))) {
  if (!library_) {
    throw std::runtime_error(
        std::format("failed to load metallib: {}", metallib_path));
  }
}

MTL::ComputePipelineState* FlashAttention::pipeline(int head_dim) {
  auto& cached = pipelines_[pipeline_slot(head_dim)];
  if (cached) {
    return cached.get();
  }

  const std::string kernel = kernel_name(head_dim);
  auto* name = NS::String::string(kernel.c_str(), NS::UTF8StringEncoding);
  const auto function = NS::TransferPtr(library_->newFunction(name));
  if (!function) {
    throw std::runtime_error(std::format("kernel {} not found", kernel));
  }

  NS::Error* error = nullptr;
  cached = NS::TransferPtr(
      device_.device()->newComputePipelineState(function.get(), &error));
  if (!cached) {
    throw std::runtime_error(std::format(
        "failed to build pipeline for {}: {}", kernel,
        error != nullptr ? error->localizedDescription()->utf8String()
                         : "no reason reported"));
  }
  return cached.get();
}

void FlashAttention::run(const Matrix& q, const Matrix& k, const Matrix& v,
                         const AttentionShape& shape, Matrix& out) {
  if (shape.head_dim != 64 && shape.head_dim != 128) {
    throw std::runtime_error(std::format(
        "head dimension must be 64 or 128, got {}", shape.head_dim));
  }

  const int head_dim = static_cast<int>(shape.head_dim);
  const auto tile = static_cast<std::size_t>(launch::num_query_rows(head_dim));
  if (shape.seq_len % tile != 0) {
    throw std::runtime_error(std::format(
        "sequence length {} must be a multiple of {}", shape.seq_len, tile));
  }

  const std::size_t bytes = shape.rows() * shape.head_dim * sizeof(float);

  auto* device = device_.device();
  if (buffers_.bytes < bytes) {
    const auto shared_buffer = [device, bytes] {
      return NS::TransferPtr(
          device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    };
    buffers_ = {.q = shared_buffer(),
                .k = shared_buffer(),
                .v = shared_buffer(),
                .o = shared_buffer(),
                .bytes = bytes};
  }
  std::memcpy(buffers_.q->contents(), q.data.data(), bytes);
  std::memcpy(buffers_.k->contents(), k.data.data(), bytes);
  std::memcpy(buffers_.v->contents(), v.data.data(), bytes);

  const FlashAttentionDims dims{
      .batch = static_cast<uint32_t>(shape.batch_size),
      .heads = static_cast<uint32_t>(shape.num_heads),
      .seq = static_cast<uint32_t>(shape.seq_len),
      .dim = static_cast<uint32_t>(shape.head_dim),
      .causal = shape.use_causal_mask ? 1u : 0u,
      .scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim))};

  auto* command = device_.queue()->commandBuffer();
  auto* encoder = command->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline(head_dim));
  encoder->setBuffer(buffers_.q.get(), 0, 0);
  encoder->setBuffer(buffers_.k.get(), 0, 1);
  encoder->setBuffer(buffers_.v.get(), 0, 2);
  encoder->setBuffer(buffers_.o.get(), 0, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setThreadgroupMemoryLength(threadgroup_memory_bytes(shape.head_dim),
                                      0);

  encoder->dispatchThreadgroups(
      MTL::Size::Make(shape.seq_len / tile, shape.num_heads, shape.batch_size),
      MTL::Size::Make(launch::num_threads(), 1, 1));
  encoder->endEncoding();
  command->commit();
  command->waitUntilCompleted();

  std::memcpy(out.data.data(), buffers_.o->contents(), bytes);
}
