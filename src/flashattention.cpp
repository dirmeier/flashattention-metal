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

/// query rows one threadgroup computes, and the key rows it folds in per
/// iteration of the inner loop
constexpr int kQueryRows = 32;
constexpr int kKeyRows = 32;
/// one simdgroup, one thread per query row
constexpr int kThreads = 32;

constexpr const char* kKernelName = "flash_attention";

struct FlashAttentionDims {
  uint32_t batch;
  uint32_t heads;
  uint32_t seq;
  uint32_t dim;
  uint32_t causal;
  float scale;
};

}  // namespace

std::size_t FlashAttention::threadgroup_memory_bytes() {
  return static_cast<std::size_t>(kQueryRows) * kKeyRows * sizeof(float);
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

MTL::ComputePipelineState* FlashAttention::pipeline() {
  if (pipeline_) {
    return pipeline_.get();
  }

  auto* name = NS::String::string(kKernelName, NS::UTF8StringEncoding);
  const auto function = NS::TransferPtr(library_->newFunction(name));
  if (!function) {
    throw std::runtime_error(std::format("kernel {} not found", kKernelName));
  }

  NS::Error* error = nullptr;
  pipeline_ = NS::TransferPtr(
      device_.device()->newComputePipelineState(function.get(), &error));
  if (!pipeline_) {
    throw std::runtime_error(std::format(
        "failed to build pipeline for {}: {}", kKernelName,
        error != nullptr ? error->localizedDescription()->utf8String()
                         : "no reason reported"));
  }
  return pipeline_.get();
}

void FlashAttention::run(const Matrix& q, const Matrix& k, const Matrix& v,
                         const AttentionShape& shape, Matrix& out) {
  if (shape.head_dim != 64 && shape.head_dim != 128) {
    throw std::runtime_error(std::format(
        "head dimension must be 64 or 128, got {}", shape.head_dim));
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
  encoder->setComputePipelineState(pipeline());
  encoder->setBuffer(buffers_.q.get(), 0, 0);
  encoder->setBuffer(buffers_.k.get(), 0, 1);
  encoder->setBuffer(buffers_.v.get(), 0, 2);
  encoder->setBuffer(buffers_.o.get(), 0, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setThreadgroupMemoryLength(threadgroup_memory_bytes(), 0);

  const std::size_t blocks = (shape.seq_len + kQueryRows - 1) / kQueryRows;
  encoder->dispatchThreadgroups(
      MTL::Size::Make(blocks, shape.num_heads, shape.batch_size),
      MTL::Size::Make(kThreads, 1, 1));
  encoder->endEncoding();
  command->commit();
  command->waitUntilCompleted();

  std::memcpy(out.data.data(), buffers_.o->contents(), bytes);
}
