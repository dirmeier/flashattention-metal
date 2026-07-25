#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string_view>

#include "attention_naive.hpp"
#include "matrix.hpp"
#include "metal_device.hpp"

class FlashAttention {
 public:
  FlashAttention(const MetalDevice& device, std::string_view metallib_path);
  FlashAttention(const FlashAttention&) = delete;
  FlashAttention& operator=(const FlashAttention&) = delete;

  void run(const Matrix& q, const Matrix& k, const Matrix& v,
           const AttentionShape& shape, Matrix& out);

  static std::size_t threadgroup_memory_bytes();

 private:
  MTL::ComputePipelineState* pipeline();

  struct Buffers {
    NS::SharedPtr<MTL::Buffer> q;
    NS::SharedPtr<MTL::Buffer> k;
    NS::SharedPtr<MTL::Buffer> v;
    NS::SharedPtr<MTL::Buffer> o;
    std::size_t bytes = 0;
  };

  const MetalDevice& device_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  Buffers buffers_;
};
