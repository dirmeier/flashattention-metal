#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <string>

class MetalDevice {
 public:
  MetalDevice();
  MetalDevice(const MetalDevice&) = delete;
  MetalDevice& operator=(const MetalDevice&) = delete;

  static bool available();

  MTL::Device* device() const { return device_.get(); }
  MTL::CommandQueue* queue() const { return queue_.get(); }
  NS::SharedPtr<MTL::Library> load_library(const std::string& path) const;

 private:
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
};
