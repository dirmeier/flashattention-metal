#include "metal_device.hpp"

bool MetalDevice::available() {
  return static_cast<bool>(NS::TransferPtr(MTL::CreateSystemDefaultDevice()));
}

MetalDevice::MetalDevice()
    : device_(NS::TransferPtr(MTL::CreateSystemDefaultDevice())) {
  if (device_) {
    queue_ = NS::TransferPtr(device_->newCommandQueue());
  }
}

NS::SharedPtr<MTL::Library> MetalDevice::load_library(
    const std::string& path) const {
  if (!device_) {
    return {};
  }
  auto* ns_path = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nullptr;
  return NS::TransferPtr(device_->newLibrary(ns_path, &error));
}
