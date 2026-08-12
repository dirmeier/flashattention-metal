#pragma once

#include <mlx/mlx.h>

class MLXAttention {
 public:
  enum class Version { v1 = 1, v2 };
  static bool available();

  mlx::core::array operator()(const mlx::core::array& q,
                              const mlx::core::array& k,
                              const mlx::core::array& v, bool causal,
                              Version version = Version::v1);
};
