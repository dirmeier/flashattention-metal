#pragma once

#include <cstddef>

#include "matrix.hpp"

struct AttentionShape {
  std::size_t batch_size;
  std::size_t num_heads;
  std::size_t seq_len;
  std::size_t head_dim;
  bool use_causal_mask = true;

  std::size_t rows() const { return batch_size * num_heads * seq_len; }
};

void attention_naive(const Matrix& q, const Matrix& k, const Matrix& v,
                     const AttentionShape& shape, Matrix& out);
