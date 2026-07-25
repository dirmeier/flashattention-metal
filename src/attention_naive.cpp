#include "attention_naive.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

void attention_naive(const Matrix& q, const Matrix& k, const Matrix& v,
                     const AttentionShape& shape, Matrix& out) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
  std::vector<float> scores(shape.seq_len);

  const std::size_t slices = shape.batch_size * shape.num_heads;
  for (std::size_t slice = 0; slice < slices; ++slice) {
    const std::size_t base = slice * shape.seq_len;

    for (std::size_t i = 0; i < shape.seq_len; ++i) {
      const std::size_t visible =
          shape.use_causal_mask ? (i + 1) : shape.seq_len;

      float max_score = -std::numeric_limits<float>::infinity();
      for (std::size_t j = 0; j < visible; ++j) {
        float dot = 0.0f;
        for (std::size_t d = 0; d < shape.head_dim; ++d) {
          dot += q[base + i, d] * k[base + j, d];
        }
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }

      float sum = 0.0f;
      for (std::size_t j = 0; j < visible; ++j) {
        scores[j] = std::exp(scores[j] - max_score);
        sum += scores[j];
      }

      for (std::size_t d = 0; d < shape.head_dim; ++d) {
        float acc = 0.0f;
        for (std::size_t j = 0; j < visible; ++j) {
          acc += scores[j] * v[base + j, d];
        }
        out[base + i, d] = acc / sum;
      }
    }
  }
}
