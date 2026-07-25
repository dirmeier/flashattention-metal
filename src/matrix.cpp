#include "matrix.hpp"

#include <cmath>
#include <random>

Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
  Matrix m(rows, cols);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : m.data) {
    v = dist(rng);
  }
  return m;
}

double relative_frobenius(const Matrix& x, const Matrix& ref) {
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t r = 0; r < ref.rows; ++r) {
    for (std::size_t c = 0; c < ref.cols; ++c) {
      const double diff =
          static_cast<double>(x[r, c]) - static_cast<double>(ref[r, c]);
      const double base = static_cast<double>(ref[r, c]);
      numerator += diff * diff;
      denominator += base * base;
    }
  }
  if (denominator == 0.0) {
    return std::sqrt(numerator);
  }
  return std::sqrt(numerator) / std::sqrt(denominator);
}
