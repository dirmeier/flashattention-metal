#pragma once

#include <cstddef>
#include <vector>

struct Matrix {
  std::size_t rows;
  std::size_t cols;
  std::size_t ld;
  std::vector<float> data;

  Matrix() = default;
  Matrix(std::size_t r, std::size_t c)
      : rows(r), cols(c), ld(c), data(r * c, 0.0f) {}

  float& operator[](std::size_t r, std::size_t c) { return data[r * ld + c]; }
  const float& operator[](std::size_t r, std::size_t c) const {
    return data[r * ld + c];
  }
};

Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed);

double relative_frobenius(const Matrix& x, const Matrix& ref);
