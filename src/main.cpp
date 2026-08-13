#include <mlx/mlx.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <utility>

#include "attention_naive.hpp"
#include "flashattention.hpp"
#include "flashattention_mlx.hpp"
#include "matrix.hpp"
#include "metal_device.hpp"

namespace mx = mlx::core;

namespace {

using Version = FlashAttention::Version;

constexpr Version kVersions[] = {Version::v1, Version::v2};

/// GFLOP/s and GB/s
std::tuple<double, double> throughput(const AttentionShape& shape,
                                      double seconds) {
  const double elements = static_cast<double>(shape.rows() * shape.head_dim);
  const double flops = 4.0 * elements * static_cast<double>(shape.seq_len) *
                       (shape.use_causal_mask ? 0.5 : 1.0);
  const double bytes = 4.0 * elements * sizeof(float);
  return {flops / seconds / 1e9, bytes / seconds / 1e9};
}

double time_call(const std::function<void()>& call, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    call();
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    call();
  }
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count() / iters;
}

void print_header() {
  std::printf("%6s %6s %6s %6s %6s %10s %10s %10s\n", "batch", "heads", "seq",
              "dim", "causal", "ms", "GFLOP/s", "GB/s");
}

void print_row(const AttentionShape& shape, double seconds) {
  const auto [gflops, bandwidth_gb] = throughput(shape, seconds);
  std::printf("%6zu %6zu %6zu %6zu %6s %10.3f %10.1f %10.1f\n",
              shape.batch_size, shape.num_heads, shape.seq_len, shape.head_dim,
              shape.use_causal_mask ? "yes" : "no", seconds * 1e3, gflops,
              bandwidth_gb);
}

constexpr int kWarmup = 3;
constexpr int kIters = 10;

/// wraps host rows as [batch, heads, sequence, dim] without copying
mx::array wrap(const Matrix& m, const AttentionShape& shape) {
  const mx::Shape dims{
      static_cast<int>(shape.batch_size), static_cast<int>(shape.num_heads),
      static_cast<int>(shape.seq_len), static_cast<int>(shape.head_dim)};
  return mx::array(const_cast<float*>(m.data.data()), dims, mx::float32,
                   [](void*) {});
}

void run_mlx(MLXAttention& attention, const Matrix& q, const Matrix& k,
             const Matrix& v, const AttentionShape& shape,
             MLXAttention::Version version, Matrix& out) {
  mx::array result = attention(wrap(q, shape), wrap(k, shape), wrap(v, shape),
                               shape.use_causal_mask, version);
  mx::eval(result);
  std::memcpy(out.data.data(), result.data<float>(),
              out.data.size() * sizeof(float));
}

}  // namespace

void check_against_reference(const char* metallib_path) {
  if (!MetalDevice::available()) {
    return;
  }

  MetalDevice device;
  FlashAttention flash(device, metallib_path);

  constexpr AttentionShape kShapes[] = {
      {1, 1, 32, 64, false},   {1, 1, 128, 64, false}, {2, 2, 160, 64, false},
      {2, 2, 512, 128, false}, {2, 4, 1024, 64, true}, {1, 2, 256, 128, true},
      {2, 1, 224, 128, true}};

  std::printf("bench scalar reference \n");
  std::printf("%6s %6s %6s %6s %6s %12s %12s\n", "batch", "heads", "seq", "dim",
              "causal", "v1", "v2");

  for (const AttentionShape& shape : kShapes) {
    const std::size_t rows = shape.rows();
    Matrix q = random_matrix(rows, shape.head_dim, 101);
    Matrix k = random_matrix(rows, shape.head_dim, 202);
    for (float& value : q.data) {
      value *= 40.0f;
    }
    for (float& value : k.data) {
      value *= 40.0f;
    }
    const Matrix v = random_matrix(rows, shape.head_dim, 303);

    Matrix expected(rows, shape.head_dim);
    attention_naive(q, k, v, shape, expected);

    std::printf("%6zu %6zu %6zu %6zu %6s", shape.batch_size, shape.num_heads,
                shape.seq_len, shape.head_dim,
                shape.use_causal_mask ? "yes" : "no");
    for (const Version version : kVersions) {
      Matrix out(rows, shape.head_dim);
      flash.run(q, k, v, shape, version, out);
      std::printf(" %12.3e", relative_frobenius(out, expected));
    }
    std::printf("\n");
  }
  std::printf("\n");
}

void bench_metal(const char* metallib_path) {
  MetalDevice device;
  FlashAttention flash(device, metallib_path);

  for (const Version version : kVersions) {
    std::printf("metal, v%d ==\n", std::to_underlying(version));
    for (const std::size_t dim : {std::size_t{64}, std::size_t{128}}) {
      std::printf("# dim=%zu: %zu bytes of threadgroup memory\n", dim,
                  FlashAttention::threadgroup_memory_bytes(version, dim));
    }
    print_header();

    for (const std::size_t seq :
         {std::size_t{512}, std::size_t{1024}, std::size_t{2048}}) {
      for (const std::size_t dim : {std::size_t{64}, std::size_t{128}}) {
        for (const bool causal : {false, true}) {
          const AttentionShape shape{2, 4, seq, dim, causal};
          const std::size_t rows = shape.rows();

          const Matrix q = random_matrix(rows, dim, 1);
          const Matrix k = random_matrix(rows, dim, 2);
          const Matrix v = random_matrix(rows, dim, 3);
          Matrix out(rows, dim);

          const double seconds =
              time_call([&] { flash.run(q, k, v, shape, version, out); },
                        kWarmup, kIters);
          print_row(shape, seconds);
        }
      }
    }
  }
}

void bench_mlx() {
  MLXAttention attention;

  std::printf("mlx attention\n");
  print_header();

  for (const std::size_t seq :
       {std::size_t{512}, std::size_t{1024}, std::size_t{2048}}) {
    for (const std::size_t dim : {std::size_t{64}, std::size_t{128}}) {
      for (const bool causal : {false, true}) {
        const AttentionShape shape{2, 4, seq, dim, causal};
        const std::size_t rows = shape.rows();

        const Matrix q = random_matrix(rows, dim, 1);
        const Matrix k = random_matrix(rows, dim, 2);
        const Matrix v = random_matrix(rows, dim, 3);
        Matrix out(rows, dim);

        const double seconds = time_call(
            [&] {
              run_mlx(attention, q, k, v, shape, MLXAttention::Version::v0,
                      out);
            },
            kWarmup, kIters);
        print_row(shape, seconds);
      }
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <flashattention.metallib>\n", argv[0]);
    return 2;
  }
  check_against_reference(argv[1]);
  bench_metal(argv[1]);
  bench_mlx();

  return 0;
}
