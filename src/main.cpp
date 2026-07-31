#include <chrono>
#include <cstdio>
#include <functional>

#include "attention_naive.hpp"
#include "flashattention.hpp"
#include "matrix.hpp"
#include "metal_device.hpp"

namespace {

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
  std::printf("%6s %6s %6s %6s %6s %12s\n", "batch", "heads", "seq", "dim",
              "causal", "v1");

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

    Matrix out(rows, shape.head_dim);
    flash.run(q, k, v, shape, out);

    std::printf("%6zu %6zu %6zu %6zu %6s %12.3e\n", shape.batch_size,
                shape.num_heads, shape.seq_len, shape.head_dim,
                shape.use_causal_mask ? "yes" : "no",
                relative_frobenius(out, expected));
  }
  std::printf("\n");
}

void bench_metal(const char* metallib_path) {
  MetalDevice device;
  FlashAttention flash(device, metallib_path);

  std::printf("metal, v1 ==\n");
  for (const std::size_t dim : {std::size_t{64}, std::size_t{128}}) {
    std::printf("# dim=%zu: %zu bytes of threadgroup memory\n", dim,
                FlashAttention::threadgroup_memory_bytes(dim));
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
            time_call([&] { flash.run(q, k, v, shape, out); }, kWarmup, kIters);
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

  return 0;
}
