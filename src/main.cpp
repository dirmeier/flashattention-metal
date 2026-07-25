#include <cstdio>

#include "attention_naive.hpp"
#include "flashattention.hpp"
#include "matrix.hpp"
#include "metal_device.hpp"

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
  std::printf("# %zu bytes of threadgroup memory\n",
              FlashAttention::threadgroup_memory_bytes());
  std::printf("%6s %6s %6s %6s %6s %12s\n", "batch", "heads", "seq", "dim",
              "causal", "error");

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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <flashattention.metallib>\n", argv[0]);
    return 2;
  }
  check_against_reference(argv[1]);

  return 0;
}
