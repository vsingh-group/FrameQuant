#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_kernel.h"

at::Tensor pack_fn(at::Tensor inp, int b) {
    return pack_launch(inp, b);
}
at::Tensor unpack_fn(at::Tensor inp, int b, int L) {
    return unpack_launch(inp, b, L);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_fn", &pack_fn, "pack_fn");
  m.def("unpack_fn", &unpack_fn, "unpack_fn");
}
