#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor pack_launch(at::Tensor inp, int b);
at::Tensor unpack_launch(at::Tensor inp, int b, int L);