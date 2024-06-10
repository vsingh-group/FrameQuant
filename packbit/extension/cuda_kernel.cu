#include "cuda_kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__global__ void pack(
  int *inp,    // [L]
  int *out,    // [N]
  int L,
  int N,
  int b
) {
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= N) {
    return;
  }

  int num_vals = 32 / b;
  int vals = 0;
  for (int idx = 0; idx < num_vals; idx++) {
    int addr = idx * N + thread_idx;
    if (addr < L) {
      int val = inp[addr];
      vals = (vals << b) | val;
    }
  }
  out[thread_idx] = vals;
}

at::Tensor pack_launch(at::Tensor inp, int b) {
  int L = inp.size(0);
  int num_vals = 32 / b;
  int N = L / num_vals;
  if (L % num_vals != 0) {
    N = N + 1;
  }
  at::Tensor out = at::empty({N}, inp.options());
  dim3 threads(1024);
  dim3 blocks(N / 1024 + 1);
  pack<<<blocks, threads>>>(inp.data_ptr<int>(), out.data_ptr<int>(), L, N, b);
  return out;
}

__global__ void unpack(
  int *inp,    // [N]
  int *out,    // [L]
  int L,
  int N,
  int b
) {
  const int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (thread_idx >= N) {
    return;
  }

  int num_vals = 32 / b;
  int mask = (1 << b) - 1;
  int vals = inp[thread_idx];
  for (int idx = num_vals - 1; idx >= 0; idx--) {
    int addr = idx * N + thread_idx;
    if (addr < L) {
      out[addr] = vals & mask;
      vals = vals >> b;
    }
  }
}

at::Tensor unpack_launch(at::Tensor inp, int b, int L) {
  int N = inp.size(0);
  at::Tensor out = at::empty({L}, inp.options());
  dim3 threads(1024);
  dim3 blocks(N / 1024 + 1);
  unpack<<<blocks, threads>>>(inp.data_ptr<int>(), out.data_ptr<int>(), L, N, b);
  return out;
}