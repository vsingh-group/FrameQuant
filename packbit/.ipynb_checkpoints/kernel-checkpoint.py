
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math
import random

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = [os.path.join(curr_path, 'extension', file) for file in ['cuda_kernel.cu', 'torch_extension.cpp']]
packbit = load('packbit', src_files, verbose = True)

import packbit

def profile(fn):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0

def unit_test():
    L = random.randrange(10000, 10000000)
    b = random.randrange(1, 9)
    print(f"L = {L}, b = {b}")
    
    weight = torch.randint(int(2 ** b), size = (L, ), device = "cuda", dtype = torch.int)
    packed = packbit.pack_fn(weight, b)
    unpacked = packbit.unpack_fn(packed, b, L)
    
    print(weight.shape, packed.shape, unpacked.shape)
    print((weight - unpacked).abs().max())

    print(profile(lambda:packbit.unpack_fn(packbit.pack_fn(weight, b), b, L)))
    print(profile(lambda:weight.clone().clone()))
    
unit_test()