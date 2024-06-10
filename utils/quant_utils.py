from utils.construct_tff import construct_real_tff
from fast_hadamard_transform import hadamard_transform
import logging
import torch
import math

def get_tff_blks(name, d, tffs, l_den, tff_redundancy, dev='cpu'):
    if d not in tffs:
        l_tff = d // l_den 
        k_tff = round(d // l_tff * tff_redundancy)
        logging.info(f'{l_den = }')
        print(f'{l_den = }')
        logging.info(f'computing TFF, {name}, red = {tff_redundancy}, {k_tff = }, {l_tff = }, n_tff = {d}')
        print(f'computing TFF, {name}, red = {tff_redundancy}, {k_tff = }, {l_tff = }, n_tff = {d}')
        tff, b_is = construct_real_tff(k_tff, l_tff // 2, d // 2)

        tff = tff.permute(1,0,2).contiguous()

        tff_blks = []
        for _l in range(l_tff //2):
            tff_blk = tff[2*_l:2*_l+2,:,b_is[_l,0]:b_is[_l,1]].contiguous()
            tff_blks.append(tff_blk.view(-1, tff_blk.shape[-1]))

        tff_blks = torch.stack(tff_blks, dim=0)
        tffs[d] = tff_blks.to(dev)

def compute_tffs(subset, gptq, name, tffs, seed, lin_count, l_den, tff_redundancy):
    l_n = subset[name].weight.shape[0]
    p_n = subset[name].weight.shape[1]
    for _d in (l_n, p_n):
        get_tff_blks(name, _d, tffs, l_den, tff_redundancy)

    gptq[name].P_l_T = tffs[l_n]
    gptq[name].P_prev_T = tffs[p_n]
    gptq[name].quantizer.l_seed = seed + lin_count
    gptq[name].quantizer.p_seed = seed + lin_count

def norm_hadamard_transform(x):
    return hadamard_transform(x, scale = 1/math.sqrt(x.shape[-1]))

def tff_project(w, seed, Proj, dev):
    # check power of 2
    n,d = w.shape
    if not d & (d-1):
        for i in range(1):
            g = torch.Generator(device=dev) # use this to store the seed for later
            g.manual_seed(seed+i)
            di = torch.randint(0,2,(d,), device=dev,generator=g)
            di = 2*di-1
            w = norm_hadamard_transform(w * di)
    else:
        logging.info(f'{d=} is not 2^n')
        d2n = d & ~(d-1) # get the largest power of 2 that divides d
        w = w.view(n, -1, d2n).permute(1,0,2)
        for i in range(1):
            g = torch.Generator(device=dev) # use this to store the seed for later
            g.manual_seed(seed+i)
            di = torch.randint(0,2,(d2n,), device=dev,generator=g)
            di = 2*di-1
            w = norm_hadamard_transform(w * di)
        w = w.permute(1,0,2).reshape(n, -1)

    if Proj is not None:
        w = torch.einsum("ujd,ukj->ukd", w.T.contiguous().view(-1,Proj.shape[-1],w.shape[0]), Proj.to(dev))
        w = w.permute(1,0,2) # to get the shape (k,u,d)
        d = w.shape[-1]
        w = w.contiguous().view(-1,d).transpose(0,1)

    return w

def inv_tff(w, seed, Proj, dev):

    if Proj is not None:
        w = w.T.contiguous().view(-1, Proj.shape[0], w.shape[0]).permute(1,0,2)
        d = w.shape[-1]
        w = torch.einsum("ukd,ukj->ujd", w, Proj)
        w = w.contiguous().view(-1, d).transpose(0,1)

    # check power of 2
    n, d = w.shape
    if not d & (d-1):
        for i in reversed(range(1)):
            g = torch.Generator(device=dev) # use this to store the seed for later
            g.manual_seed(seed+i)
            di = torch.randint(0,2,(d,), device=dev,generator=g)
            di = 2*di-1
            w = norm_hadamard_transform(w) * di
    else:
        logging.info(f'{d=} is not 2^n')
        d2n = d & ~(d-1) # get the largest power of 2 that divides d
        w = w.view(n, -1, d2n).permute(1,0,2)
        for i in reversed(range(1)):
            g = torch.Generator(device=dev) # use this to store the seed for later
            g.manual_seed(seed+i)
            di = torch.randint(0,2,(d2n,), device=dev,generator=g)
            di = 2*di-1
            w = norm_hadamard_transform(w) * di
        w = w.permute(1,0,2).reshape(n, -1)

    return w
