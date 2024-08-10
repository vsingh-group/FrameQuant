import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *
from utils.quant_utils import tff_project, inv_tff 

import logging

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float64)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, tff_transform=False
    ):

        # compute TFF projections of the data if FrameQuant is enabled
        if tff_transform:
            W = self.layer.weight.data.clone().to(torch.float32)
            w_clone = W.clone()
            dev = W.device
            H = self.H.data.clone().to(torch.float32)
            H = H * (H.shape[0] / (torch.trace(H) + 1e-8)) + 1e-2 * torch.eye(H.shape[0], device=dev)
            #####################################
            # taking TFF projections
            W = tff_project(W, self.quantizer.l_seed*10000+1234, self.P_prev_T, dev)
            W = tff_project(W.T, self.quantizer.l_seed*10000+4321, self.P_l_T, dev)
            W = W.T

            H = tff_project(H, self.quantizer.l_seed*10000+1234, self.P_prev_T, dev)
            H = tff_project(H.T, self.quantizer.l_seed*10000+1234, self.P_prev_T, dev)
            H = H.T
            #####################################
            self.rows = W.shape[0]
            self.columns = W.shape[1]

        # find the parameters for quantizing
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # 2sigma clip the weights
        xvar = W.var(dim=1, keepdim=True)
        xstd = torch.sqrt(xvar)
        clip_scale = 2*xstd
        clip_zero = W.mean(dim=1, keepdim=True)

        W = torch.clamp(W, min=clip_zero-clip_scale, max=clip_zero+clip_scale)

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H += H + damp * torch.eye(self.columns, device=H.device)
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), 
                    self.quantizer.scale, 
                    self.quantizer.zero, 
                    self.quantizer.maxq, 
                    use_float_bias = self.quantizer.use_float_bias
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.to(self.layer.weight.data.dtype)

        # undo the TFF transform after quantization
        if tff_transform:
            w = self.layer.weight.data.clone().to(torch.float32)
            P_prev_T = self.P_prev_T.to(w.device)
            P_l_T = self.P_l_T.to(w.device)
            #################################################3
            # inverting TFF projections
            w = inv_tff(w, self.quantizer.l_seed*10000+1234, P_prev_T, dev)
            w = inv_tff(w.T, self.quantizer.l_seed*10000+4321, P_l_T, dev)
            w = w.T
            #################################################3
            error = ((w-w_clone)**2).mean().sqrt()
            print(f'rms {error = }')
            logging.info(f'rms {error = }')
            self.layer.weight.data = w.to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))


    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
