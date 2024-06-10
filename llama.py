import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *
import os
import logging
from utils.construct_tff import construct_real_tff
from utils.quant_utils import compute_tffs
from utils.eval_utils import llama_eval
from datetime import datetime
from transformers import LlamaForCausalLM
import argparse
from datautils import *

from tqdm import tqdm
import math

def get_llama(model, seqlen):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = seqlen
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev, seed=0):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    tffs = {}
    lin_count = 0
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                lin_count += 1 
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, 
                    perchannel=True, 
                    sym=args.sym, 
                    mse=False,
                    x_sigma=args.x_sigma,
                    use_float_bias = args.use_float_bias
                )
                gptq[name].name = name

                if args.tff_transform:
                    compute_tffs(subset, gptq, name, tffs, seed=seed, lin_count=lin_count,
                                 l_den=args.l_den, tff_redundancy=args.tff_redundancy)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                if gptq[name].nsamples == 0:
                    print(f'{name} has nsamples = 0')
                    breakpoint()
                    continue

                print(i, name)
                print('Quantizing ...')
                logging.info(f'{i}, {name}')
                logging.info('Quantizing ...')

                gptq[name].fasterquant(
                    percdamp=args.percdamp, 
                    groupsize=args.groupsize, 
                    actorder=args.act_order, 
                    static_groups=args.static_groups,
                    tff_transform=args.tff_transform
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers, tffs


def llama_pack3(model, quantizers, l_den, tff_redundancy, wbits, tffs = {}):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, l_den, tff_redundancy, wbits)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        qlayers[name].pack(layers[name], quantizers[name], tffs)
    print('Done.')
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save quantized checkpoint'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--exp_name', type=str, default='Llama_FQ',
        help='name of the experiment'
    )
    parser.add_argument(
        '--tff_redundancy', type=float, default=1.0,
        help='Redundancy in TFF representations'
    )
    parser.add_argument(
        '--tff_transform', action='store_true',
        help='Whether to enable TFF transform'
    )
    parser.add_argument(
        '--x_sigma', type=float, default=2.0,
        help='multiply factor for sigma to clip the weights. If None, all the range of the weights is used'
    )
    parser.add_argument(
        '--use_float_bias', action='store_true',
        help='use floating point bias for quantizing'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='Whether to evaluate the quantized model'
    )

    parser.add_argument(
        '--l_den', type=int, default=16,
        help='denominator to be used for L_tff'
    )

    parser.add_argument(
        '--seqlen', type=int, default=2048,
        help='sequence length for the model'
    )

    parser.add_argument(
        '--results_dir', type=str, default='./results/',
        help='Where to extract calibration data from.'
    )

    args = parser.parse_args()

    # seed the experiment
    set_seed(args.seed)

    # setup logs
    results_dir = args.results_dir
    exp_dir = os.path.join(results_dir, f'{args.exp_name}_'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    os.makedirs(exp_dir, exist_ok=True)
    filename = os.path.join(exp_dir, f'{args.exp_name}.log')
    logging.basicConfig(filename=filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model = get_llama(args.model, args.seqlen)
    model.eval()

    print(model)

    if args.wbits < 16 and not args.nearest:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
        )

        tick = time.time()
        quantizers, tffs = llama_sequential(model, dataloader, DEV, args.seed)
        print(time.time() - tick)

    if args.eval:
        ppls = []
        datasets = ['wikitext2', 'ptb', 'c4'] 
        if args.new_eval:
            datasets = ['wikitext2', 'ptb-new', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            ppl, _ = llama_eval(model, testloader, DEV)
            logging.info(f'{dataset = }, quantized model {ppl = }')
            ppls.append(ppl.item())

        import csv
        import os

        results = [args.exp_name]
        results.extend([args.wbits, args.tff_redundancy])
        results.extend(ppls)
        csv_file_path = os.path.join(results_dir, 'results.csv')
        with open(csv_file_path, mode='a', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(results)

    if args.save:
        print(f'saving results to {exp_dir}')
        logging.info(f'saving results to {exp_dir}')
        # save the fake quantized model
        torch.save(model.state_dict(), os.path.join(exp_dir, 'fake_quantized_model.ckpt'))
        # save the quantizers
        torch.save(quantizers, os.path.join(exp_dir, 'quantizers.pt'))
        # save the packed the model
        llama_pack3(model, quantizers, args.l_den, args.tff_redundancy, args.wbits, tffs=tffs)
        torch.save(model.state_dict(), os.path.join(exp_dir, 'packed_model.ckpt'))

