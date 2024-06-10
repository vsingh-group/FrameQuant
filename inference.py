import time

import torch
import torch.nn	as nn
import argparse

from gptq import *
from modelutils	import *
from quant import *
import os
from utils.construct_tff import	construct_real_tff
from utils.quant_utils import compute_tffs
from utils.eval_utils import llama_eval
from datetime import datetime
from transformers import LlamaConfig, LlamaForCausalLM,	modeling_utils

from tqdm import tqdm
import math
from datautils import get_loaders

def	load_llama_from_config(model, default_type=torch.half):
	config = LlamaConfig.from_pretrained(model)

	def	noop(*args,	**kwargs):
		pass

	torch.nn.init.kaiming_uniform_ = noop
	torch.nn.init.uniform_ = noop
	torch.nn.init.normal_ =	noop

	if default_type	is not None:
		torch.set_default_dtype(default_type)
	modeling_utils._init_weights = False
	if default_type	is not None:
		torch.set_default_dtype(default_type)
	model =	LlamaForCausalLM(config)

	return model

def	load_quant(model, checkpoint, l_den, tff_redundancy, wbits, eval=True):

	model = load_llama_from_config(model, default_type=torch.half)

	torch.set_default_dtype(torch.float)
	if eval:
		model =	model.eval()
	layers = find_layers(model)
	for	name in	['lm_head']:
		if name	in layers:
			del	layers[name]
	make_quant3(model, layers, l_den, tff_redundancy, wbits)

	del	layers

	print('Loading model ...')
	model.load_state_dict(torch.load(checkpoint))

	model.seqlen = 2048
	print('Done.')

	return model

def	parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'model', type=str,
		help='path to the directory	where model	config file	is stored'
	)
	parser.add_argument(
		'saved_model', type=str,
		help='path to the directory	where saved model is stored'
	)
	parser.add_argument(
		'--l_den', type=int, default=16,
		help='denominator to be	used for L_tff'
	)
	parser.add_argument(
		'--tff_redundancy',	type=float,	default=1.0,
		help='Redundancy in	TFF	representations'
	)
	parser.add_argument(
		'--wbits', type=int, default=2,
		help='#bits	to use for quantization; use 16	for	evaluating base	model.'
	)
	parser.add_argument(
		'--seed',
		type=int, default=0, help='Seed	for	sampling the calibration data.'
	)

	args = parser.parse_args()

	return args

if __name__	== '__main__':

	args = parse_args()

	ckpt_path =	os.path.join(args.saved_model, 'packed_model.ckpt')

	print('loading model ...')
	model =	load_quant(args.model, ckpt_path, args.l_den, args.tff_redundancy, args.wbits, eval=True)

	datasets = ['wikitext2']
	for	dataset	in datasets:
		dataloader,	testloader = get_loaders(
			dataset, nsamples=2, seed=args.seed, model=args.model, seqlen=model.seqlen
		)
		print(dataset)
		ppl, results = llama_eval(model, testloader, DEV, nsamples = 2, verbose=True)



