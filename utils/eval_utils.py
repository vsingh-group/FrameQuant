import torch
import torch.nn as nn
import time
from tqdm import tqdm

@torch.no_grad()
def	llama_eval(model, testenc, dev, nsamples=None, verbose=False):
	print('Evaluating ...')
	results	= {}

	testenc	= testenc.input_ids
	if nsamples is None: # calculate the nsamples from the test encodings given
		nsamples = testenc.numel() //	model.seqlen
	
	results['num_tokens'] =	nsamples * model.seqlen

	use_cache =	model.config.use_cache
	model.config.use_cache = False
	layers = model.model.layers

	model.model.embed_tokens = model.model.embed_tokens.to(dev)
	layers[0] =	layers[0].to(dev)

	dtype =	next(iter(model.parameters())).dtype
	inps = torch.zeros(
		(nsamples, model.seqlen, model.config.hidden_size),	dtype=dtype, device=dev
	)
	cache =	{'i': 0, 'attention_mask': None}

	class Catcher(nn.Module):
		def	__init__(self, module):
			super().__init__()
			self.module	= module
		def	forward(self, inp, **kwargs):
			inps[cache['i']] = inp
			cache['i'] += 1
			cache['attention_mask']	= kwargs['attention_mask']
			cache['position_ids'] =	kwargs['position_ids']
			raise ValueError
	layers[0] =	Catcher(layers[0])
	for	i in range(nsamples):
		batch =	testenc[:, (i *	model.seqlen):((i +	1) * model.seqlen)].to(dev)
		try:
			model(batch)
		except ValueError:
			pass
	layers[0] =	layers[0].module

	layers[0] =	layers[0].cpu()
	model.model.embed_tokens = model.model.embed_tokens.cpu()
	torch.cuda.empty_cache()

	outs = torch.zeros_like(inps)
	attention_mask = cache['attention_mask']
	position_ids = cache['position_ids']

	start_time = time.time()
	for	i in tqdm(range(len(layers))):
		layer =	layers[i].to(dev)
		
		for	j in range(nsamples):
			outs[j]	= layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
		layers[i] =	layer.cpu()
		del	layer
		torch.cuda.empty_cache()
		inps, outs = outs, inps
	stop_time =	time.time()
	results['run_time']	= stop_time	- start_time
	results['inf_speed'] = results['num_tokens'] / results['run_time']

	if verbose:
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
		print('numtokens = '+str(results['num_tokens']))
		print('runtime = '+str(results['run_time']))
		print('inference speed = '+str(results['inf_speed']))
		print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
	
	if model.model.norm	is not None:
		model.model.norm = model.model.norm.to(dev)
	model.lm_head =	model.lm_head.to(dev)

	testenc	= testenc.to(dev)
	nlls = []
	for	i in tqdm(range(nsamples)):
		hidden_states =	inps[i].unsqueeze(0)
		if model.model.norm	is not None:
			hidden_states =	model.model.norm(hidden_states)
		lm_logits =	model.lm_head(hidden_states)
		shift_logits = lm_logits[:,	:-1, :].contiguous()
		shift_labels = testenc[
			:, (i *	model.seqlen):((i +	1) * model.seqlen)
		][:, 1:]
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
		neg_log_likelihood = loss.float() *	model.seqlen
		nlls.append(neg_log_likelihood)
	ppl	= torch.exp(torch.stack(nlls).sum()	/ (nsamples	* model.seqlen))
	print(ppl.item())

	model.config.use_cache = use_cache

	return ppl,	results
