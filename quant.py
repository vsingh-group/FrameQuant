import numpy as	np
import torch
import torch.nn	as nn
from torch.utils.cpp_extension import load
from utils.quant_utils import get_tff_blks,	tff_project, inv_tff
from modelutils	import DEV
import os

curr_path =	'./packbit'
src_files =	[os.path.join(curr_path, 'extension', file)	for	file in	['cuda_kernel.cu', 'torch_extension.cpp']]
packbit	= load('packbit', src_files, verbose = True)


def	quantize(x,	scale, zero, maxq, use_float_bias=False):
	if use_float_bias:
		return quantize_float_bias(x,scale,	zero, maxq)
	else:
		return quantize_int_bias(x,	scale, zero, maxq)

def	quantize_float_bias(x, scale, zero,	maxq):
	q =	torch.round(x/scale	+zero)
	q =	torch.clamp(q, 0, maxq)

	return scale *(q-zero)

def	quantize_int_bias(x, scale,	zero, maxq):
	if maxq	< 0:
		return (x >	scale /	2).float() * scale + (x	< zero / 2).float()	* zero
	q =	torch.clamp(torch.round(x /	scale) + zero, 0, maxq)
	return scale * (q -	zero)

class Quantizer(nn.Module):

	def	__init__(self, shape=1):
		super(Quantizer, self).__init__()
		self.register_buffer('maxq', torch.tensor(0))
		self.register_buffer('scale', torch.zeros(shape))
		self.register_buffer('zero', torch.zeros(shape))

	def	configure(
		self,
		bits, perchannel=False,	sym=True, 
		mse=False, norm=2.4, grid=100, maxshrink=.8,
		trits=False, x_sigma=None, use_float_bias =False
	):
		self.maxq =	torch.tensor(2 ** bits - 1)
		self.perchannel	= perchannel
		self.sym = sym
		self.mse = mse
		self.norm =	norm
		self.grid =	grid
		self.maxshrink = maxshrink 
		self.x_sigma = x_sigma
		self.use_float_bias	= use_float_bias
		if trits:
			self.maxq =	torch.tensor(-1) 

	def	find_params(self, x, weight=False):
		if self.x_sigma	is None:
			self.find_params_gptq(x, weight=weight)
		else:
			self.find_params_FrameQuant(x, weight=weight)

	def	find_params_gptq(self, x, weight=False):
		dev	= x.device
		self.maxq =	self.maxq.to(dev)

		shape =	x.shape
		if self.perchannel:
			if weight:
				x =	x.flatten(1)
			else:
				if len(shape) == 4:
					x =	x.permute([1, 0, 2,	3])
					x =	x.flatten(1)
				if len(shape) == 3:
					x =	x.reshape((-1, shape[-1])).t()
				if len(shape) == 2:
					x =	x.t()
		else:
			x =	x.flatten().unsqueeze(0)

		tmp	= torch.zeros(x.shape[0], device=dev)
		xmin = torch.minimum(x.min(1)[0], tmp).to(dev)
		xmax = torch.maximum(x.max(1)[0], tmp).to(dev)

		if self.sym:
			xmax = torch.maximum(torch.abs(xmin), xmax)
			tmp	= xmin < 0
			if torch.any(tmp):
				xmin[tmp] =	-xmax[tmp]
		tmp	= (xmin	== 0) &	(xmax == 0)
		xmin[tmp] =	-1
		xmax[tmp] =	+1

		if self.maxq < 0:
			self.scale = xmax
			self.zero	= xmin
		else:
			self.scale = (xmax - xmin) / self.maxq
			if self.sym:
				self.zero	= torch.full_like(self.scale, (self.maxq + 1) /	2)
			else:
				self.zero	= torch.round(-xmin	/ self.scale)

		if self.mse:
			best = torch.full([x.shape[0]],	float('inf'), device=dev)
			for	i in range(int(self.maxshrink *	self.grid)):
				p =	1 -	i /	self.grid 
				xmin1 =	p *	xmin
				xmax1 =	p *	xmax
				scale1 = (xmax1	- xmin1) / self.maxq
				zero1 =	torch.round(-xmin1 / scale1) if	not	self.sym else self.zero
				q =	quantize(x,	scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
				q -= x
				q.abs_()
				q.pow_(self.norm)
				err	= torch.sum(q, 1)
				tmp	= err <	best
				if torch.any(tmp):
					best[tmp] =	err[tmp]
					self.scale[tmp]	= scale1[tmp]
					self.zero[tmp] = zero1[tmp]
		if not self.perchannel:
			if weight:
				tmp	= shape[0]
			else:
				tmp	= shape[1] if len(shape) !=	3 else shape[2]
			self.scale = self.scale.repeat(tmp)
			self.zero =	self.zero.repeat(tmp)

		if weight:
			shape =	[-1] + [1] * (len(shape) - 1)
			self.scale = self.scale.reshape(shape)
			self.zero =	self.zero.reshape(shape)
			return
		if len(shape) == 4:
			self.scale = self.scale.reshape((1,	-1,	1, 1))
			self.zero =	self.zero.reshape((1, -1, 1, 1))
		if len(shape) == 3:
			self.scale = self.scale.reshape((1,	1, -1))
			self.zero =	self.zero.reshape((1, 1, -1)) 
		if len(shape) == 2:
			self.scale = self.scale.unsqueeze(0)
			self.zero =	self.zero.unsqueeze(0)

	def	find_params_FrameQuant(self, x,	weight=False):
		dev	= x.device
		self.maxq =	self.maxq.to(dev)

		shape =	x.shape
		if self.perchannel:
			if weight:
				x =	x.flatten(1)
			else:
				if len(shape) == 4:
					x =	x.permute([1, 0, 2,	3])
					x =	x.flatten(1)
				if len(shape) == 3:
					x =	x.reshape((-1, shape[-1])).t()
				if len(shape) == 2:
					x =	x.t()
		else:
			x =	x.flatten().unsqueeze(0)

		xmin = x.min(1,	keepdim=True)[0]
		xmax = x.max(1,	keepdim=True)[0]

		if self.sym:
			xmax = torch.maximum(torch.abs(xmin), xmax)
			tmp	= xmin < 0
			if torch.any(tmp):
				xmin[tmp] =	-xmax[tmp]

		tmp	= (xmin	== 0) &	(xmax == 0)
		xmin[tmp] =	-1
		xmax[tmp] =	+1

		xvar = x.var(dim=1,	keepdim=True)
		xstd = torch.sqrt(xvar)
		self.scale = self.x_sigma*xstd/(self.maxq/2)

		if self.sym:
			self.zero =	torch.full_like(self.scale,	(self.maxq + 1)	/ 2)
		else:
			self.zero =	(1 - x.mean(dim=1, keepdim=True) / (2*xstd)) * self.maxq/2
			if not self.use_float_bias:
				self.zero =	torch.round(self.zero)

		if self.mse:
			best = torch.full([x.shape[0]],	float('inf'), device=dev)
			for	i in range(int(self.maxshrink *	self.grid)):
				p =	1 -	i /	self.grid
				xmin1 =	p *	xmin
				xmax1 =	p *	xmax
				scale1 = (xmax1	- xmin1) / self.maxq
				zero1 =	torch.round(-xmin1 /
									scale1)	if not self.sym	else self.zero
				q =	quantize(x,	scale1.unsqueeze(1), zero1.unsqueeze(1),
							 self.maxq)
				q -= x
				q.abs_()
				q.pow_(self.norm)
				err	= torch.sum(q, 1)
				tmp	= err <	best
				if torch.any(tmp):
					best[tmp] =	err[tmp]
					self.scale[tmp]	= scale1[tmp]
					self.zero[tmp] = zero1[tmp]
		if not self.perchannel:
			if weight:
				tmp	= shape[0]
			else:
				tmp	= shape[1] if len(shape) !=	3 else shape[2]
			self.scale = self.scale.repeat(tmp)
			self.zero =	self.zero.repeat(tmp)

		if weight:
			shape =	[-1] + [1] * (len(shape) - 1)
			self.scale = self.scale.reshape(shape)
			self.zero =	self.zero.reshape(shape)
			return
		if len(shape) == 4:
			self.scale = self.scale.reshape((1,	-1,	1, 1))
			self.zero =	self.zero.reshape((1, -1, 1, 1))
		if len(shape) == 3:
			self.scale = self.scale.reshape((1,	1, -1))
			self.zero =	self.zero.reshape((1, 1, -1))
		if len(shape) == 2:
			self.scale = self.scale.unsqueeze(0)
			self.zero =	self.zero.unsqueeze(0)

	def	quantize(self, x):
		if self.ready():
			return quantize(x, self.scale, self.zero, self.maxq)
		return x

	def	enabled(self):
		return self.maxq > 0

	def	ready(self):
		return torch.all(self.scale	!= 0)
	


tffs = {}

class Quant3Linear(nn.Module): 

	def	__init__(self, infeatures, outfeatures, l_den=16, tff_redundancy=1.0, wbits=2, faster=False):
		super().__init__()
		self.register_buffer('zeros', torch.zeros((wbits*outfeatures//32, ), dtype=torch.int))
		self.register_buffer('scale', torch.zeros((outfeatures, 1), dtype=torch.half))
		self.register_buffer('bias', torch.zeros(outfeatures))
		self.register_buffer(
			'qweight', torch.zeros((infeatures // 32 * wbits * outfeatures), dtype=torch.int)
		)
		self.faster	= faster
		self.register_buffer('l_seed', torch.tensor(1, dtype=torch.int))
		self.register_buffer('p_seed', torch.tensor(1, dtype=torch.int))
		self.register_buffer('ori_l', torch.tensor(1, dtype=torch.int))
		self.register_buffer('ori_p', torch.tensor(1, dtype=torch.int))
		self.wbits = wbits
		self.tff_redundancy=tff_redundancy
		self.l_den = l_den

	@torch.no_grad
	def	pack(self, linear, quantizer, projs):
		if linear.bias is not None:
			self.bias = linear.bias.clone()
		wt = linear.weight.data
		# compute projs
		l_n, p_n = wt.shape
		for	_d in (l_n,	p_n):
			get_tff_blks('', _d, projs,	self.l_den, self.tff_redundancy)
		P_l_T, P_prev_T	= projs[l_n], projs[p_n]

		# integer weights
		W =	tff_project(wt.type(P_l_T.type()).to(DEV), quantizer.l_seed+1234, P_prev_T,	DEV)
		W =	tff_project(W.T, quantizer.l_seed+4321,	None, DEV)
		W =	W.T
		int_wt = (W) / quantizer.scale.to(DEV) + quantizer.zero.to(DEV)
		int_wt = torch.clamp(torch.round(int_wt), min=0, max = quantizer.maxq).type(torch.int).ravel()
		# pack weights
		self.qweight = packbit.pack_fn(int_wt, self.wbits).to('cpu')
		###	pack the quantizer
		# pack the scale
		self.scale = quantizer.scale.half().contiguous().clone() # GPTQ4Llama also transposes the weights
		# pack the zeros
		self.zeros = packbit.pack_fn(quantizer.zero.contiguous().ravel().type(torch.int), self.wbits)
		# pack the roatation seeds
		self.l_seed	= torch.tensor(quantizer.l_seed).type(torch.int)
		self.p_seed	= torch.tensor(quantizer.p_seed).type(torch.int)
		# pack the original	sizes
		self.ori_l = torch.tensor(wt.shape[0]).type(torch.int)
		self.ori_p = torch.tensor(wt.shape[1]).type(torch.int)

	@torch.no_grad
	def	unpack(self, out_size, inp_size, projs):
		# unpack weights and zero
		wt = packbit.unpack_fn(self.qweight, self.wbits, out_size * inp_size).to(torch.float32)
		zero = packbit.unpack_fn(self.zeros, self.wbits, out_size).to(torch.float32).reshape(-1,1)
		# fake quant weights
		wt = wt.reshape(out_size, inp_size)
		wt = (wt - zero) * self.scale.to(torch.float32)
		# compute projs
		l_n, p_n = wt.shape
		for	_d in (l_n,	p_n):
			get_tff_blks('', _d, projs,	self.l_den, self.tff_redundancy, DEV)
		P_l_T, P_prev_T	= projs[l_n], projs[p_n]
		# inv tff transform
		wt = inv_tff(wt, self.l_seed.item()+1234, P_prev_T, DEV)
		wt = inv_tff(wt.T, self.l_seed.item()+4321, None, DEV)
		wt = wt.T

		return wt

	def	forward(self, x):
		out_size = self.scale.shape[0]
		inp_size = len(self.qweight) * 32 //(2*out_size)
		unpacked_wt	= self.unpack(out_size,	inp_size, tffs)

		out = nn.functional.linear(x, unpacked_wt.type(x.type()), self.bias.type(x.type()))
		return out
		

def	make_quant3(module,	names, l_den, tff_redundancy, wbits=2, name='',	faster=False):
	if isinstance(module, Quant3Linear):
		return
	for	attr in	dir(module):
		tmp	= getattr(module, attr)
		name1 =	name + '.' + attr if name != ''	else attr
		if name1 in	names:
			inp_size = tmp.in_features
			out_size = tmp.out_features
			l_tff =	inp_size //	l_den
			k_tff =	round(inp_size // l_tff	* tff_redundancy)
			inp_size = l_tff * k_tff
			setattr(
				module,	attr, Quant3Linear(inp_size, out_size, l_den=l_den, tff_redundancy=tff_redundancy, wbits=wbits, faster=faster)
			)
	for	name1, child in	module.named_children():
		make_quant3(child, names, l_den, tff_redundancy, wbits=wbits, name=name	+ '.' +	name1 if name != ''	else name1,	faster=faster)
