import json
import random
import copy
import math
import os
import numpy as np


from torch.utils.data import Subset
from transformers import EsmTokenizer
from ..data_interface import register_dataset
from ..lmdb_dataset import *


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
	batch_size = len(sequences)
	shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

	if dtype is None:
		dtype = sequences[0].dtype

	if isinstance(sequences[0], np.ndarray):
		array = np.full(shape, constant_value, dtype=dtype)
	elif isinstance(sequences[0], torch.Tensor):
		device = sequences[0].device
		array = torch.full(shape, constant_value, dtype=dtype, device=device)

	for arr, seq in zip(array, sequences):
		arrslice = tuple(slice(dim) for dim in seq.shape)
		arr[arrslice] = seq

	return array


@register_dataset
class EsmLMDataset(LMDBDataset):
	"""
	Dataset of Mask Token Reconstruction with Structure information
	"""

	def __init__(self,
	             tokenizer: str,
	             max_length: int = 512,
	             use_bias_feature: bool = False,
				 mask_ratio: float = 0.15,
				 **kwargs):
		"""

		Args:
			tokenizer: EsmTokenizer config path
			max_length: max length of sequence
			use_bias_feature: whether to use structure information
			mask_ratio: ratio of masked tokens
			**kwargs: other arguments for LMDBDataset
		"""
		super().__init__(**kwargs)
		self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
		self.aa = [k for k in self.tokenizer.get_vocab().keys()]

		self.max_length = max_length
		self.use_bias_feature = use_bias_feature
		self.mask_ratio = mask_ratio
	
	def __len__(self):
		return int(self._get("length"))
	
	def __getitem__(self, index):
		entry = json.loads(self._get(index))
		seq = entry['seq'][:self.max_length]
		
		# mask sequence for training
		ids = self.tokenizer.encode(seq, add_special_tokens=False)
		tokens = self.tokenizer.convert_ids_to_tokens(ids)
		masked_tokens, labels = self._apply_bert_mask(tokens)
		masked_seq = " ".join(masked_tokens)
		
		if self.use_bias_feature:
			coords = {k: v[:self.max_length] for k, v in entry['coords'].items()}
			# coords = {}
			# mask = (labels != -1)[1: -1]
			#
			# for k, v in entry["coords"].items():
			# 	coord = torch.tensor(v, dtype=torch.float32)
			# 	mean = torch.zeros_like(coord)
			# 	std = torch.full_like(coord, self.noise)
			#
			# 	# Add noise on masked amino acids
			# 	coord[mask] += torch.normal(mean, std)[mask]
			# 	coords[k] = coord.numpy().tolist()
				
			# # Get mask indices
			# # Get distance map of CA
			# CA = torch.tensor(coords['CA'])
			# dist = torch.cdist(CA, CA)
			# nearest = torch.argsort(dist, dim=-1)
			#
			# # Add indices of amino acids that are nearest to masked tokens to the mask list
			# mask = []
			# for idx in (labels[1:-1] != -1).nonzero():
			# 	mask += [(idx.item(), pos.item()) for pos in nearest[idx, :self.mask_structure_num].squeeze(dim=0)]

		else:
			coords = None

		return masked_seq, labels, coords
	
	def _apply_bert_mask(self, tokens):
		masked_tokens = copy.copy(tokens)
		labels = torch.full((len(tokens)+2,), -1, dtype=torch.long)
		for i in range(len(tokens)):
			token = tokens[i]
			
			prob = random.random()
			if prob < self.mask_ratio:
				prob /= self.mask_ratio
				labels[i+1] = self.tokenizer.convert_tokens_to_ids(token)
				
				if prob < 0.8:
					# 80% random change to mask token
					token = self.tokenizer.mask_token
				elif prob < 0.9:
					# 10% chance to change to random token
					token = random.choice(self.aa)
				else:
					# 10% chance to keep current token
					pass
				
				masked_tokens[i] = token
		
		return masked_tokens, labels
	
	def _apply_dist_mask(self, tokens, CA: torch.Tensor, ratio=0.15):
		"""
		Apply mask to sequence based on distance between CA
		Args:
			tokens: sequence token ids [Seq_len]
			CA: CA coordinates [Seq_len, 3]
			ratio: ratio of masked tokens

		Returns:
			masked_tokens: masked sequence tokens
			labels: labels for masked tokens

		"""
		
		# Calculate distance matrix
		CA = CA.unsqueeze(1).expand(-1, CA.size(0), -1)
		dist = (CA - CA.transpose(0, 1)).norm(dim=-1)
		indices = dist.argsort(dim=-1)

		# Randomly select first mask position
		pos = random.choice(range(len(tokens)))
		maks_set = {pos}
		cnt = 1
		mut_num = int(len(tokens) * ratio)
		while cnt < mut_num:
			# Choose the nearest amino acid as the next mask position
			for next_pos in indices[pos, 1:]:
				next_pos = next_pos.item()
				
				if next_pos in maks_set or math.fabs(next_pos - pos) <= 6:
					continue
				else:
					maks_set.add(next_pos)
					cnt += 1
					pos = next_pos
					break
		
		# Mask sequence and generate labels
		masked_tokens = copy.copy(tokens)
		labels = torch.full((len(tokens)+2,), -1, dtype=torch.long)
		for i in maks_set:
			token = tokens[i]
			labels[i+1] = self.tokenizer.convert_tokens_to_ids(token)
			prob = random.random()
			if prob < 0.8:
				# 80% random change to mask token
				token = self.tokenizer.mask_token
			elif prob < 0.9:
				# 10% chance to change to random token
				token = random.choice(self.aa)
			else:
				# 10% chance to keep current token
				pass
			
			masked_tokens[i] = token
		
		return masked_tokens, labels
	
	def collate_fn(self, batch):
		seqs, label_ids, coords = tuple(zip(*batch))

		label_ids = pad_sequences(label_ids, -1)
		labels = {"labels": label_ids}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
		inputs = {"inputs": encoder_info}

		if self.use_bias_feature:
			inputs['coords'] = coords

		return inputs, labels