import json
import random
import numpy as np

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
class EsmFoldseekDataset(LMDBDataset):
	"""
	Dataset of Mask Token Reconstruction with Structure information
	"""
	
	def __init__(self,
				 tokenizer: str,
				 max_length: int = 1024,
				 mask_ratio: float = 0.15,
				 **kwargs):
		"""

		Args:
			tokenizer: EsmTokenizer config path
			max_length: max length of sequence
			mask_ratio: ratio of masked tokens
			**kwargs: other arguments for LMDBDataset
		"""
		super().__init__(**kwargs)
		self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
		self.aa = [k for k in self.tokenizer.get_vocab().keys()]
		
		self.max_length = max_length
		self.mask_ratio = mask_ratio
	
	def __len__(self):
		return int(self._get("length"))

	def __getitem__(self, index):
		entry = json.loads(self._get(index))
		seq = entry['seq']
		
		# mask sequence for training
		tokens = self.tokenizer.tokenize(seq)[:self.max_length]

		while True:
			masked_tokens, labels = self._apply_bert_mask(tokens)
			masked_seq = " ".join(masked_tokens)
			
			# Check if there is at least one masked token
			if (labels != -1).sum().item() != 0:
				break

		return masked_seq, labels
	
	def _apply_bert_mask(self, tokens):
		masked_tokens = copy.copy(tokens)
		labels = torch.full((len(tokens) + 2,), -1, dtype=torch.long)
		for i in range(len(tokens)):
			token = tokens[i]
			
			prob = random.random()
			if prob < self.mask_ratio:
				prob /= self.mask_ratio
				labels[i + 1] = self.tokenizer.convert_tokens_to_ids(token)
				
				if prob < 0.8:
					# 80% mask combined token
					token = "#" + token[-1]
					# token = self.tokenizer.mask_token
				elif prob < 0.9:
					# 10% chance to change to random token
					token = random.choice(self.aa)
				else:
					# 10% chance to keep current token
					pass

				masked_tokens[i] = token
				
		return masked_tokens, labels
	
	def collate_fn(self, batch):
		seqs, label_ids = tuple(zip(*batch))
		
		label_ids = pad_sequences(label_ids, -1)
		labels = {"labels": label_ids}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
		inputs = {"inputs": encoder_info}

		return inputs, labels