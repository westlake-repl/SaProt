import json
import random
import copy
import math
import os

from torch.utils.data import Subset
from transformers import EsmTokenizer
from ..data_interface import register_dataset
from ..lmdb_dataset import *
from data.data_transform import *


@register_dataset
class EsmSequenceDesignDataset(LMDBDataset):
    """
    Dataset of Mask Token Reconstruction with Structure information
    """
    
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """

        Args:
            tokenizer: EsmTokenizer config path
            max_length: max length of sequence
            **kwargs: other arguments for LMDBDataset
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.aa = [k for k in self.tokenizer.get_vocab().keys()]
        
        self.max_length = max_length
    
    def __len__(self):
        return int(self._get("length"))
    
    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length]
        
        # mask sequence for training
        ids = self.tokenizer.encode(seq, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        masked_tokens, labels = self._apply_mask(tokens)
        masked_seq = " ".join(masked_tokens)
        
        return masked_seq, labels
    
    def _apply_mask(self, tokens):
        masked_tokens = copy.copy(tokens)
        labels = torch.full((len(tokens) + 2,), -1, dtype=torch.long)
        for i in range(len(tokens)):
            token = tokens[i]
            labels[i + 1] = self.tokenizer.convert_tokens_to_ids(token)
            token = "#" + token[-1]
            masked_tokens[i] = token
        
        return masked_tokens, labels
    
    def collate_fn(self, batch):
        seqs, label_ids = tuple(zip(*batch))
        
        label_ids = pad_sequences(label_ids, -1)
        labels = {"labels": label_ids}
        
        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}
        
        return inputs, labels