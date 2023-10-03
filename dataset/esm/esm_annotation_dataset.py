import pandas as pd
import json
import numpy as np

from torch.utils.data import Subset
from transformers import EsmTokenizer
from ..lmdb_dataset import *
from ..data_interface import register_dataset


@register_dataset
class EsmAnnotationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 bias_feature: bool = False,
                 max_length: int = 1024,
                 mask_struc_ratio: float = None,
                 plddt_threshold: float = None,
                 **kwargs):
        """

        Args:
            tokenizer: EsmTokenizer config path
            
            bias_feature: If True, structure information will be used
            
            max_length: Max length of sequence
            
            mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
            
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            
            **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.bias_feature = bias_feature
        self.max_length = max_length
        self.mask_struc_ratio = mask_struc_ratio
        self.plddt_threshold = plddt_threshold
    
    def __getitem__(self, index):
        data = json.loads(self._get(index))
        seq = data['seq']
        
        # Mask structure tokens
        if self.mask_struc_ratio is not None:
            tokens = self.tokenizer.tokenize(seq)
            mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]
            
            # Randomly select tokens to mask
            mask_num = int(len(mask_candi) * self.mask_struc_ratio)
            mask_idx = np.random.choice(mask_candi, mask_num, replace=False)
            for i in mask_idx:
                tokens[i] = tokens[i][:-1] + "#"
            
            seq = "".join(tokens)
        
        # Mask structure tokens with pLDDT < threshold
        if self.plddt_threshold is not None:
            plddt = data["plddt"]
            tokens = self.tokenizer.tokenize(seq)
            seq = ""
            for token, score in zip(tokens, plddt):
                if score < self.plddt_threshold:
                    seq += token[:-1] + "#"
                else:
                    seq += token
                    
        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)
            
        coords = data['coords'][:self.max_length] if self.bias_feature else None
        
        label = data['label']
        if isinstance(label, str):
            label = json.loads(label)
        
        return seq, label, coords

    def collate_fn(self, batch):
        seqs, labels, coords = zip(*batch)

        model_inputs = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": model_inputs}
        # print(self.tokenizer.convert_ids_to_tokens(inputs['inputs']['input_ids'][0]))
        if self.bias_feature:
            inputs['structure_info'] = (coords,)

        labels = {"labels": torch.tensor(labels, dtype=torch.long)}
        
        return inputs, labels
    
    def __len__(self):
        return int(self._get('length'))
