import torch
import json
import random

from ..data_interface import register_dataset
from transformers import EsmTokenizer
from ..lmdb_dataset import *


@register_dataset
class EsmClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 use_bias_feature: bool = False,
                 max_length: int = 1024,
                 preset_label: int = None,
                 mask_struc_ratio: float = None,
                 mask_seed: int = 20000812,
                 plddt_threshold: float = None,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            use_bias_feature: If True, structure information will be used
            max_length: Max length of sequence
            preset_label: If not None, all labels will be set to this value
            mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
            mask_seed: Seed for mask_struc_ratio
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.use_bias_feature = use_bias_feature
        self.preset_label = preset_label
        self.mask_struc_ratio = mask_struc_ratio
        self.mask_seed = mask_seed
        self.plddt_threshold = plddt_threshold

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq']

        # Mask structure tokens
        if self.mask_struc_ratio is not None:
            tokens = self.tokenizer.tokenize(seq)
            mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]
            
            # Randomly shuffle the mask candidates and set seed to ensure mask is consistent
            random.seed(self.mask_seed)
            random.shuffle(mask_candi)
            
            # Mask first n structure tokens
            mask_num = int(len(mask_candi) * self.mask_struc_ratio)
            for i in range(mask_num):
                idx = mask_candi[i]
                tokens[idx] = tokens[idx][:-1] + "#"
            
            seq = "".join(tokens)

        # Mask structure tokens with pLDDT < threshold
        if self.plddt_threshold is not None:
            plddt = entry["plddt"]
            tokens = self.tokenizer.tokenize(seq)
            seq = ""
            for token, score in zip(tokens, plddt):
                if score < self.plddt_threshold:
                    seq += token[:-1] + "#"
                else:
                    seq += token

        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)
        
        if self.use_bias_feature:
            coords = {k: v[:self.max_length] for k, v in entry['coords'].items()}
        else:
            coords = None

        label = entry["label"] if self.preset_label is None else self.preset_label

        return seq, label, coords

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids, coords = tuple(zip(*batch))

        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}
    
        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}
        if self.use_bias_feature:
            inputs["coords"] = coords

        return inputs, labels