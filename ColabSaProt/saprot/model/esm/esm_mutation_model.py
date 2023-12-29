import os
import torch
import copy
import numpy as np
import json
import torchmetrics


from utils.constants import aa_set
from ..model_interface import register_model
from .base import EsmBaseModel


@register_model
class EsmMutationModel(EsmBaseModel):
    def __init__(self,
                 use_bias_feature: bool = False,
                 MSA_log_path: str = None,
                 log_clinvar: bool = False,
                 log_dir: str = None,
                 **kwargs):
        """
        Args:
            use_bias_feature: Whether to use structure information as bias feature
            
            MSA_log_path: If not None, the model will load MSA log from this path (following Tranception paper)
            
            log_clinvar: If True, the model will log the predicted evolutionary indices for ClinVar variants
            
            log_dir: If log_clinvar is True, the model will save the predicted evolutionary indices for ClinVar variants
            
            **kwargs: other arguments for EsmBaseModel
        """
        self.use_bias_feature = use_bias_feature
        self.MSA_log_path = MSA_log_path
        self.MSA_info_dict = {}
        if MSA_log_path:
            with open(MSA_log_path, "r") as r:
                for line in r:
                    data = json.loads(line)
                    data["MSA_log_prior"] = torch.tensor(data["MSA_log_prior"])
                    self.MSA_info_dict[data["DMS_id"]] = data
        
        self.log_clinvar = log_clinvar
        self.log_dir = log_dir
        if log_clinvar:
            self.mut_info_list = []
                    
        super().__init__(task="lm", **kwargs)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_spearman": torchmetrics.SpearmanCorrCoef()}

    def forward(self, wild_type, seqs, mut_info, structure_content, structure_type, plddt):
        # if self.use_bias_feature and getattr(self, "coords", None) is None:
        #     structure_type = "cif" if structure_type == "mmcif" else structure_type
        #     tmp_path = f"EsmMutationModel_{self.global_rank}.{structure_type}"
        #     with open(tmp_path, "w") as f:
        #         f.write(structure_content)
        #
        #     self.coords = parse_structure(tmp_path, ["A"])["A"]["coords"]
        #     os.remove(tmp_path)
            
        ins_seqs = []
        ori_seqs = []
        mut_data = []
        
        # The running bottleneck is two forward passes of the model to deal with insertion
        # Therefore we only forward pass the model twice for sequences with insertion
        ins_dict = {}
        
        for i, (seq, info) in enumerate(zip(seqs, mut_info)):
            # We adopt the same strategy for esm2 model as in esm2 inverse folding paper
            ori_seq = [aa for aa in wild_type]
            ins_seq = copy.deepcopy(ori_seq)
            tmp_data = []
            ins_num = 0
            
            # To indicate whether there is insertion in the sequence
            flag = False
            
            for single in info.split(":"):
                # Mask the amino acid where the mutation happens
                # -1 is added because the index starts from 1 and we need to convert it to 0
                if single[0] in aa_set:
                    ori_aa, pos, mut_aa = single[0], int(single[1:-1]), single[-1]
                    ori_seq[pos - ins_num - 1] = self.tokenizer.mask_token
                    ins_seq[pos - 1] = self.tokenizer.mask_token
                    
                    tmp_data.append((ori_aa, pos - ins_num, mut_aa, pos))
                
                # For insertion
                else:
                    ins_dict[i] = len(ins_dict)
                    flag = True
                    
                    ins_num += 1
                    ins_pos = int(single[:-1])
                    ins_seq = ins_seq[:ins_pos - 1] + [self.tokenizer.mask_token] + ins_seq[ins_pos - 1:]
            
            if flag:
                ins_seqs.append(" ".join(ins_seq))
                
            ori_seqs.append(" ".join(ori_seq))
            mut_data.append(tmp_data)
        
        device = self.device
        
        if len(ins_seqs) > 0:
            ins_inputs = self.tokenizer.batch_encode_plus(ins_seqs, return_tensors="pt", padding=True)
            ins_inputs = {k: v.to(device) for k, v in ins_inputs.items()}
            if self.use_bias_feature:
                coords = [copy.deepcopy(self.coords) for _ in range(len(seqs))]
                self.add_bias_feature(ins_inputs, coords)
                
            ins_outputs = self.model(**ins_inputs)
            ins_probs = ins_outputs['logits'].softmax(dim=-1)
            
        ori_inputs = self.tokenizer.batch_encode_plus(ori_seqs, return_tensors="pt", padding=True)
        ori_inputs = {k: v.to(device) for k, v in ori_inputs.items()}
        if self.use_bias_feature:
            coords = [copy.deepcopy(self.coords) for _ in range(len(seqs))]
            self.add_bias_feature(ori_inputs, coords)
            
        ori_outputs = self.model(**ori_inputs)
        ori_probs = ori_outputs['logits'].softmax(dim=-1)
        
        if self.MSA_log_path is not None:
            aa2id = {"A": 5, "C": 6, "D": 7, "E": 8, "F": 9, "G": 10, "H": 11, "I": 12, "K": 13, "L": 14, "M": 15,
                     "N": 16, "P": 17, "Q": 18, "R": 19, "S": 20, "T": 21, "V": 22, "W": 23, "Y": 24}
            DMS_id = os.path.basename(self.trainer.datamodule.test_lmdb)
            MSA_info = self.MSA_info_dict[DMS_id]
            MSA_log_prior = MSA_info["MSA_log_prior"].to(device)
            st, ed = MSA_info["MSA_start"], MSA_info["MSA_end"]

        preds = []
        for i, data_list in enumerate(mut_data):
            pred = 0
            for data in data_list:
                ori_aa, ori_pos, mut_aa, ins_pos = data

                ori_prob = ori_probs[i, ori_pos, self.tokenizer.convert_tokens_to_ids(ori_aa)]
                
                if i in ins_dict:
                    mut_prob = ins_probs[ins_dict[i], ins_pos, self.tokenizer.convert_tokens_to_ids(mut_aa)]
                else:
                    mut_prob = ori_probs[i, ins_pos, self.tokenizer.convert_tokens_to_ids(mut_aa)]

                # Add MSA info if available
                if self.MSA_log_path is not None and st <= ori_pos -1 < ed:
                    ori_msa_prob = MSA_log_prior[ori_pos - 1 - st, aa2id[ori_aa]]
                    mut_msa_prob = MSA_log_prior[ori_pos - 1 - st, aa2id[mut_aa]]
                    pred += 0.4 * torch.log(mut_prob / ori_prob) + 0.6 * (mut_msa_prob - ori_msa_prob)
                
                else:
                    # compute zero-shot score
                    pred += torch.log(mut_prob / ori_prob)

            preds.append(pred)

        if self.log_clinvar:
            self.mut_info_list.append((mut_info, -torch.tensor(preds)))

        return torch.tensor(preds).to(ori_probs)

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels']
        self.test_spearman(outputs, fitness)

    def test_epoch_end(self, outputs):
        spearman = self.test_spearman.compute()
        self.reset_metrics("test")
        self.log("spearman", spearman, sync_dist=True)
        
        if self.use_bias_feature:
            self.coords = None
        
        if self.log_clinvar:
            # Get dataset name
            name = os.path.basename(self.trainer.datamodule.test_lmdb)
            log_path = f"{self.log_dir}/{name}.csv"
            with open(log_path, "w") as w:
                w.write("protein_name,mutations,evol_indices\n")
               
                for mut_info, preds in self.mut_info_list:
                    for mut, pred in zip(mut_info, preds):
                        w.write(f"{name},{mut},{pred}\n")
                        
            self.mut_info_list = []