import copy
import os
import torch
import json
import torchmetrics
import random
import numpy as np

from utils.constants import aa_set, foldseek_struc_vocab, aa_list
from ..model_interface import register_model
from .base import EsmBaseModel


@register_model
class EsmFoldseekMutationModel(EsmBaseModel):
    def __init__(self,
                 foldseek_path: str,
                 plddt_threshold: float = 0.,
                 mask_rate: float = None,
                 substitute_rate: float = None,
                 MSA_log_path: str = None,
                 log_clinvar: bool = False,
                 log_dir: str = None,
                 **kwargs):
        """
        Args:
            foldseek_path: Foldseek binary file path
            
            plddt_threshold: The threshold for plddt to determine whether a structure token should be masked

            mask_rate: If not None, the model will randomly mask structure tokens with this rate

            substitute_rate: If not None, the model will randomly substitute structure tokens with this rate
            
            MSA_log_path: If not None, the model will load MSA log from this path (follow Tranception paper)
            
            log_clinvar: If True, the model will log the predicted evolutionary indices for ClinVar variants
            
            log_dir: If log_clinvar is True, the model will save the predicted evolutionary indices for ClinVar variants
            
            **kwargs: Other arguments for EsmBaseModel
        """
        self.foldseek_path = foldseek_path
        self.plddt_threshold = plddt_threshold
        self.mask_rate = mask_rate
        self.substitute_rate = substitute_rate
        
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
    
    def get_struc_seq(self, structure_content, structure_type, plddt):
        structure_type = "cif" if structure_type == "mmcif" else structure_type
        
        # Sample a random rank to avoid file conflict
        rank = random.randint(0, 1000000)
        
        tmp_pdb_path = f"EsmFoldseekMutationModel_{self.global_rank}_{rank}.{structure_type}"
        tmp_save_path = f"EsmFoldseekMutationModel_{self.global_rank}_{rank}.tsv"
        
        # Save structure content to temporary file
        with open(tmp_pdb_path, "w") as w:
            w.write(structure_content)
        
        # Get foldseek structural sequecne
        cmd = f"{self.foldseek_path} structureto3didescriptor -v 0 --threads 1 {tmp_pdb_path} {tmp_save_path}"
        os.system(cmd)
        
        with open(tmp_save_path, "r") as r:
            line = r.readline()
            struc_seq = line.split("\t")[2]
        
        if plddt is not None:
            plddts = np.array(plddt)
            
            # Mask regions with plddt < threshold
            indices = np.where(plddts < self.plddt_threshold)[0]
            np_seq = np.array(list(struc_seq))
            np_seq[indices] = "#"
            struc_seq = "".join(np_seq)

        if self.mask_rate is not None:
            # Mask random structure tokens
            indices = np.random.choice(len(struc_seq), int(len(struc_seq) * self.mask_rate), replace=False)
            np_seq = np.array(list(struc_seq))
            np_seq[indices] = "#"
            struc_seq = "".join(np_seq)

        if self.substitute_rate is not None:
            # Substitute random structure tokens
            indices = np.random.choice(len(struc_seq), int(len(struc_seq) * self.substitute_rate), replace=False)
            np_seq = np.array(list(struc_seq))
            np_seq[indices] = np.random.choice(list(foldseek_struc_vocab), len(indices))
            struc_seq = "".join(np_seq)

        os.remove(tmp_pdb_path)
        os.remove(tmp_save_path)
        os.remove(tmp_save_path + ".dbtype")
        return struc_seq
    
    def forward(self, wild_type, seqs, mut_info, structure_content, structure_type, plddt):
        device = self.device
        
        if getattr(self, "struc_seq", None) is None:
            self.struc_seq = self.get_struc_seq(structure_content, structure_type, plddt)
        
        ins_seqs = []
        ori_seqs = []
        mut_data = []
        
        # The running bottleneck is two forward passes of the model to deal with insertion
        # Therefore we only forward pass the model twice for sequences with insertion
        ins_dict = {}
        
        for i, (seq, info) in enumerate(zip(seqs, mut_info)):
            # We adopt the same strategy for esm2 model as in esm2 inverse folding paper
            ori_seq = [a+b.lower() for a, b in zip(wild_type, self.struc_seq)]
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
                    
                    ori_seq[pos-ins_num-1] = "#" + ori_seq[pos-ins_num-1][-1]
                    ins_seq[pos-1] = "#" + ins_seq[pos-1][-1]
                    # ori_seq[pos - ins_num - 1] = self.tokenizer.mask_token
                    # ins_seq[pos - 1] = self.tokenizer.mask_token

                    tmp_data.append((ori_aa, pos-ins_num, mut_aa, pos))

                # For insertion
                else:
                    ins_dict[i] = len(ins_dict)
                    flag = True
                    
                    ins_num += 1
                    ins_pos = int(single[:-1])
                    ins_seq = ins_seq[:ins_pos-1] + ["##"] + ins_seq[ins_pos-1:]
                    # ins_seq = ins_seq[:ins_pos-1] + [self.tokenizer.mask_token] + ins_seq[ins_pos-1:]

            if flag:
                ins_seqs.append(" ".join(ins_seq))
                
            ori_seqs.append(" ".join(ori_seq))
            mut_data.append(tmp_data)
        
        if len(ins_seqs) > 0:
            ins_inputs = self.tokenizer.batch_encode_plus(ins_seqs, return_tensors="pt", padding=True)
            ins_inputs = {k: v.to(device) for k, v in ins_inputs.items()}
            ins_outputs = self.model(**ins_inputs)
            ins_probs = ins_outputs['logits'].softmax(dim=-1)
        
        ori_inputs = self.tokenizer.batch_encode_plus(ori_seqs, return_tensors="pt", padding=True)
        ori_inputs = {k: v.to(device) for k, v in ori_inputs.items()}
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

                ori_st = self.tokenizer.get_vocab()[ori_aa + foldseek_struc_vocab[0]]
                mut_st = self.tokenizer.get_vocab()[mut_aa + foldseek_struc_vocab[0]]

                ori_prob = ori_probs[i, ori_pos, ori_st: ori_st + len(foldseek_struc_vocab)].sum()
                # ori_sturc_aa = ori_aa + self.struc_seq[ori_pos-1].lower()
                # ori_struc_id = self.tokenizer.get_vocab()[ori_sturc_aa]
                # ori_prob = ori_probs[i, ori_pos, ori_struc_id]

                if i in ins_dict:
                    mut_prob = ins_probs[ins_dict[i], ins_pos, mut_st: mut_st + len(foldseek_struc_vocab)].sum()
                    # mut_prob = ins_probs[ins_dict[i], ins_pos, mut_st: mut_st + len(foldseek_struc_vocab)].max()
                else:
                    mut_prob = ori_probs[i, ori_pos, mut_st: mut_st + len(foldseek_struc_vocab)].sum()
                    # mut_prob = ori_probs[i, ins_pos, mut_st: mut_st + len(foldseek_struc_vocab)].max()
                    # mut_struc_aa = mut_aa + self.struc_seq[ori_pos-1].lower()
                    # mut_struc_id = self.tokenizer.get_vocab()[mut_struc_aa]
                    # mut_prob = ori_probs[i, ori_pos, mut_struc_id]
                
                # print(ori_prob, mut_prob)
                # struc_logits = []
                # ori_idx, mut_idx = None, None
                # for aa in aa_list:
                #     struct_aa = aa + self.struc_seq[ori_pos - 1].lower()
                #     aa_logits = ori_outputs['logits'][i, ori_pos, self.tokenizer.get_vocab()[struct_aa]]
                #     struc_logits.append(aa_logits)
                #
                #     if aa == ori_aa:
                #         ori_idx = len(struc_logits) - 1
                #
                #     if aa == mut_aa:
                #         mut_idx = len(struc_logits) - 1
                #
                # struc_probs = torch.softmax(torch.tensor(struc_logits), dim=-1)
                # ori_prob = struc_probs[ori_idx]
                # mut_prob = struc_probs[mut_idx]
                
                # Add MSA info if available
                if self.MSA_log_path is not None and st <= ori_pos -1 < ed:
                    ori_msa_prob = MSA_log_prior[ori_pos - 1 - st, aa2id[ori_aa]]
                    mut_msa_prob = MSA_log_prior[ori_pos - 1 - st, aa2id[mut_aa]]
                    pred += 0.4 * torch.log(mut_prob / ori_prob) + 0.6 * (mut_msa_prob - ori_msa_prob)
                
                # compute zero-shot score
                else:
                    pred += torch.log(mut_prob / ori_prob)
                    # ori_prob = ori_probs[i, ori_pos, ori_st: ori_st + len(foldseek_struc_vocab)]
                    # mut_prob = ori_probs[i, ori_pos, mut_st: mut_st + len(foldseek_struc_vocab)]
                    # pred += torch.log(mut_prob / ori_prob).mean()

            preds.append(pred)
        
        if self.log_clinvar:
            self.mut_info_list.append((mut_info, -torch.tensor(preds)))

        return torch.tensor(preds).to(ori_probs)

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels']

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(outputs.detach().float(), fitness.float())

    def test_epoch_end(self, outputs):
        spearman = self.test_spearman.compute()
        self.struc_seq = None
        self.reset_metrics("test")
        self.log("spearman", spearman)
        
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
    
    def predict_mut(self, seq: str, mut_info: str) -> float:
        """
               Predict the mutational effect of a given mutation
               Args:
                   seq: The wild type sequence

                   mut_info: The mutation information in the format of "A123B", where A is the original amino acid, 123 is the
                             position and B is the mutated amino acid. If multiple mutations are provided, they should be
                             separated by colon, e.g. "A123B:C124D".

               Returns:
                   The predicted mutational effect
               """
        tokens = self.tokenizer.tokenize(seq)
        for single in mut_info.split(":"):
            pos = int(single[1:-1])
            tokens[pos - 1] = "#" + tokens[pos - 1][-1]
        
        mask_seq = " ".join(tokens)
        inputs = self.tokenizer(mask_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(dim=-1)
            
            score = 0
            for single in mut_info.split(":"):
                ori_aa, pos, mut_aa = single[0], int(single[1:-1]), single[-1]
                ori_st = self.tokenizer.get_vocab()[ori_aa + foldseek_struc_vocab[0]]
                mut_st = self.tokenizer.get_vocab()[mut_aa + foldseek_struc_vocab[0]]
                
                ori_prob = probs[0, pos, ori_st: ori_st + len(foldseek_struc_vocab)].sum()
                mut_prob = probs[0, pos, mut_st: mut_st + len(foldseek_struc_vocab)].sum()
                
                score += torch.log(mut_prob / ori_prob)
        
        return score.item()
    
    def predict_pos_mut(self, seq: str, pos: int) -> dict:
        """
        Predict the mutational effect of mutations at a given position
        Args:
            seq: The wild type sequence

            pos: The position of the mutation

        Returns:
            The predicted mutational effect
        """
        tokens = self.tokenizer.tokenize(seq)
        ori_aa = tokens[pos - 1][0]
        tokens[pos - 1] = "#" + tokens[pos - 1][-1]
        
        mask_seq = " ".join(tokens)
        inputs = self.tokenizer(mask_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(dim=-1)[0, pos]
            
            scores = {}
            ori_st = self.tokenizer.get_vocab()[ori_aa + foldseek_struc_vocab[0]]
            for mut_aa in aa_list:
                mut_st = self.tokenizer.get_vocab()[mut_aa + foldseek_struc_vocab[0]]
                
                ori_prob = probs[ori_st: ori_st + len(foldseek_struc_vocab)].sum()
                mut_prob = probs[mut_st: mut_st + len(foldseek_struc_vocab)].sum()
                
                score = torch.log(mut_prob / ori_prob)
                scores[f"{ori_aa}{pos}{mut_aa}"] = score.item()
        
        return scores
    
    def predict_pos_prob(self, seq: str, pos: int) -> dict:
        """
        Predict the probability of all amino acids at a given position
        Args:
            seq: The wild type sequence

            pos: The position of the mutation

        Returns:
            The predicted probability of all amino acids
        """
        tokens = self.tokenizer.tokenize(seq)
        tokens[pos - 1] = "#" + tokens[pos - 1][-1]
        
        mask_seq = " ".join(tokens)
        inputs = self.tokenizer(mask_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(dim=-1)[0, pos]
            
            scores = {}
            for aa in aa_list:
                st = self.tokenizer.get_vocab()[aa + foldseek_struc_vocab[0]]
                prob = probs[st: st + len(foldseek_struc_vocab)].sum()
                
                scores[aa] = prob.item()
        
        return scores