import torch
import torch.distributed as dist
import torchmetrics

from torch.nn.functional import cross_entropy

from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaProtIFModel(SaprotBaseModel):
    """
    SaProt inverse folding model.
    """
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)
    
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}
    
    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        outputs = self.model(**inputs)
        return outputs
    
    def loss_func(self, stage, outputs, labels):
        logits = outputs['logits']
        # merge the first and second dimension of logits
        logits = logits.view(-1, logits.size(-1))
        
        # flatten labels
        labels = labels['labels'].flatten().to(logits.device)
        
        loss = cross_entropy(logits, labels, ignore_index=-1)
        getattr(self, f"{stage}_acc").update(logits.detach(), labels)
        
        if stage == 'train':
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)
            self.reset_metrics("train")
        
        return loss
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        
        # Re-sample the subset of the training data
        if getattr(self.trainer.datamodule.train_dataset, "num_per_cluster", None) is not None:
            self.trainer.datamodule.train_dataset.sample_subset()
    
    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        
        if dist.get_rank() == 0:
            print(log_dict)
        self.log_info(log_dict)
        
        self.reset_metrics("test")
    
    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        
        if dist.get_rank() == 0:
            print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        
        valid_acc = log_dict["valid_acc"]
        self.check_save_condition(valid_acc, mode="max")
    
    def predict(self, aa_seq: str, struc_seq: str, method: str = "argmax", num_samples: int = 1) -> str:
        """
        Predict all masked amino acids in the sequence.
        Args:
            aa_seq: Amino acid sequence (could be all masked or partially masked).

            struc_seq: Foldseek sequence.

            method: Prediction method. It could be either "argmax" or "multinomial". If "argmax", the most probable
                    amino acid will be selected. If "multinomial", the amino acid will be sampled from the multinomial
                    distribution.
            
            num_samples: Number of predicted sequences. Only works when method is "multinomial".

        Returns:
            Predicted residue sequence.
        """
        assert len(aa_seq) == len(struc_seq), "The length of the amino acid sequence and the foldseek sequence must be the same."
        assert method in ["argmax", "multinomial"], "The prediction method must be either 'argmax' or 'multinomial'."
        if method == "argmax":
            assert num_samples == 1, "The sample number must be 1 when the prediction method is 'argmax'."
            
        sa_seq = "".join(f"{aa}{struc}" for aa, struc in zip(aa_seq, struc_seq))

        # Record the index of masked amino acids
        mask_indices = [i for i, aa in enumerate(aa_seq) if aa == '#']

        with torch.no_grad():
            inputs = self.tokenizer(sa_seq, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits[0, 1: -1]
            probs = torch.softmax(logits, dim=-1)

            # Predict amino acids
            if method == "argmax":
                batch_preds = probs.argmax(dim=-1).unsqueeze(0)
            else:
                batch_preds = torch.multinomial(probs, num_samples, replacement=True).permute(1, 0)
                
            pred_aa_seqs = []
            for preds in batch_preds:
                masked_preds = preds[mask_indices]
                pred_tokens = self.tokenizer.convert_ids_to_tokens(masked_preds)
    
                tokens = list(aa_seq)
                for i, pred_token in zip(mask_indices, pred_tokens):
                    tokens[i] = pred_token[0]
    
                pred_aa_seq = "".join(tokens)
                pred_aa_seqs.append(pred_aa_seq)
                
            return pred_aa_seqs
