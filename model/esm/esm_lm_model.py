import torch
import os
import torchmetrics

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import EsmBaseModel


@register_model
class EsmLMModel(EsmBaseModel):
    def __init__(self, **kwargs):
        super().__init__(task='lm', **kwargs)
    
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy(ignore_index=-1)}
    
    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
            
        outputs = self.model(**inputs)
        
        # Get hidden representations
        if "output_hidden_states" in inputs and inputs["output_hidden_states"]:
            input_ids = inputs["input_ids"]
            ends = (input_ids == 2).int()
            indices = ends.argmax(dim=-1)
            repr_list = []
            hidden_states = outputs["hidden_states"][-1]
            for i, idx in enumerate(indices):
                repr = hidden_states[i][1:idx].mean(dim=0)
                repr_list.append(repr)
            
            reprs = torch.stack(repr_list, dim=0)
            outputs["hidden_states"] = reprs
        
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
    
    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        
        print(log_dict)
        self.log_info(log_dict)
        
        self.reset_metrics("test")
    
    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()
        
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
