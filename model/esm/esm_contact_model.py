import torchmetrics
import torch
import numpy as np
import math

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import EsmBaseModel


@register_model
class EsmContactModel(EsmBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for EsmBaseModel
        """
        super().__init__(task="base", **kwargs)

    def initialize_model(self):
        super().initialize_model()

        # hidden_size = self.model.config.hidden_size * 2
        hidden_size = self.model.config.num_attention_heads
        # hidden_size = self.model.config.num_hidden_layers * self.model.config.num_attention_heads
        
        classifier = torch.nn.Sequential(
            # Linear(hidden_size, hidden_size),
            # ReLU(),
            Linear(hidden_size, 2)
        )
        
        # Freeze all parameters except classifier
        for param in self.model.parameters():
            param.requires_grad = False

        setattr(self.model, "classifier", classifier)

    def initialize_metrics(self, stage):
        metric_dict = {}
        for length in ["P@L", "P@L/2", "P@L/5"]:
            for range in ["short_range", "medium_range", "long_range"]:
                metric_dict[f"{stage}_{range}_{length}"] = torchmetrics.Accuracy(ignore_index=-1)

        return metric_dict

    def forward(self, inputs):
        # inputs["output_hidden_states"] = True
        # outputs = self.model.esm(**inputs)
        #
        # hidden_states = outputs["hidden_states"][-1]
        # prod = hidden_states[:, :, None, :] * hidden_states[:, None, :, :]
        # diff = hidden_states[:, :, None, :] - hidden_states[:, None, :, :]
        # pairwise_features = torch.cat((prod, diff), -1)
        # pairwise_features = (pairwise_features + pairwise_features.transpose(1, 2)) / 2
        #
        # logits = self.model.classifier(pairwise_features)
        # logits = logits[:, 1: -1, 1: -1].contiguous()
        #
        # return logits

        inputs["output_attentions"] = True
        outputs = self.model.esm(**inputs)

        attention_maps = torch.cat(outputs["attentions"][-1:], 1).permute(0, 2, 3, 1)
        # attention_maps = torch.cat(outputs["attentions"], 1).permute(0, 2, 3, 1)
        attention_maps = (attention_maps + attention_maps.transpose(1, 2)) / 2
        logits = self.model.classifier(attention_maps)
        logits = logits[:, 1: -1, 1: -1].contiguous()

        return logits

    def loss_func(self, stage, logits, labels):
        lengths = labels["lengths"]
        targets = labels["targets"].to(logits.device)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.flatten(), ignore_index=-1)

        # Iterate through all proteins and count accuracy
        length_dict = {"P@L": 1, "P@L/2": 2, "P@L/5": 5}
        range_dict = ["short_range", "medium_range", "long_range"]
        for pred_map, label_map, L in zip(logits.detach(), targets, lengths):
            x_inds, y_inds = np.indices(label_map.shape)
            for r in range_dict:
                if r == "short_range":
                    mask = (np.abs(y_inds - x_inds) < 6) | (np.abs(y_inds - x_inds) > 11)

                elif r == "medium_range":
                    mask = (np.abs(y_inds - x_inds) < 12) | (np.abs(y_inds - x_inds) > 23)

                else:
                    mask = np.abs(y_inds - x_inds) < 24

                mask = torch.from_numpy(mask)
                copy_label_map = label_map.clone()
                copy_label_map[mask] = -1
                
                # Mask the lower triangle
                mask = torch.triu(torch.ones_like(copy_label_map), diagonal=1)
                copy_label_map[mask == 0] = -1

                selector = copy_label_map != -1
                preds = pred_map[selector].float()
                labels = copy_label_map[selector]

                probs = preds.softmax(dim=-1)[:, 1]
                for k, v in length_dict.items():
                    l = min(math.ceil(L / v), (labels == 1).sum().item())

                    top_inds = torch.argsort(probs, descending=True)[:l]
                    top_labels = labels[top_inds]

                    if top_labels.numel() == 0:
                        continue

                    metric = f"{stage}_{r}_{k}"
                    self.metrics[stage][metric].update(top_labels, torch.ones_like(top_labels))

        if stage == "train":
            # log_dict = self.get_log_dict("train")
            # log_dict["train_loss"] = loss
            # self.log_info(log_dict)

            # Reset train metrics
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
        self.check_save_condition(log_dict["valid_medium_range_P@L/5"], mode="max")
