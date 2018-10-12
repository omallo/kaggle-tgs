import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        all_prediction_logits = [m(x) for m in self.models]
        sum_mask_predictions = torch.zeros_like(all_prediction_logits[0][0])
        sum_has_salt_predictions = torch.zeros_like(all_prediction_logits[0][1])
        for prediction_logits in all_prediction_logits:
            sum_mask_predictions += torch.sigmoid(prediction_logits[0])
            sum_has_salt_predictions += torch.sigmoid(prediction_logits[1])
        return sum_mask_predictions / len(self.models), sum_has_salt_predictions / len(self.models)
