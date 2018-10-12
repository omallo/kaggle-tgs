import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        all_predictions = [torch.sigmoid(m(x)[0]) for m in self.models]
        sum_predictions = torch.zeros_like(all_predictions[0])
        for p in all_predictions:
            sum_predictions += p
        return sum_predictions / len(self.models)
