import torch.nn as nn


class AggregateLoss(nn.Module):
    def __init__(self, criterions, criterion_weights):
        super().__init__()
        self.criterions = criterions
        self.criterion_weights = criterion_weights

    def forward(self, input, targets):
        for criterion in self.criterions:
            criterion.weight = self.weight

        loss = 0.0
        for criterion, criterion_weight in zip(self.criterions, self.criterion_weights):
            loss += criterion_weight * criterion(input, targets)

        return loss
