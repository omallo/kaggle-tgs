import torch.nn as nn


class AggregateLoss(nn.Module):
    def __init__(self, *delegates):
        super().__init__()
        self.delegates = delegates

    def forward(self, input, targets):
        for delegate in self.delegates:
            delegate.weight = self.weight

        loss = self.delegates[0](input, targets)
        for delegate in self.delegates[1:]:
            loss += delegate(input, targets)

        return loss
