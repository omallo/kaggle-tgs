import torch.nn as nn

from metrics.lovasz_losses import lovasz_hinge


class LovaszWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        return lovasz_hinge(logits, targets, per_image=False)
