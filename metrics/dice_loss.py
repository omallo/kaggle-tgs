import torch
import torch.nn as nn


class DiceWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        w = self.weight.view(num, -1) if self.weight is not None else torch.ones_like(m1)
        w2 = w * w

        score = 2. * ((w2 * intersection).sum(1) + smooth) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + smooth)
        loss = 1 - score.sum() / num

        return loss
