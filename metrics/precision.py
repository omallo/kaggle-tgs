import numpy as np


def precision(outputs, labels):
    predictions = outputs.round()

    intersection = float((predictions * labels).sum())
    union = float(((predictions + labels) > 0).sum())

    if union == 0:
        return 1.0

    iou = intersection / union

    thresholds = np.arange(0.5, 1.0, 0.05)
    precision = (iou > thresholds).sum() / float(len(thresholds))

    return precision


def precision_batch(outputs, labels):
    batch_size = labels.shape[0]
    return [precision(outputs[batch], labels[batch]) for batch in range(batch_size)]
