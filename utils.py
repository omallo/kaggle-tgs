from multiprocessing import Pool

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch import nn

from processing import rlenc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def with_he_normal_weights(layer):
    nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
    return layer


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def write_submission(df, mask_name, file_path):
    with Pool(16) as pool:
        rlenc_results = [r for r in pool.map(rlenc, df[mask_name])]
    pred_dict = {idx: r for idx, r in zip(df.index.values, rlenc_results)}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ["id"]
    sub.columns = ["rle_mask"]
    sub.to_csv(file_path)


def kfold_split(n_splits, values, classes):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_value_indexes, test_value_indexes in skf.split(values, classes):
        train_values = [values[i] for i in train_value_indexes]
        test_values = [values[i] for i in test_value_indexes]
        yield train_values, test_values
