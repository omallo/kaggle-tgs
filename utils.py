from multiprocessing import Pool

import pandas as pd

from processing import rlenc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def moving_parameter_average(target_model, source_model, alpha):
    for param1, param2 in zip(target_model.parameters(), source_model.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def write_submission(df, mask_name, file_path):
    with Pool(16) as pool:
        rlenc_results = [r for r in pool.map(rlenc, df[mask_name])]
    pred_dict = {idx: r for idx, r in zip(df.index.values, rlenc_results)}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ["id"]
    sub.columns = ["rle_mask"]
    sub.to_csv(file_path)
