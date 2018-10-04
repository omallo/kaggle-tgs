from multiprocessing import Pool

import pandas as pd
from torch import nn

from deeplab_resnet import DeepLabv3_plus
from drn_unet import UNetDrn
from models import UNetResNet
from processing import rlenc


def create_model(type, input_size, pretrained):
    if type == "unet_resnet":
        return UNetResNet(34, 1, input_size, num_filters=32, dropout_2d=0.2, pretrained=pretrained, is_deconv=False)
    elif type == "unet_drn":
        return UNetDrn(1, input_size, pretrained=pretrained)
    elif type == "deeplab":
        return DeepLabv3_plus(n_classes=1, pretrained=pretrained)
    else:
        raise Exception("Unsupported model type: '{}".format(type))


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
