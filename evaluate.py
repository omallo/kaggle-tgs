import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import calculate_coverage_class, TestDataset
from metrics import precision
from processing import crf
from transforms import downsample

image_size_original = 101
image_size_target = 128
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


# https://www.microsoft.com/developerblog/2018/05/17/using-otsus-method-generate-data-training-deep-learning-image-segmentation-models/
def compute_otsu_mask(image):
    image = 255 * image
    image = np.stack((image,) * 3, -1)
    image = image.astype(np.uint8)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] / 255


def predict_image_over_4_crops(image, model):
    b, _, h, w = image.shape

    predictions = torch.zeros((b, 1, h, w), dtype=image.dtype, layout=image.layout, device=image.device)
    weights = torch.zeros_like(predictions)

    predictions[:, :, 0:96, 0:96] += model(image[:, :, 0:96, 0:96].contiguous())
    weights[:, :, 0:96, 0:96] += 1

    predictions[:, :, 5:101, 0:96] += model(image[:, :, 5:101, 0:96].contiguous())
    weights[:, :, 5:101, 0:96] += 1

    predictions[:, :, 0:96, 5:101] += model(image[:, :, 0:96, 5:101].contiguous())
    weights[:, :, 0:96, 5:101] += 1

    predictions[:, :, 5:101, 5:101] += model(image[:, :, 5:101, 5:101].contiguous())
    weights[:, :, 5:101, 5:101] += 1

    return predictions / weights


def predict(model, data_loader, use_tta):
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for _, image in enumerate(data_loader):
            image = image.to(device)

            if use_tta:
                predictions1 = model(image)
                predictions2 = model(image.flip(3)).flip(3)
                predictions = 0.5 * (predictions1 + predictions2)
            else:
                predictions = model(image)

            val_predictions += [p for p in predictions.cpu().numpy()]
    val_predictions = np.asarray(val_predictions).reshape(-1, image_size_target, image_size_target)
    val_predictions = [downsample(p, image_size_original) for p in val_predictions]
    return val_predictions


def calculate_best_threshold(df):
    thresholds = np.linspace(0, 1, 51)
    precisions_per_threshold = []
    for threshold in thresholds:
        precisions = []
        for idx in df.index:
            mask = df.loc[idx].masks
            prediction = df.loc[idx].predictions
            prediction_mask = np.int32(prediction > threshold)
            precisions.append(precision(prediction_mask, mask))
        precisions_per_threshold.append(np.mean(precisions))
    return thresholds[np.argmax(precisions_per_threshold)]


def calculate_best_precision(precision, precision_otsu, precision_crf, prediction_cc):
    if prediction_cc == 0 or prediction_cc == 9:
        return precision
    elif prediction_cc == 10:
        return precision_crf
    else:
        return precision_otsu


def analyze(model, df, use_tta):
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 160)

    data_set = TestDataset(df, image_size_target)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=4)

    df["predictions"] = predict(model, data_loader, use_tta)

    mask_threshold_global = calculate_best_threshold(df)

    df["prediction_masks"] = [np.int32(p > mask_threshold_global) for p in df.predictions]
    df["precisions"] = [precision(pm, m) for pm, m in zip(df.prediction_masks, df.masks)]

    df["predictions_cc"] = df.prediction_masks.map(calculate_coverage_class)

    df["prediction_masks_otsu"] = [np.int32(compute_otsu_mask(p)) for p in df.predictions]
    df["precisions_otsu"] = [precision(pm, m) for pm, m in zip(df.prediction_masks_otsu, df.masks)]

    df["prediction_masks_crf"] = [crf(i, pm) for i, pm in zip(df.images, df.prediction_masks)]
    df["precisions_crf"] = [precision(pm, m) for pm, m in zip(df.prediction_masks_crf, df.masks)]

    df["precisions_best"] = [calculate_best_precision(p1, p2, p3, cc) for p1, p2, p3, cc in
                             zip(df.precisions, df.precisions_otsu, df.precisions_crf, df.predictions_cc)]

    mask_threshold_per_cc = {}
    for cc, cc_df in df.groupby("predictions_cc"):
        mask_threshold_per_cc[cc] = calculate_best_threshold(cc_df)

    df["precisions_cc"] = [precision(np.int32(p > mask_threshold_per_cc[cc]), m) for p, m, cc in
                           zip(df.predictions, df.masks, df.predictions_cc)]

    print()
    print(
        "threshold: %.3f, precision: %.3f, precision_crf: %.3f, precision_otsu: %.3f, precision_best: %.3f, precision_cc: %.3f" % (
            mask_threshold_global,
            df.precisions.mean(),
            df.precisions_crf.mean(),
            df.precisions_otsu.mean(),
            df.precisions_best.mean(),
            df.precisions_cc.mean()))

    print()
    print(df
        .groupby("coverage_class")
        .agg({
        "precisions": "mean",
        "precisions_crf": "mean",
        "precisions_otsu": "mean",
        "precisions_best": "mean",
        "precisions_cc": "mean",
        "coverage_class": "count"
    }))

    print()
    print(df
        .groupby("predictions_cc")
        .agg({
        "precisions": "mean",
        "precisions_crf": "mean",
        "precisions_otsu": "mean",
        "precisions_best": "mean",
        "precisions_cc": "mean",
        "predictions_cc": "count"
    }))

    return mask_threshold_global, mask_threshold_per_cc
