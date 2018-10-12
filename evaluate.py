import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import calculate_coverage_class, TestDataset
from metrics import precision
from processing import crf_batch
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


def predict(model, data_loader, use_tta):
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for image in data_loader:
            image = image.to(device)

            if use_tta:
                predictions1, _ = model(image)
                predictions2, _ = model(image.flip(3))
                predictions2 = predictions2.flip(3)
                predictions = 0.5 * (predictions1 + predictions2)
            else:
                predictions, _ = model(image)

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


def calculate_best_threshold_per_cc(df):
    mask_threshold_per_cc = {}
    for cc, cc_df in df.groupby("predictions_cc"):
        mask_threshold_per_cc[cc] = calculate_best_threshold(cc_df)
    return mask_threshold_per_cc


def calculate_predictions_cc(df, threshold):
    df["predictions_cc"] = [calculate_coverage_class(np.int32(p > threshold)) for p in df.predictions]


def calculate_predictions(df, model, use_tta):
    data_set = TestDataset(df, image_size_target)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=4)
    df["predictions"] = predict(model, data_loader, use_tta)


def calculate_prediction_masks(df, threshold, threshold_per_cc):
    df["prediction_masks"] = [np.int32(p > threshold) for p in df.predictions]
    df["prediction_masks_otsu"] = [np.int32(compute_otsu_mask(p)) for p in df.predictions]
    df["prediction_masks_crf"] = crf_batch(df.images, df.prediction_masks)
    df["prediction_masks_cc"] = [np.int32(p > threshold_per_cc[cc]) for p, cc in zip(df.predictions, df.predictions_cc)]


def calculate_precisions(df):
    df["precisions"] = [precision(pm, m) for pm, m in zip(df.prediction_masks, df.masks)]
    df["precisions_otsu"] = [precision(pm, m) for pm, m in zip(df.prediction_masks_otsu, df.masks)]
    df["precisions_crf"] = [precision(pm, m) for pm, m in zip(df.prediction_masks_crf, df.masks)]
    df["precisions_cc"] = [precision(pm, m) for pm, m in zip(df.prediction_masks_cc, df.masks)]


def calculate_best_mask_per_cc(df):
    best_mask_per_cc = {}

    for cc, df_cc in df.groupby("predictions_cc"):
        precision = np.mean(df_cc.precisions)
        precision_otsu = np.mean(df_cc.precisions_otsu)
        precision_crf = np.mean(df_cc.precisions_crf)

        if precision_otsu >= precision and precision_otsu >= precision_crf:
            best_mask_per_cc[cc] = "prediction_masks_otsu"
        elif precision >= precision_otsu and precision >= precision_crf:
            best_mask_per_cc[cc] = "prediction_masks"
        else:
            best_mask_per_cc[cc] = "prediction_masks_crf"

    return best_mask_per_cc


def calculate_best_prediction_masks(df, best_mask_per_cc):
    df["prediction_masks_best"] = [df.loc[idx][best_mask_per_cc[df.loc[idx].predictions_cc]] for idx in df.index]


def analyze(model, df, use_tta):
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 160)

    calculate_predictions(df, model, use_tta)

    mask_threshold_global = calculate_best_threshold(df)
    calculate_predictions_cc(df, mask_threshold_global)
    mask_threshold_per_cc = calculate_best_threshold_per_cc(df)
    calculate_prediction_masks(df, mask_threshold_global, mask_threshold_per_cc)
    calculate_precisions(df)

    best_mask_per_cc = calculate_best_mask_per_cc(df)
    calculate_best_prediction_masks(df, best_mask_per_cc)
    df["precisions_best"] = [precision(pm, m) for pm, m in zip(df.prediction_masks_best, df.masks)]

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

    print()
    print("best masks per prediction coverage class:")
    for cc, best_mask in best_mask_per_cc.items():
        print("  {}: {}".format(cc, best_mask))

    return mask_threshold_global, mask_threshold_per_cc, best_mask_per_cc
