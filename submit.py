import cv2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainData, TrainDataset
from metrics import precision
from models import UNetResNet
from processing import crf, rlenc

input_dir = "/storage/kaggle/tgs"
output_dir = "/artifacts"
img_size_ori = 101
img_size_target = 128
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


def predict(model, data_loader):
    val_predictions = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            inputs = batch.to(device)
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)
            val_predictions += [p for p in predictions.cpu().numpy()]
    val_predictions = np.asarray(val_predictions).reshape(-1, img_size_target, img_size_target)
    val_predictions = [downsample(p) for p in val_predictions]
    return val_predictions


def calculate_best_threshold(df):
    thresholds = np.linspace(0, 1, 51)
    precisions_per_threshold = []
    for threshold in tqdm(thresholds, desc="Calculate optimal threshold"):
        precisions = []
        for idx in df.index:
            mask = df.loc[idx].masks
            prediction = df.loc[idx].predictions
            prediction_mask = np.int32(prediction > threshold)
            precisions.append(precision(prediction_mask, mask))
        precisions_per_threshold.append(np.mean(precisions))
    return thresholds[np.argmax(precisions_per_threshold)]


def analyze(model, data_loader, val_set_df):
    val_set_df["predictions"] = predict(model, data_loader)

    best_threshold = calculate_best_threshold(val_set_df)

    val_set_df["prediction_masks"] = [np.int32(p > best_threshold) for p in val_set_df.predictions]
    val_set_df["precisions"] = [precision(pm, m) for pm, m in zip(val_set_df.prediction_masks, val_set_df.masks)]

    val_set_df["prediction_coverage"] = val_set_df.prediction_masks.map(np.sum) / pow(img_size_ori, 2)
    val_set_df["prediction_coverage_class"] = val_set_df.prediction_coverage.map(coverage_to_class)

    val_set_df["prediction_masks_otsu"] = [np.int32(compute_otsu_mask(p)) for p in val_set_df.predictions]
    val_set_df["precisions_otsu"] = [precision(pm, m) for pm, m in
                                     zip(val_set_df.prediction_masks_otsu, val_set_df.masks)]

    val_set_df["prediction_masks_crf"] = [crf(i, pm) for i, pm in zip(val_set_df.images, val_set_df.prediction_masks)]
    val_set_df["precisions_crf"] = [precision(pm, m) for pm, m in
                                    zip(val_set_df.prediction_masks_crf, val_set_df.masks)]

    val_set_df["precisions_max"] = [max(p1, p2, p3) for p1, p2, p3 in
                                    zip(val_set_df.precisions, val_set_df.precisions_otsu, val_set_df.precisions_crf)]

    val_set_df["prediction_masks_avg"] = [np.int32((p1 + p2 + p3) >= 2) for p1, p2, p3 in
                                          zip(val_set_df.prediction_masks, val_set_df.prediction_masks_otsu,
                                              val_set_df.prediction_masks_crf)]
    val_set_df["precisions_avg"] = [precision(pm, m) for pm, m in
                                    zip(val_set_df.prediction_masks_avg, val_set_df.masks)]

    best_thresholds_per_coverage_class = {}
    for cc, cc_df in val_set_df.groupby("prediction_coverage_class"):
        best_thresholds_per_coverage_class[cc] = calculate_best_threshold(cc_df)

    val_set_df["precisions_cc"] = [precision(np.int32(p > best_thresholds_per_coverage_class[cc]), m) for p, m, cc in
                                   zip(val_set_df.predictions, val_set_df.masks, val_set_df.prediction_coverage_class)]

    print()
    print(
        "threshold: %.3f, precision: %.3f, precision_crf: %.3f, precision_otsu: %.3f, precision_max: %.3f, precision_avg: %.3f, precision_cc: %.3f" % (
            best_threshold, val_set_df.precisions.mean(), val_set_df.precisions_crf.mean(),
            val_set_df.precisions_otsu.mean(), val_set_df.precisions_max.mean(), val_set_df.precisions_avg.mean(),
            val_set_df.precisions_cc.mean()))

    print()
    print(val_set_df
        .groupby("coverage_class")
        .agg({
        "precisions": "mean",
        "precisions_crf": "mean",
        "precisions_otsu": "mean",
        "precisions_max": "mean",
        "precisions_avg": "mean",
        "precisions_cc": "mean",
        "coverage_class": "count"
    }))

    print()
    print(val_set_df
        .groupby("prediction_coverage_class")
        .agg({
        "precisions": "mean",
        "precisions_crf": "mean",
        "precisions_otsu": "mean",
        "precisions_max": "mean",
        "precisions_avg": "mean",
        "precisions_cc": "mean",
        "prediction_coverage_class": "count"
    }))

    return best_threshold


def main():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 160)

    input_dir = "/storage/kaggle/tgs"
    output_dir = "/artifacts"
    image_size_target = 128

    train_data = TrainData(input_dir)

    train_set = TrainDataset(train_data.train_set_df, image_size_target, augment=True)
    train_set_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    val_set = TrainDataset(train_data.val_set_df, image_size_target, augment=False)
    val_set_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNetResNet(101, 1, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False).to(device)
    model.load_state_dict(torch.load("/storage/model.pth", map_location=device))

    mask_threshold = analyze(model, val_data_loader, val_set_df)

    test_df["images"] = load_images("{}/test/images".format(input_dir), test_df.index)

    test_set = TrainDataset(test_df.images.tolist())
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    test_df["predictions"] = predict(model, test_data_loader)
    test_df["prediction_masks"] = [np.int32(p > mask_threshold) for p in test_df.predictions]
    test_df["prediction_masks_crf"] = [crf(i, pm) for i, pm in zip(test_df.images, test_df.prediction_masks)]

    pred_dict = {idx: rlenc(test_df.loc[idx].prediction_masks) for i, idx in tqdm(enumerate(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission.csv".format(output_dir))

    pred_dict = {idx: rlenc(test_df.loc[idx].prediction_masks_crf) for i, idx in tqdm(enumerate(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission_crf.csv".format(output_dir))


if __name__ == "__main__":
    main()
