import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import precision
from models import AlbuNet
from utils import crf

input_dir = "/storage/kaggle/tgs"
output_dir = "/artifacts"
img_size_ori = 101
img_size_target = 128
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.image_transform = transforms.Compose([
            prepare_input,
            transforms.ToTensor(),
            lambda t: t.type(torch.FloatTensor)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.image_transform(self.images[index])


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def load_image(path, id):
    image = np.array(Image.open("{}/{}.png".format(path, id)))
    return np.squeeze(image[:, :, 0:1]) / 255 if len(image.shape) == 3 else image / 65535


def load_images(path, ids):
    return [load_image(path, id) for id in tqdm(ids)]


def upsample(img):
    if img_size_target >= 2 * img.shape[0]:
        return upsample(
            np.pad(np.pad(img, ((0, 0), (0, img.shape[0])), "reflect"), ((0, img.shape[0]), (0, 0)), "reflect"))
    p = (img_size_target - img.shape[0]) / 2
    return np.pad(img, (int(np.ceil(p)), int(np.floor(p))), mode='reflect')


def downsample(img):
    if img.shape[0] >= 2 * img_size_ori:
        p = (img.shape[0] - 2 * img_size_ori) / 2
    else:
        p = (img.shape[0] - img_size_ori) / 2
    s = int(np.ceil(p))
    e = s + img_size_ori
    unpadded = img[s:e, s:e]
    if img.shape[0] >= 2 * img_size_ori:
        return unpadded[0:img_size_ori, 0:img_size_ori]
    else:
        return unpadded


def coverage_to_class(coverage):
    for i in range(0, 11):
        if coverage * 10 <= i:
            return i


def prepare_input(image):
    return np.expand_dims(upsample(image), axis=2).repeat(3, axis=2)


def contour(mask, width=3):
    edge_x = ndimage.convolve(mask, np.array([[-1, 0, +1], [-1, 0, +1], [-1, 0, +1]]))
    edge_y = ndimage.convolve(mask, np.array([[-1, -1, -1], [0, 0, 0], [+1, +1, +1]]))
    contour = np.abs(edge_x) + np.abs(edge_y)

    for _ in range(width - 1):
        contour = ndimage.convolve(contour, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    return np.int32(contour != 0)


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


def calculate_precision_based_on_contour(prediction, mask, prediction_contour, thresholds):
    isects = [(contour(np.int32(prediction > t)) * prediction_contour).sum() for t in thresholds]
    best_isect_index = np.argmax(isects)
    best_threshold = thresholds[best_isect_index]
    return precision(np.int32(prediction > best_threshold), mask)


def analyze(model, data_loader, val_set_df):
    val_set_df["predictions"] = predict(model, data_loader)

    thresholds = np.linspace(0, 1, 51)

    precisions_per_threshold = []
    for threshold in tqdm(thresholds, desc="Calculate optimal threshold"):
        precisions = []
        for idx in val_set_df.index:
            mask = val_set_df.loc[idx].masks
            prediction = val_set_df.loc[idx].predictions
            prediction_mask = np.int32(prediction > threshold)
            precisions.append(precision(prediction_mask, mask))
        precisions_per_threshold.append(np.mean(precisions))

    threshold_best = thresholds[np.argmax(precisions_per_threshold)]

    val_set_df["prediction_masks"] = [np.int32(p > threshold_best) for p in val_set_df.predictions]
    val_set_df["precisions"] = [precision(pm, m) for pm, m in zip(val_set_df.prediction_masks, val_set_df.masks)]

    val_set_df["prediction_masks_otsu"] = [np.int32(compute_otsu_mask(p)) for p in val_set_df.predictions]
    val_set_df["precisions_otsu"] = [precision(pm, m) for pm, m in
                                     zip(val_set_df.prediction_masks_otsu, val_set_df.masks)]

    val_set_df["prediction_masks_crf"] = [crf(i, pm) for i, pm in zip(val_set_df.images, val_set_df.prediction_masks)]
    val_set_df["precisions_crf"] = [precision(pm, m) for pm, m in
                                    zip(val_set_df.prediction_masks_crf, val_set_df.masks)]

    val_set_df["predictions_confidence"] = [(p * pm).sum() / pm.sum() for p, pm in
                                            zip(val_set_df.predictions, val_set_df.prediction_masks)]

    print()
    print("threshold: %.3f, precision: %.3f, precision_crf: %.3f, precision_otsu: %.3f" % (
        threshold_best, val_set_df.precisions.mean(), val_set_df.precisions_crf.mean(),
        val_set_df.precisions_otsu.mean()))

    print()
    print(val_set_df
        .groupby("coverage_class")
        .agg({
        "precisions": "mean",
        "precisions_crf": "mean",
        "precisions_otsu": "mean"
    }))

    print()
    print(val_set_df.groupby("coverage_class").agg({"coverage_class": "count"}))

    return threshold_best


def main():
    train_df = pd.read_csv("{}/train.csv".format(input_dir), index_col="id", usecols=[0])
    depths_df = pd.read_csv("{}/depths.csv".format(input_dir), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    train_df["images"] = load_images("{}/train/images".format(input_dir), train_df.index)
    train_df["masks"] = load_images("{}/train/masks".format(input_dir), train_df.index)

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(coverage_to_class)

    train_df["contours"] = train_df.masks.map(contour)

    train_val_split = int(0.8 * len(train_df))
    val_set_ids = train_df.index.tolist()[train_val_split:]

    val_set_df = train_df[train_df.index.isin(val_set_ids)].copy()

    model = AlbuNet(pretrained=True, is_deconv=False).to(device)
    model.load_state_dict(torch.load("/storage/model.pth"))

    val_set = TrainDataset(val_set_df.images.tolist())
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    mask_threshold = analyze(model, val_data_loader, val_set_df)

    test_df["images"] = load_images("{}/test/images".format(input_dir), test_df.index)

    test_set = TrainDataset(test_df.images.tolist())
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    test_df["predictions"] = predict(model, test_data_loader)
    test_df["prediction_masks"] = [np.int32(p > mask_threshold) for p in test_df.predictions]

    pred_dict = {idx: RLenc(test_df.loc[idx].prediction_masks) for i, idx in tqdm(enumerate(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission.csv".format(output_dir))

    pred_dict = {idx: RLenc(test_df.loc[idx].prediction_masks_crf) for i, idx in tqdm(enumerate(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission_crf.csv".format(output_dir))


if __name__ == "__main__":
    main()
