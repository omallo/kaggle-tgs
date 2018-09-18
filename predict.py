import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import precision_batch, precision
from models import ResNetUNet
from models import AlbuNet
from utils import crf

input_dir = "/storage/kaggle/tgs"
output_dir = "/artifacts"
img_size_ori = 101
img_size_target = 128
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainDataset(Dataset):
    def __init__(self, images, masks, augment):
        super().__init__()
        self.images = images
        self.masks = masks
        self.augment = augment
        self.image_transform = transforms.Compose([
            prepare_input,
            transforms.ToTensor(),
            lambda t: t.type(torch.FloatTensor)
        ])
        self.mask_transform = transforms.Compose([
            prepare_label,
            transforms.ToTensor(),
            lambda t: t.type(torch.FloatTensor)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if self.augment:
            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if np.random.rand() < 0.5:
                c = np.random.choice(3)
                if c == 0:
                    image, mask = apply_elastic_transform(image, mask, alpha=150, sigma=8, alpha_affine=0)
                elif c == 1:
                    image, mask = apply_elastic_transform(image, mask, alpha=0, sigma=0, alpha_affine=8)
                elif c == 2:
                    image, mask = apply_elastic_transform(image, mask, alpha=150, sigma=10, alpha_affine=5)

            if np.random.rand() < 0.5:
                c = np.random.choice(2)
                if c == 0:
                    image = multiply_brightness(image, np.random.uniform(1 - 0.1, 1 + 0.1))
                elif c == 1:
                    image = adjust_gamma(image, np.random.uniform(1 - 0.1, 1 + 0.1))

        mask_weights = calculate_mask_weights(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask_weights = self.mask_transform(mask_weights)

        return image, mask, mask_weights

        is_blurry = cv2.Laplacian(image, cv2.CV_32F).var() < 0.001
        if is_blurry:
            if np.random.rand() < 0.5:
                blurr_filter = ndimage.gaussian_filter(image, 1)
                alpha = 30
                image = image + alpha * (image - blurr_filter)


def multiply_brightness(image, coefficient):
    i = np.expand_dims(image, axis=2)
    i = i.repeat(3, axis=2)
    i = (255 * i).astype("uint8")
    image_HLS = cv2.cvtColor(i, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * coefficient
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    result = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    result = result[:, :, 0:1].squeeze()
    result = (result / 255.0).astype(image.dtype)
    return result


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    result = cv2.LUT((255 * image).astype("uint8"), table)
    return (result / 255.0).astype(image.dtype)


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def apply_elastic_transform(image, mask, alpha, sigma, alpha_affine):
    channels = np.concatenate((image[..., None], mask[..., None]), axis=2)
    result = elastic_transform(channels, alpha, sigma, alpha_affine, random_state=np.random.RandomState(None))
    image_result = result[..., 0]
    mask_result = result[..., 1]
    mask_result = (mask_result > 0.5).astype(mask.dtype)
    return image_result, mask_result


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


def prepare_label(mask):
    return np.expand_dims(upsample(mask), axis=2)


def contour(mask, width=3):
    edge_x = ndimage.convolve(mask, np.array([[-1, 0, +1], [-1, 0, +1], [-1, 0, +1]]))
    edge_y = ndimage.convolve(mask, np.array([[-1, -1, -1], [0, 0, 0], [+1, +1, +1]]))
    contour = np.abs(edge_x) + np.abs(edge_y)

    for _ in range(width - 1):
        contour = ndimage.convolve(contour, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    return np.int32(contour != 0)


def calculate_mask_weights(mask):
    return np.ones_like(mask) + 2 * contour(mask)


# https://www.microsoft.com/developerblog/2018/05/17/using-otsus-method-generate-data-training-deep-learning-image-segmentation-models/
def compute_otsu_mask(image):
    image = 255 * image
    image = np.stack((image,) * 3, -1)
    image = image.astype(np.uint8)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] / 255


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def moving_average(net1, net2, alpha):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def eval(model, data_loader, criterion):
    loss_sum = 0.0
    precision_sum = 0.0
    step_count = 0

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            inputs, labels, label_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)
            # TODO: add again
            # criterion.weight = label_weights
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            precision_sum += np.mean(precision_batch(predictions, labels))
            step_count += 1

    loss_avg = loss_sum / step_count
    precision_avg = precision_sum / step_count

    return loss_avg, precision_avg


def predict(model, val_pred_loader):
    val_predictions = []
    with torch.no_grad():
        for _, batch in enumerate(val_pred_loader):
            inputs = batch[0].to(device)
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


def analyze(mask_model, mask_data_loader, contour_model, contour_data_loader, val_set_df):
    val_set_df["predictions"] = predict(mask_model, mask_data_loader)
    val_set_df["predictions_contours"] = predict(contour_model, contour_data_loader)

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

    val_set_df["precisions_contour"] = [calculate_precision_based_on_contour(p, m, pc, thresholds) for p, m, pc in
                                        zip(val_set_df.predictions, val_set_df.masks, val_set_df.predictions_contours)]

    val_set_df["prediction_masks_crf"] = [crf(i, pm) for i, pm in zip(val_set_df.images, val_set_df.prediction_masks)]
    val_set_df["precisions_crf"] = [precision(pm, m) for pm, m in
                                    zip(val_set_df.prediction_masks_crf, val_set_df.masks)]

    val_set_df["predictions_confidence"] = [(p * pm).sum() / pm.sum() for p, pm in
                                            zip(val_set_df.predictions, val_set_df.prediction_masks)]

    print()
    print("threshold: %.3f, precision: %.3f, precision_crf: %.3f, precision_otsu: %.3f, precision_contour: %.3f" % (
        threshold_best, val_set_df.precisions.mean(), val_set_df.precisions_crf.mean(),
        val_set_df.precisions_otsu.mean(), val_set_df.precisions_contour.mean()))

    print()
    print(val_set_df
        .groupby("coverage_class")
        .agg({
        "precisions": "mean",
        "precisions_crf": "mean",
        "precisions_otsu": "mean",
        "precisions_contour": "mean",
        "predictions_confidence": "mean",
        "coverage_class": "count"
    }))

    print()
    print(val_set_df.predictions_confidence.describe())

    print()
    print("prediction-confidence correlation: %.3f" % val_set_df.precisions.corr(val_set_df.predictions_confidence))


def main():
    train_df = pd.read_csv("{}/train.csv".format(input_dir), index_col="id", usecols=[0])
    depths_df = pd.read_csv("{}/depths.csv".format(input_dir), index_col="id")
    train_df = train_df.join(depths_df)

    train_df["images"] = load_images("{}/train/images".format(input_dir), train_df.index)
    train_df["masks"] = load_images("{}/train/masks".format(input_dir), train_df.index)

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(coverage_to_class)

    train_df["contours"] = train_df.masks.map(contour)
    train_df["mask_weights"] = [calculate_mask_weights(m) for m, c in zip(train_df.masks, train_df.coverage_class)]

    train_val_split = int(0.8 * len(train_df))
    val_set_ids = train_df.index.tolist()[train_val_split:]

    val_set_df = train_df[train_df.index.isin(val_set_ids)].copy()

    mask_model = AlbuNet(pretrained=True).to(device)
    mask_model.load_state_dict(torch.load("/storage/masks.pth"))

    contour_model = ResNetUNet(n_class=1).to(device)
    contour_model.load_state_dict(torch.load("/storage/contours.pth"))

    mask_set = TrainDataset(val_set_df.images.tolist(), val_set_df.masks.tolist(), augment=False)
    mask_data_loader = DataLoader(mask_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    contour_set = TrainDataset(val_set_df.images.tolist(), val_set_df.contours.tolist(), augment=False)
    contour_data_loader = DataLoader(contour_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    analyze(mask_model, mask_data_loader, contour_model, contour_data_loader, val_set_df)


if __name__ == "__main__":
    main()