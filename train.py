import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from metrics.precision import precision_batch
from models.unet import UNet

input_dir = "/storage/kaggle/tgs"
output_dir = "/artifacts"
img_size_ori = 101
img_size_target = 128
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TgsDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)

    def __getitem__(self, index):
        item = super().__getitem__(index)

        if np.random.rand() < 0.5:
            item = tuple(i.flip(dims=(0, 2, 1)) for i in item)

        image_np = item[0].cpu().data.numpy()[0:1, :, :].squeeze()
        is_blurry = cv2.Laplacian(image_np, cv2.CV_32F).var() < 0.001
        if is_blurry:
            if np.random.rand() < 0.5:
                blurr_filter = ndimage.gaussian_filter(image_np, 1)
                alpha = 30
                sharpened = image_np + alpha * (image_np - blurr_filter)
                sharpened = sharpened.reshape(1, image_np.shape[0], image_np.shape[1]).repeat(3, axis=0)
                item_list = list(item)
                item_list[0] = torch.FloatTensor(sharpened).to(item[0].device)
                item = tuple(item_list)

        return item


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


def prepare_input(image, transform):
    return transform(np.expand_dims(upsample(image), axis=2).repeat(3, axis=2))


def prepare_inputs(images, transform):
    return torch.stack([prepare_input(image, transform) for image in images]) \
        .type(torch.FloatTensor)


def prepare_label(mask, transform):
    return transform(np.expand_dims(upsample(mask), axis=2))


def prepare_labels(masks, transform):
    return torch.stack([prepare_label(mask, transform) for mask in masks]) \
        .type(torch.FloatTensor)


def contour(mask, width=3):
    edge_x = ndimage.convolve(mask, np.array([[-1, 0, +1], [-1, 0, +1], [-1, 0, +1]]))
    edge_y = ndimage.convolve(mask, np.array([[-1, -1, -1], [0, 0, 0], [+1, +1, +1]]))
    contour = np.abs(edge_x) + np.abs(edge_y)

    for _ in range(width - 1):
        contour = ndimage.convolve(contour, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    return np.int32(contour != 0)


def mask_weights(mask, coverage_class):
    coverage_class_weight_factor = 1 if coverage_class <= 3 else 1
    return coverage_class_weight_factor * (np.ones_like(mask) + 2 * contour(mask))


# https://www.microsoft.com/developerblog/2018/05/17/using-otsus-method-generate-data-training-deep-learning-image-segmentation-models/
def compute_otsu_mask(image):
    image = np.stack((image,) * 3, -1)
    image = image.astype(np.uint8)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


train_df = pd.read_csv("{}/train.csv".format(input_dir), index_col="id", usecols=[0])
depths_df = pd.read_csv("{}/depths.csv".format(input_dir), index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = load_images("{}/train/images".format(input_dir), train_df.index)
train_df["masks"] = load_images("{}/train/masks".format(input_dir), train_df.index)

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(coverage_to_class)

train_df["contours"] = train_df.masks.map(contour)
train_df["mask_weights"] = [mask_weights(m, c) for m, c in zip(train_df.masks, train_df.coverage_class)]

train_val_split = int(0.8 * len(train_df))
train_set_ids = train_df.index.tolist()[:train_val_split]
val_set_ids = train_df.index.tolist()[train_val_split:]

train_set_df = train_df[train_df.index.isin(train_set_ids)].copy()
val_set_df = train_df[train_df.index.isin(val_set_ids)].copy()

train_set_x = train_set_df.images.tolist()
train_set_y = train_set_df.masks.tolist()
train_set_w = train_set_df.mask_weights.tolist()

val_set_x = val_set_df.images.tolist()
val_set_y = val_set_df.masks.tolist()
val_set_w = val_set_df.mask_weights.tolist()

input_transform = transforms.Compose([
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.ToTensor()
])

# model = FusionNet(in_depth=3, out_depth=1, base_channels=32).to(device)
model = UNet(in_depth=3, out_depth=1, base_channels=32).to(device)
# model = AlbuNet(pretrained=True).to(device)

# model.load_state_dict(torch.load("{}/model.pth".format(output_dir)))

criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():
    train_set_inputs = prepare_inputs(train_set_x, input_transform)
    train_set_labels = prepare_labels(train_set_y, label_transform)
    train_set_weights = prepare_labels(train_set_w, label_transform)

    val_set_inputs = prepare_inputs(val_set_x, input_transform)
    val_set_labels = prepare_labels(val_set_y, label_transform)
    val_set_weights = prepare_labels(val_set_w, label_transform)

train_set = TgsDataset(train_set_inputs, train_set_labels, train_set_weights)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)

val_set = TensorDataset(val_set_inputs, val_set_labels, val_set_weights)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)

print("train_set_samples: %d, val_set_samples: %d" % (len(train_set), len(val_set)))

epochs_to_train = 64
global_val_precision_best_avg = float("-inf")

clr_base_lr = 0.0001
clr_max_lr = 0.001

epoch_iterations = len(train_set) // batch_size
clr_step_size = 2 * epoch_iterations
# clr_scale_fn = lambda x: 1.0
clr_scale_fn = lambda x: 1.0 / (1.1 ** (x - 1))
# clr_scale_fn = lambda x: 0.5 * (1 + np.sin(x * np.pi / 2.))
clr_iterations = 0

optimizer = optim.Adam(model.parameters(), lr=clr_base_lr)

for epoch in range(epochs_to_train):

    epoch_start_time = time.time()

    epoch_train_loss_sum = 0.0
    epoch_train_precision_sum = 0.0
    epoch_train_step_count = 0
    for _, batch in enumerate(train_loader):
        inputs, labels, label_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        clr_cycle = np.floor(1 + clr_iterations / (2 * clr_step_size))
        clr_x = np.abs(clr_iterations / clr_step_size - 2 * clr_cycle + 1)
        lr = clr_base_lr + (clr_max_lr - clr_base_lr) * np.maximum(0, (1 - clr_x)) * clr_scale_fn(clr_cycle)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)
        criterion.weight = label_weights
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss_sum += loss.item()
        epoch_train_precision_sum += np.mean(precision_batch(predictions, labels))
        clr_iterations += 1
        epoch_train_step_count += 1

    epoch_val_loss_sum = 0.0
    epoch_val_precision_sum = 0.0
    epoch_val_step_count = 0
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            inputs, labels, label_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)
            criterion.weight = label_weights
            loss = criterion(outputs, labels)

            epoch_val_loss_sum += loss.item()
            epoch_val_precision_sum += np.mean(precision_batch(predictions, labels))
            epoch_val_step_count += 1

    epoch_train_loss_avg = epoch_train_loss_sum / epoch_train_step_count
    epoch_val_loss_avg = epoch_val_loss_sum / epoch_val_step_count
    epoch_train_precision_avg = epoch_train_precision_sum / epoch_train_step_count
    epoch_val_precision_avg = epoch_val_precision_sum / epoch_val_step_count

    ckpt_saved = False
    if epoch_val_precision_avg > global_val_precision_best_avg:
        torch.save(model.state_dict(), "{}/model.pth".format(output_dir))
        global_val_precision_best_avg = epoch_val_precision_avg
        ckpt_saved = True

    epoch_end_time = time.time()
    epoch_duration_time = epoch_end_time - epoch_start_time

    print(
        "[%03d/%03d] time: %ds, lr: %.6f, loss: %.3f, val_loss: %.3f, precision: %.3f, val_precision: %.3f, ckpt: %s" % (
            epoch + 1,
            epochs_to_train,
            epoch_duration_time,
            lr,
            epoch_train_loss_avg,
            epoch_val_loss_avg,
            epoch_train_precision_avg,
            epoch_val_precision_avg,
            ckpt_saved))
