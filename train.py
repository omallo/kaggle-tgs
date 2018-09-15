import time

import cv2
import numpy as np
import pandas as pd
import pydensecrf.densecrf as dcrf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from pydensecrf.utils import unary_from_labels
from scipy import ndimage
from skimage.color import gray2rgb
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from lovasz_losses import lovasz_hinge
from models.unet import UNet

# input_dir = "../salt/input"
# output_dir = "."
input_dir = "/storage/kaggle/tgs"
output_dir = "/storage"
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


class DiceWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        w = self.weight.view(num, -1) if self.weight is not None else torch.ones_like(m1)
        w2 = w * w

        score = 2. * ((w2 * intersection).sum(1) + smooth) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + smooth)
        loss = 1 - score.sum() / num

        return loss


class AggregateLoss(nn.Module):
    def __init__(self, *delegates):
        super().__init__()
        self.delegates = delegates

    def forward(self, input, targets):
        for delegate in self.delegates:
            delegate.weight = self.weight

        loss = self.delegates[0](input, targets)
        for delegate in self.delegates[1:]:
            loss += delegate(input, targets)

        return loss


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class LovaszWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        return lovasz_hinge(logits, labels, per_image=False)


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


def load_image(path, id):
    image = np.array(Image.open("{}/{}.png".format(path, id)))
    return np.squeeze(image[:, :, 0:1]) / 255 if len(image.shape) == 3 else image / 65535


def load_images(path, ids):
    return [load_image(path, id) for id in tqdm(ids)]


def upsample(img):
    p = (img_size_target - img_size_ori) / 2
    return np.pad(img, (int(np.ceil(p)), int(np.floor(p))), mode='reflect')


def downsample(img):
    p = (img_size_target - img_size_ori) / 2
    s = int(np.ceil(p))
    e = s + img_size_ori
    return img[s:e, s:e]


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


def precision_array(outputs, labels):
    return [precision(o, l) for o, l in zip(outputs, labels)]


# https://www.microsoft.com/developerblog/2018/05/17/using-otsus-method-generate-data-training-deep-learning-image-segmentation-models/
def compute_otsu_mask(image):
    image = np.stack((image,) * 3, -1)
    image = image.astype(np.uint8)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


"""
Function which returns the labelled image after applying CRF
"""


# Original_image = Image which has to labelled
# Mask image = Which has been labelled by some technique..
def crf(original_image, mask_img):
    # Converting annotated image to RGB if it is Gray scale
    if (len(mask_img.shape) < 3):
        mask_img = gray2rgb(mask_img)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0], original_image.shape[1]))


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
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
label_transform = transforms.Compose([
    transforms.ToTensor()
])

model = UNet(in_depth=3, out_depth=1, base_channels=64).to(device)
# model = AlbuNet(pretrained=True).to(device)

# model.load_state_dict(torch.load("{}/albunet.pth".format(output_dir)))

# criterion = AggregateLoss(nn.BCEWithLogitsLoss(), LovaszWithLogitsLoss())
criterion = nn.BCEWithLogitsLoss()
# criterion = DiceWithLogitsLoss()
# criterion = FocalWithLogitsLoss(2.0)
# criterion = RobustFocalLoss2d()
# criterion = LovaszWithLogitsLoss()

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

epochs_to_train = 40
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
        torch.save(model.state_dict(), "{}/albunet.pth".format(output_dir))
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

val_pred_set = TensorDataset(val_set_inputs)
val_pred_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

model.load_state_dict(torch.load("{}/albunet.pth".format(output_dir)))

val_predictions = []
with torch.no_grad():
    for _, batch in enumerate(val_pred_loader):
        inputs = batch[0].to(device)
        outputs = model(inputs)
        predictions = torch.sigmoid(outputs)
        val_predictions += [p for p in predictions.cpu().numpy()]

val_predictions = np.asarray(val_predictions).reshape(-1, img_size_target, img_size_target)
val_predictions = [downsample(p) for p in val_predictions]
val_set_df["predictions"] = val_predictions

thresholds = np.linspace(0, 1, 51)

precisions_per_threshold = np.array(
    [np.mean(precision_array(np.int32(np.asarray(val_set_df.predictions.tolist()) > t), val_set_df.masks)) for t in
     tqdm(thresholds)])

threshold_best_index = np.argmax(precisions_per_threshold)
precision_best = precisions_per_threshold[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

val_set_df["prediction_masks"] = [np.int32(p > threshold_best) for p in val_set_df.predictions]
val_set_df["precisions"] = [precision(pm, m) for pm, m in zip(val_set_df.prediction_masks, val_set_df.masks)]


def calc_max_precision(y_pred_raw, y_true):
    ious = [precision(np.int32(y_pred_raw > t), y_true) for t in thresholds]
    am = np.argmax(ious)
    return (thresholds[am], ious[am])


optimals = [calc_max_precision(p, m) for p, m in zip(val_set_df.predictions, val_set_df.masks)]

val_set_df["thresholds_opt"] = [o[0] for o in optimals]
val_set_df["precisions_opt"] = [o[1] for o in optimals]

print()
print("precision_best: %.3f, threshold_best: %.3f" % (precision_best, threshold_best))
print("precision_opt: %.3f, threshold_opt_mean: %.3f"
      % (val_set_df.precisions_opt.mean(), val_set_df.thresholds_opt.mean()))

print()
print(val_set_df.thresholds_opt.describe())

print()
print(val_set_df.precisions.describe())

print()
print(val_set_df.precisions_opt.describe())

val_set_df["prediction_coverage"] = val_set_df.predictions.map(np.sum) / pow(img_size_ori, 2)
val_set_df["prediction_coverage_class"] = val_set_df.prediction_coverage.map(coverage_to_class)

val_set_df["predictions_otsu"] = [np.int32(compute_otsu_mask(255 * p) / 255) for p in val_set_df.predictions]
val_set_df["precisions_otsu"] = [precision(p, m) for p, m in zip(val_set_df.predictions_otsu, val_set_df.masks)]

print()
print("precision_otsu: %.3f" % val_set_df.precisions_otsu.mean())

val_set_df["predictions_crf"] = \
    [crf(np.array(i), np.int32(np.array(p))) for i, p in zip(val_set_df.images, val_set_df.prediction_masks)]
val_set_df["precisions_crf"] = [precision(p, m) for p, m in zip(val_set_df.predictions_crf, val_set_df.masks)]

print()
print(val_set_df.precisions_crf.describe())

print()
print(val_set_df
    .groupby("prediction_coverage_class")
    .agg({
    "precisions": "mean",
    "precisions_opt": "mean",
    "precisions_otsu": "mean",
    "precisions_crf": "mean",
    "prediction_coverage_class": "count"
}))

print()
print(val_set_df
    .groupby("coverage_class")
    .agg({
    "precisions": "mean",
    "precisions_opt": "mean",
    "precisions_otsu": "mean",
    "precisions_crf": "mean",
    "coverage_class": "count"
}))
