import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from unet_models import AlbuNet

# input_dir = "../salt/input"
# output_dir = "."
input_dir = "/storage/kaggle/tgs"
output_dir = "."
img_size_ori = 101
img_size_target = 128
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def mask_weights(mask):
    return np.ones_like(mask) + 2 * contour(mask)


def dice_loss(outputs, labels):
    smooth = 1.

    outputs_flat = outputs.view(-1)
    labels_flat = labels.view(-1)
    intersection = (outputs_flat * labels_flat).sum()

    return 1 - ((2. * intersection + smooth) / (outputs_flat.sum() + labels_flat.sum() + smooth))


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


train_df = pd.read_csv("{}/train.csv".format(input_dir), index_col="id", usecols=[0])
depths_df = pd.read_csv("{}/depths.csv".format(input_dir), index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

train_df["images"] = load_images("{}/train/images".format(input_dir), train_df.index)
train_df["masks"] = load_images("{}/train/masks".format(input_dir), train_df.index)

train_df["contours"] = train_df.masks.map(contour)
train_df["mask_weights"] = train_df.masks.map(mask_weights)

train_val_split = int(0.8 * len(train_df))
train_set_ids = train_df.index.tolist()[:train_val_split]
val_set_ids = train_df.index.tolist()[train_val_split:]

train_set_df = train_df[train_df.index.isin(train_set_ids)].copy()
val_set_df = train_df[train_df.index.isin(val_set_ids)].copy()

train_set_x = train_set_df.images.tolist()
train_set_y = train_set_df.masks.tolist()
train_set_w = train_set_df.mask_weights.tolist()

train_set_x = np.append(train_set_x, [np.fliplr(x) for x in train_set_x], axis=0)
train_set_y = np.append(train_set_y, [np.fliplr(y) for y in train_set_y], axis=0)
train_set_w = np.append(train_set_w, [np.fliplr(w) for w in train_set_w], axis=0)

val_set_x = val_set_df.images.tolist()
val_set_y = val_set_df.masks.tolist()
val_set_w = val_set_df.mask_weights.tolist()

input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
label_transform = transforms.Compose([
    transforms.ToTensor()
])

model = AlbuNet(pretrained=True) \
    .to(device)

# model.load_state_dict(torch.load("{}/albunet.pth".format(output_dir)))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# TODO: pass transforms to data loader
with torch.no_grad():
    train_set_inputs = prepare_inputs(train_set_x, input_transform)
    train_set_labels = prepare_labels(train_set_y, label_transform)
    train_set_weights = prepare_labels(train_set_w, label_transform)

    val_set_inputs = prepare_inputs(val_set_x, input_transform)
    val_set_labels = prepare_labels(val_set_y, label_transform)
    val_set_weights = prepare_labels(val_set_w, label_transform)

train_set = TensorDataset(train_set_inputs, train_set_labels, train_set_weights)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)

val_set = TensorDataset(val_set_inputs, val_set_labels, val_set_weights)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=False)

print("train_set_samples: %d, val_set_samples: %d" % (len(train_set), len(val_set)))

epochs_to_train = 30
global_val_precision_best_avg = float("-inf")

clr_base_lr = 0.0001
clr_max_lr = 0.001

epoch_iterations = len(train_set) // batch_size
clr_step_size = 2 * epoch_iterations
clr_scale_fn = lambda x: 1.0 / (1.1 ** (x - 1))
clr_iterations = 0

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
