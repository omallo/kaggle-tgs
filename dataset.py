import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from processing import calculate_mask_weights
from transforms import augment


class TrainData:
    def __init__(self, base_dir):
        train_df = pd.read_csv("{}/train.csv".format(base_dir), index_col="id", usecols=[0])
        depths_df = pd.read_csv("{}/depths.csv".format(base_dir), index_col="id")
        train_df = train_df.join(depths_df)

        train_df["images"] = load_images("{}/train/images".format(base_dir), train_df.index)
        train_df["masks"] = load_masks("{}/train/masks".format(base_dir), train_df.index)
        train_df["coverage_class"] = train_df.masks.map(calculate_coverage_class)

        train_set_ids, val_set_ids = train_test_split(
            sorted(train_df.index.values),
            test_size=0.2,
            stratify=train_df.coverage_class,
            random_state=42)

        self.train_set_df = train_df[train_df.index.isin(train_set_ids)].copy()
        self.val_set_df = train_df[train_df.index.isin(val_set_ids)].copy()


class TrainDataset(Dataset):
    def __init__(self, df, image_size_target, augment):
        super().__init__()
        self.df = df
        self.image_size_target = image_size_target
        self.augment = augment

    def __len__(self):
        return 2 * len(self.df) if self.augment else len(self.df)

    def __getitem__(self, index):
        image = self.df.images[index % len(self.df)]
        mask = self.df.masks[index % len(self.df)]

        if self.augment and index >= len(self.df):
            image, mask = augment(image, mask)

        mask_weights = calculate_mask_weights(mask)

        image = upsample(image, self.image_size_target)
        mask = upsample(mask, self.image_size_target)
        mask_weights = upsample(mask_weights, self.image_size_target)

        image = normalize(image_to_tensor(image), (0.4719, 0.4719, 0.4719), (0.1610, 0.1610, 0.1610))
        mask = mask_to_tensor(mask)
        mask_weights = mask_to_tensor(mask_weights)

        return image, mask, mask_weights


def load_images(path, ids):
    return [load_image(path, id) for id in ids]


def load_image(path, id):
    image = cv2.imread("{}/{}.png".format(path, id))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_masks(path, ids):
    return [load_mask(path, id) for id in ids]


def load_mask(path, id):
    mask = cv2.imread("{}/{}.png".format(path, id), 0)
    return (mask > 0).astype(np.uint8)


def upsample(image, image_size_target):
    padding = (image_size_target - image.shape[0]) / 2
    padding_start = int(np.ceil(padding))
    padding_end = int(np.floor(padding))
    return cv2.copyMakeBorder(image, padding_start, padding_end, padding_start, padding_end, cv2.BORDER_REFLECT_101)


def downsample(image, image_size_original):
    padding = (image_size_original - image.shape[0]) / 2
    padding_start = int(np.ceil(padding))
    return image[padding_start:padding_start + image_size_original, padding_start:padding_start + image_size_original]


def prepare_image(image, image_size_target):
    return np.expand_dims(upsample(image, image_size_target), axis=2).repeat(3, axis=2)


def prepare_mask(mask, image_size_target):
    return np.expand_dims(upsample(mask, image_size_target), axis=2)


def calculate_coverage_class(mask):
    coverage = mask.sum() / mask.size
    for i in range(0, 11):
        if coverage * 10 <= i:
            return i


def image_to_tensor(image):
    return torch.from_numpy(np.moveaxis(image, -1, 0) / 255).float()


def mask_to_tensor(mask):
    return torch.from_numpy(np.expand_dims(mask, 0)).float()
