import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

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
        self.image_size_original = df.images.shape[0]
        self.image_size_target = image_size_target
        self.augment = augment
        self.image_transform = lambda i: prepare_image(i, image_size_target)
        self.mask_transform = lambda m: prepare_mask(m, image_size_target)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = self.df.images[index]
        mask = self.df.masks[index]

        if self.augment:
            image, mask = augment(image, mask)

        mask_weights = calculate_mask_weights(mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask_weights = self.mask_transform(mask_weights)

        return image, mask, mask_weights


def load_images(path, ids):
    return [load_image(path, id) for id in tqdm(ids)]


def load_image(path, id):
    image = cv2.imread("{}/{}.png".format(path, id))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_masks(path, ids):
    return [load_mask(path, id) for id in tqdm(ids)]


def load_mask(path, id):
    mask = cv2.imread("{}/{}.png".format(path, id), 0)
    return (mask > 0).astype(np.uint8)


def upsample(img, image_size_target):
    if image_size_target >= 2 * img.shape[0]:
        return upsample(
            np.pad(np.pad(img, ((0, 0), (0, img.shape[0])), "reflect"), ((0, img.shape[0]), (0, 0)), "reflect"),
            image_size_target)
    p = (image_size_target - img.shape[0]) / 2
    return np.pad(img, (int(np.ceil(p)), int(np.floor(p))), mode='reflect')


def downsample(img, image_size_original):
    if img.shape[0] >= 2 * image_size_original:
        p = (img.shape[0] - 2 * image_size_original) / 2
    else:
        p = (img.shape[0] - image_size_original) / 2
    s = int(np.ceil(p))
    e = s + image_size_original
    unpadded = img[s:e, s:e]
    if img.shape[0] >= 2 * image_size_original:
        return unpadded[0:image_size_original, 0:image_size_original]
    else:
        return unpadded


def prepare_image(image, image_size_target):
    return np.expand_dims(upsample(image, image_size_target), axis=2).repeat(3, axis=2)


def prepare_mask(mask, image_size_target):
    return np.expand_dims(upsample(mask, image_size_target), axis=2)


def calculate_coverage_class(mask):
    coverage = mask.sum() / mask.size
    for i in range(0, 11):
        if coverage * 10 <= i:
            return i
