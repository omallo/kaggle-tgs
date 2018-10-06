import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from processing import calculate_mask_weights, rldec
from transforms import augment, upsample, reduce_salt_coverage
from utils import kfold_split


class TrainData:
    def __init__(self, base_dir, kfold_count, kfold_index, pseudo_labeling_enabled, pseudo_labeling_submission_csv,
                 pseudo_labeling_all_in):
        train_df = pd.read_csv("{}/train.csv".format(base_dir), index_col="id", usecols=[0])
        depths_df = pd.read_csv("{}/depths.csv".format(base_dir), index_col="id")
        train_df = train_df.join(depths_df)

        train_df["images"] = load_images("{}/train/images".format(base_dir), train_df.index)
        train_df["masks"] = load_masks("{}/train/masks".format(base_dir), train_df.index)
        train_df["coverage_class"] = train_df.masks.map(calculate_coverage_class)

        if pseudo_labeling_all_in:
            train_set_ids, val_set_ids = train_test_split(
                sorted(train_df.index.values),
                train_size=0.8,
                stratify=train_df.coverage_class,
                random_state=42)
        else:
            train_set_ids, val_set_ids = \
                list(kfold_split(kfold_count, sorted(train_df.index.values), train_df.coverage_class))[kfold_index]

        train_set_df = train_df[train_df.index.isin(train_set_ids)].copy()
        val_set_df = train_df[train_df.index.isin(val_set_ids)].copy()

        if pseudo_labeling_enabled:
            test_df = pd.read_csv(pseudo_labeling_submission_csv, index_col="id")
            test_df["rle_mask"] = test_df.rle_mask.astype(str)
            test_df["masks"] = test_df.rle_mask.map(rldec)
            test_df["images"] = load_images("{}/test/images".format(base_dir), test_df.index)
            test_df["coverage_class"] = test_df.masks.map(calculate_coverage_class)
            test_df = test_df.drop(columns=["rle_mask"])

            if pseudo_labeling_all_in:
                leftover_train_set_ids, test_train_set_ids = test_df.index.values, test_df.index.values
            else:
                leftover_train_set_ids, test_train_set_ids = \
                    list(kfold_split(kfold_count, sorted(test_df.index.values), test_df.coverage_class))[kfold_index]

            test_train_set_df = test_df[test_df.index.isin(test_train_set_ids)].copy()

            cc1_count = np.sum(test_train_set_df.coverage_class == 1)
            np.random.shuffle(leftover_train_set_ids)
            ids_with_reduced_cc1 = []
            for id in leftover_train_set_ids:
                image = test_df.loc[id].images
                mask = test_df.loc[id].masks
                if calculate_coverage_class(mask) > 2:
                    image, mask = reduce_salt_coverage(image, mask)
                    if calculate_coverage_class(mask) == 1:
                        ids_with_reduced_cc1.append(id)
            print("reduced the salt coverage of {} test images to 1, {} had been dropped".format(
                len(ids_with_reduced_cc1), cc1_count))
            test_train_set_df = test_train_set_df.drop(test_train_set_df.index[test_train_set_df.coverage_class == 1])

            if pseudo_labeling_all_in:
                test_train_set_df = pd.concat(
                    [test_train_set_df, test_df[test_df.index.isin(ids_with_reduced_cc1)].copy()])
            else:
                test_train_set_df = pd.concat(
                    [test_train_set_df, test_df[test_df.index.isin(ids_with_reduced_cc1[:cc1_count])].copy()])

            train_set_df = pd.concat([train_set_df, test_train_set_df])

        self.train_set_df = train_set_df
        self.val_set_df = val_set_df


class TestData:
    def __init__(self, base_dir):
        train_df = pd.read_csv("{}/train.csv".format(base_dir), index_col="id", usecols=[0])
        depths_df = pd.read_csv("{}/depths.csv".format(base_dir), index_col="id")
        train_df = train_df.join(depths_df)
        test_df = depths_df[~depths_df.index.isin(train_df.index)].copy()

        test_df["images"] = load_images("{}/test/images".format(base_dir), test_df.index)

        self.df = test_df


class TrainDataset(Dataset):
    def __init__(self, df, image_size_target, augment, train_set_scale_factor=1.0):
        super().__init__()
        self.df = df
        self.image_size_target = image_size_target
        self.augment = augment
        self.train_set_scale_factor = train_set_scale_factor

    def __len__(self):
        return int(self.train_set_scale_factor * len(self.df))

    def __getitem__(self, index):
        image = self.df.images[index % len(self.df)]
        mask = self.df.masks[index % len(self.df)]

        if self.augment:
            image, mask = augment(image, mask)

        mask_weights = calculate_mask_weights(mask)

        image = upsample(image, self.image_size_target)
        mask = upsample(mask, self.image_size_target)
        mask_weights = upsample(mask_weights, self.image_size_target)

        image = image_to_tensor(image)
        mask = mask_to_tensor(mask)
        mask_weights = mask_to_tensor(mask_weights)

        image_mean = 0.4719
        image_std = 0.1610

        image = normalize(image, (image_mean, image_mean, image_mean), (image_std, image_std, image_std))

        return image, mask, mask_weights


class TestDataset(Dataset):
    def __init__(self, df, image_size_target):
        super().__init__()
        self.df = df
        self.image_size_target = image_size_target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = self.df.images[index]
        image = upsample(image, self.image_size_target)
        image = image_to_tensor(image)

        image_mean = 0.4719
        image_std = 0.1610

        image = normalize(image, (image_mean, image_mean, image_mean), (image_std, image_std, image_std))

        return image


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


def load_glcm_features(path, feature_name, ids):
    return [load_glcm_feature(path, feature_name, id) for id in ids]


def load_glcm_feature(path, feature_name, id):
    feature_0 = cv2.imread("{}/{}-0/{}.png".format(path, feature_name, id), 0)
    feature_90 = cv2.imread("{}/{}-90/{}.png".format(path, feature_name, id), 0)
    return cv2.addWeighted(feature_0, 0.5, feature_90, 0.5, 0)


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
    return torch.from_numpy((np.moveaxis(image, -1, 0) / 255).copy()).float()


def mask_to_tensor(mask):
    return torch.from_numpy(np.expand_dims(mask, 0).copy()).float()


def set_depth_channels(image, depth):
    max_depth = 1010
    image = image.copy()
    h, w, _ = image.shape
    for row, const in enumerate(np.linspace(0, 1, h)):
        image[row, :, 1] = int(np.round(255 * (depth - 50 + row) / max_depth))
        image[row, :, 2] = np.round(const * image[row, :, 0]).astype(image.dtype)
    return image


def add_depth_channels(image_tensor):
    _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[1, row, :] = const
    image_tensor[2] = image_tensor[0] * image_tensor[1]
    return image_tensor
