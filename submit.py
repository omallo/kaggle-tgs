import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from unet_models import AlbuNet

input_dir = "/storage/kaggle/tgs"
output_dir = "/artifacts"
img_size_ori = 101
img_size_target = 128
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        return tuple(self.image_transform(self.images[index]))


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


def prepare_input(image):
    return np.expand_dims(upsample(image), axis=2).repeat(3, axis=2)


def predict(model, val_pred_loader):
    val_predictions = []
    with torch.no_grad():
        for _, batch in enumerate(val_pred_loader):
            inputs = batch.to(device)
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs)
            val_predictions += [p for p in predictions.cpu().numpy()]
    val_predictions = np.asarray(val_predictions).reshape(-1, img_size_target, img_size_target)
    val_predictions = [downsample(p) for p in val_predictions]
    return val_predictions


def main():
    train_df = pd.read_csv("{}/train.csv".format(input_dir), index_col="id", usecols=[0])
    depths_df = pd.read_csv("{}/depths.csv".format(input_dir), index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    test_df["images"] = load_images("{}/test/images".format(input_dir), test_df.index)

    mask_model = AlbuNet(pretrained=True).to(device)
    mask_model.load_state_dict(torch.load("/storage/masks.pth"))

    test_set = TrainDataset(test_df.images.tolist())
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    test_df["predictions"] = predict(mask_model, test_data_loader)
    test_df["prediction_masks"] = [np.int32(p > 0.4) for p in test_df.predictions]

    pred_dict = {idx: RLenc(test_df.loc[idx].prediction_masks) for i, idx in tqdm(enumerate(test_df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission.csv".format(output_dir))


if __name__ == "__main__":
    main()
