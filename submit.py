import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TrainData, TestData, TestDataset, calculate_coverage_class
from evaluate import analyze, predict
from models import create_model
from processing import rle_encode, crf

input_dir = "/storage/kaggle/tgs"
output_dir = "/artifacts"
image_size_target = 128
batch_size = 32

model_dir = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    train_data = TrainData(input_dir)

    model = create_model(pretrained=False).to(device)
    model.load_state_dict(torch.load("{}/model.pth".format(model_dir), map_location=device))

    mask_threshold_global, mask_threshold_per_cc = analyze(model, train_data.val_set_df)

    test_data = TestData(input_dir)

    test_set = TestDataset(test_data.df, image_size_target)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    test_data.df["predictions"] = predict(model, test_data_loader)
    test_data.df["prediction_masks"] = [np.int32(p > mask_threshold_global) for p in test_data.df.predictions]

    test_data.df["predictions_cc"] = test_data.df.prediction_masks.map(calculate_coverage_class)
    test_data.df["prediction_masks_cc"] = [np.int32(p > mask_threshold_per_cc[cc]) for p, cc in
                                           zip(test_data.df.predictions, test_data.df.predictions_cc)]

    test_data.df["prediction_masks_crf"] = [crf(i, pm) for i, pm in zip(test_data.df.images, test_data.df.prediction_masks)]

    pred_dict = {idx: rle_encode(test_data.df.loc[idx].prediction_masks) for i, idx in
                 tqdm(enumerate(test_data.df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission.csv".format(output_dir))

    pred_dict = {idx: rle_encode(test_data.df.loc[idx].prediction_masks_cc) for i, idx in
                 tqdm(enumerate(test_data.df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission_cc.csv".format(output_dir))

    pred_dict = {idx: rle_encode(test_data.df.loc[idx].prediction_masks_crf) for i, idx in
                 tqdm(enumerate(test_data.df.index.values))}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv("{}/submission_crf.csv".format(output_dir))


if __name__ == "__main__":
    main()
