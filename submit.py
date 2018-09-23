import sys

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset
from evaluate import analyze
from models import create_model

input_dir = "/storage/kaggle/tgs"
image_size_target = 128
batch_size = 32

model_dir = sys.argv[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 160)

    train_data = TrainData(input_dir)

    val_set = TrainDataset(train_data.val_set_df, image_size_target, augment=False)
    val_set_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = create_model(pretrained=False).to(device)
    model.load_state_dict(torch.load("{}/model.pth".format(model_dir), map_location=device))
    model.eval()

    analyze(model, val_set_data_loader, train_data.val_set_df)


if __name__ == "__main__":
    main()
