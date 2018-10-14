import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset
from ensemble import Ensemble
from evaluate import analyze
from train import create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    input_dir = "/storage/kaggle/tgs"
    output_dir = "/artifacts"
    image_size_target = 128
    batch_size = 32
    ensemble_model_count = 3
    fold_count = 5
    fold_index = 3
    use_parallel_model = True
    use_val_set = True
    pin_memory = False
    swa_enabled = False
    pseudo_labeling_enabled = False
    pseudo_labeling_submission_csv = None
    pseudo_labeling_test_fold_count = 3
    pseudo_labeling_test_fold_index = 0
    pseudo_labeling_loss_weight_factor = 1.0

    criterion = nn.BCEWithLogitsLoss()

    train_data = TrainData(
        input_dir,
        fold_count,
        fold_index,
        use_val_set,
        pseudo_labeling_enabled,
        pseudo_labeling_submission_csv,
        pseudo_labeling_test_fold_count,
        pseudo_labeling_test_fold_index)

    val_set = TrainDataset(
        train_data.val_set_df,
        image_size_target,
        augment=False,
        train_set_scale_factor=1.0,
        pseudo_mask_weight_scale_factor=pseudo_labeling_loss_weight_factor)

    val_set_data_loader = \
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)

    eval_start_time = time.time()

    print()
    print("evaluation of the training model")

    m = create_model(type="unet_seresnext50_hc", input_size=128, pretrained=False, parallel=True).to(device)
    m.load_state_dict(torch.load("/storage/models/tgs/fold-3-pl/model.pth", map_location=device))

    ensemble_model = Ensemble([m])

    mask_threshold, best_mask_per_cc = analyze(ensemble_model, train_data.val_set_df, use_tta=False)

    eval_end_time = time.time()
    print()
    print("Eval time: %s" % str(datetime.timedelta(seconds=eval_end_time - eval_start_time)))


if __name__ == "__main__":
    main()
