import datetime
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset
from swa_utils import moving_average, bn_update
from train import create_model, evaluate

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
    pseudo_labeling_extend_val_set = False
    pseudo_labeling_loss_weight_factor = 0.6

    criterion = nn.BCEWithLogitsLoss()

    train_data = TrainData(
        input_dir,
        fold_count,
        fold_index,
        use_val_set,
        pseudo_labeling_enabled,
        pseudo_labeling_submission_csv,
        pseudo_labeling_test_fold_count,
        pseudo_labeling_test_fold_index,
        pseudo_labeling_extend_val_set)

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

    swa_model = None
    for i in range(17):
        model_file_path = "/storage/models/tgs/pl-weight-0.6/model-{}.pth".format(i)
        m = create_model(type="unet_seresnext50_hc", input_size=128, pretrained=False, parallel=use_parallel_model).to(
            device)
        m.load_state_dict(torch.load(model_file_path, map_location=device))

        if swa_model is None:
            swa_model = m
        else:
            moving_average(swa_model, m, 1.0 / (i + 1))
            bn_update(val_set_data_loader, swa_model)

        val_loss_avg, val_precision_avg = evaluate(swa_model, val_set_data_loader, criterion)

        print("val_loss: {}, val_precision: {}".format(val_loss_avg, val_precision_avg))

    eval_end_time = time.time()
    print()
    print("Eval time: %s" % str(datetime.timedelta(seconds=eval_end_time - eval_start_time)))


if __name__ == "__main__":
    main()
