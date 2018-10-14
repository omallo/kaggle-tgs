import datetime
import time

import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset, TestData
from ensemble import Ensemble
from evaluate import analyze, calculate_predictions, calculate_predictions_cc, calculate_prediction_masks, \
    calculate_best_prediction_masks
from train import load_ensemble_model
from utils import write_submission


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

    models = []

    _, ensemble_model = load_ensemble_model(
        ensemble_model_count, "/storage/models/tgs/seresnext-hc-fold-3", val_set_data_loader,
        criterion, swa_enabled, "unet_seresnext50_hc",
        image_size_target, use_parallel_model=use_parallel_model)
    models.append(ensemble_model)

    _, ensemble_model = load_ensemble_model(
        ensemble_model_count, "/storage/models/tgs/seresnext101-hc-fold-3", val_set_data_loader,
        criterion, swa_enabled, "unet_seresnext101_hc",
        image_size_target, use_parallel_model=use_parallel_model)
    models.append(ensemble_model)

    _, ensemble_model = load_ensemble_model(
        ensemble_model_count, "/storage/models/tgs/senet", val_set_data_loader,
        criterion, swa_enabled, "unet_senet",
        image_size_target, use_parallel_model=use_parallel_model)
    models.append(ensemble_model)

    ensemble_model = Ensemble(models)

    mask_threshold, best_mask_per_cc = analyze(ensemble_model, train_data.val_set_df, use_tta=True)

    eval_end_time = time.time()
    print()
    print("Eval time: %s" % str(datetime.timedelta(seconds=eval_end_time - eval_start_time)))

    print()
    print("submission preparation")

    submission_start_time = time.time()

    test_data = TestData(input_dir)
    calculate_predictions(test_data.df, ensemble_model, use_tta=True)
    calculate_predictions_cc(test_data.df, mask_threshold)
    calculate_prediction_masks(test_data.df, mask_threshold)
    calculate_best_prediction_masks(test_data.df, best_mask_per_cc)

    print()
    print(test_data.df.groupby("predictions_cc").agg({"predictions_cc": "count"}))

    write_submission(test_data.df, "prediction_masks", "{}/{}".format(output_dir, "submission.csv"))
    write_submission(test_data.df, "prediction_masks_best", "{}/{}".format(output_dir, "submission_best.csv"))
    write_submission(test_data.df, "prediction_masks_best_pp", "{}/{}".format(output_dir, "submission_best_pp.csv"))

    submission_end_time = time.time()
    print()
    print("Submission time: %s" % str(datetime.timedelta(seconds=submission_end_time - submission_start_time)))


if __name__ == "__main__":
    main()
