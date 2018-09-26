import datetime
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset, TestData, TestDataset, calculate_coverage_class
from evaluate import analyze, predict
from losses import BCELovaszLoss
from metrics import precision_batch
from models import create_model
from processing import crf_batch
from utils import get_learning_rate, write_submission, unfreeze

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def evaluate(model, data_loader, criterion):
    model.eval()

    loss_sum = 0.0
    precision_sum = 0.0
    step_count = 0

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            images, masks, mask_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            prediction_logits = model(images)

            masks = masks[:, :, 14:14 + 101, 14:14 + 101].contiguous()
            mask_weights = mask_weights[:, :, 14:14 + 101, 14:14 + 101].contiguous()
            prediction_logits = prediction_logits[:, :, 14:14 + 101, 14:14 + 101].contiguous()

            predictions = torch.sigmoid(prediction_logits)
            criterion.weight = mask_weights
            loss = criterion(prediction_logits, masks)

            loss_sum += loss.item()
            precision_sum += np.mean(precision_batch(predictions, masks))
            step_count += 1

    loss_avg = loss_sum / step_count
    precision_avg = precision_sum / step_count

    return loss_avg, precision_avg


def main():
    input_dir = "/storage/kaggle/tgs"
    output_dir = "/artifacts"
    image_size_target = 128
    batch_size = 32
    epochs_to_train = 160
    bce_loss_weight_gamma = 0.98
    # swa_start_epoch = 10
    # swa_cycle_epochs = 5
    sgdr_min_lr = 0.001  # 0.0001, 0.001
    sgdr_max_lr = 0.03  # 0.001, 0.03
    sgdr_cycle_epochs = 20
    sgdr_cycle_end_patience = 3
    train_abort_epochs_without_improval = 20
    epoch_to_unfreeze_encoder = 10

    model_dir = sys.argv[1] if len(sys.argv) > 1 else None

    train_data = TrainData(input_dir)

    train_set = TrainDataset(train_data.train_set_df, image_size_target, augment=True)
    train_set_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    val_set = TrainDataset(train_data.val_set_df, image_size_target, augment=False)
    val_set_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    if model_dir:
        model = create_model(pretrained=False).to(device)
        model.load_state_dict(torch.load("{}/model.pth".format(model_dir), map_location=device))
    else:
        model = create_model(pretrained=True).to(device)

    torch.save(model.state_dict(), "{}/model.pth".format(output_dir))

    # freeze(model.encoder)

    # swa_model = create_model(pretrained=False).to(device)
    # swa_model.load_state_dict(model.state_dict())

    print("train_set_samples: %d, val_set_samples: %d" % (len(train_set), len(val_set)))

    global_val_precision_overall_avg = float("-inf")
    global_val_precision_best_avg = float("-inf")
    global_val_precision_swa_best_avg = float("-inf")

    epoch_iterations = len(train_set) // batch_size

    optimizer = optim.SGD(model.parameters(), lr=sgdr_max_lr, weight_decay=0, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=sgdr_max_lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=sgdr_cycle_epochs, eta_min=sgdr_min_lr)

    optim_summary_writer = SummaryWriter(log_dir="{}/logs/optim".format(output_dir))
    train_summary_writer = SummaryWriter(log_dir="{}/logs/train".format(output_dir))
    val_summary_writer = SummaryWriter(log_dir="{}/logs/val".format(output_dir))
    # val_swa_summary_writer = SummaryWriter(log_dir="{}/logs/val_swa".format(output_dir))

    sgdr_iterations = 0
    sgdr_reset_count = 0
    # swa_update_count = 0
    batch_count = 0
    epoch_of_last_improval = 0
    sgdr_next_cycle_end_epoch = sgdr_cycle_epochs

    print('{"chart": "best_val_precision", "axis": "epoch"}')
    print('{"chart": "val_precision", "axis": "epoch"}')
    print('{"chart": "val_loss", "axis": "epoch"}')
    # print('{"chart": "val_precision_swa", "axis": "epoch"}')
    # print('{"chart": "val_loss_swa", "axis": "epoch"}')
    print('{"chart": "sgdr_reset", "axis": "epoch"}')
    # print('{"chart": "swa_update", "axis": "epoch"}')

    train_start_time = time.time()

    for epoch in range(epochs_to_train):
        epoch_start_time = time.time()

        criterion = BCELovaszLoss(bce_weight=bce_loss_weight_gamma ** epoch)

        model.train()

        if epoch + 1 == epoch_to_unfreeze_encoder:
            unfreeze(model.encoder)

        train_loss_sum = 0.0
        train_precision_sum = 0.0
        train_step_count = 0
        for _, batch in enumerate(train_set_data_loader):
            images, masks, mask_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            lr_scheduler.step(epoch=min(sgdr_cycle_epochs, sgdr_iterations / epoch_iterations))

            optimizer.zero_grad()
            prediction_logits = model(images)
            predictions = torch.sigmoid(prediction_logits)
            criterion.weight = mask_weights
            loss = criterion(prediction_logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_precision_sum += np.mean(precision_batch(predictions, masks))
            sgdr_iterations += 1
            train_step_count += 1
            batch_count += 1

            optim_summary_writer.add_scalar("lr", get_learning_rate(optimizer), batch_count + 1)

        train_loss_avg = train_loss_sum / train_step_count
        train_precision_avg = train_precision_sum / train_step_count

        val_loss_avg, val_precision_avg = evaluate(model, val_set_data_loader, criterion)

        model_improved = val_precision_avg > global_val_precision_best_avg
        ckpt_saved = False
        if model_improved:
            torch.save(model.state_dict(), "{}/model.pth".format(output_dir))
            global_val_precision_best_avg = val_precision_avg
            ckpt_saved = True

        # swa_updated = False
        # if epoch + 1 >= swa_start_epoch and (model_improved or ((epoch + 1) % swa_cycle_epochs == 0)):
        #     swa_update_count += 1
        #     moving_parameter_average(swa_model, model, 1.0 / swa_update_count)
        #     swa_updated = True
        #
        # val_loss_swa_avg, val_precision_swa_avg = evaluate(swa_model, val_set_data_loader, criterion)
        #
        # swa_model_improved = epoch + 1 >= swa_start_epoch and val_precision_swa_avg > global_val_precision_swa_best_avg
        # swa_ckpt_saved = False
        # if swa_model_improved:
        #     torch.save(swa_model.state_dict(), "{}/swa_model.pth".format(output_dir))
        #     global_val_precision_swa_best_avg = val_precision_swa_avg
        #     swa_ckpt_saved = True

        epoch_end_time = time.time()
        epoch_duration_time = epoch_end_time - epoch_start_time

        model_improved_overall = global_val_precision_best_avg > global_val_precision_overall_avg
        # swa_model_improved_overall = swa_updated and global_val_precision_swa_best_avg > global_val_precision_overall_avg
        swa_model_improved_overall = False
        if model_improved_overall or swa_model_improved_overall:
            global_val_precision_overall_avg = max(global_val_precision_best_avg, global_val_precision_swa_best_avg)
            epoch_of_last_improval = epoch

        sgdr_reset = False
        if (epoch + 1 >= sgdr_next_cycle_end_epoch) and (epoch - epoch_of_last_improval >= sgdr_cycle_end_patience):
            sgdr_iterations = 0
            sgdr_next_cycle_end_epoch = epoch + 1 + sgdr_cycle_epochs
            sgdr_reset_count += 1
            sgdr_reset = True

        optim_summary_writer.add_scalar("sgdr_reset", sgdr_reset_count, epoch + 1)
        # optim_summary_writer.add_scalar("swa_update", swa_update_count, epoch + 1)

        train_summary_writer.add_scalar("loss", train_loss_avg, epoch + 1)
        train_summary_writer.add_scalar("precision", train_precision_avg, epoch + 1)

        val_summary_writer.add_scalar("loss", val_loss_avg, epoch + 1)
        val_summary_writer.add_scalar("precision", val_precision_avg, epoch + 1)

        # if swa_updated:
        #     val_swa_summary_writer.add_scalar("loss", val_loss_swa_avg, epoch + 1)
        #     val_swa_summary_writer.add_scalar("precision", val_precision_swa_avg, epoch + 1)

        print(
            "[%03d/%03d] %ds, lr: %.6f, loss: %.3f, val_loss: %.3f, prec: %.3f, val_prec: %.3f, ckpt: %d, rst: %d" % (
                epoch + 1,
                epochs_to_train,
                epoch_duration_time,
                get_learning_rate(optimizer),
                train_loss_avg,
                val_loss_avg,
                # val_loss_swa_avg,
                train_precision_avg,
                val_precision_avg,
                # val_precision_swa_avg,
                # int(swa_updated),
                int(ckpt_saved),
                # int(swa_ckpt_saved),
                int(sgdr_reset)),
            flush=True)

        print('{"chart": "best_val_precision", "x": %d, "y": %.3f}' % (epoch + 1, global_val_precision_overall_avg))
        print('{"chart": "val_precision", "x": %d, "y": %.3f}' % (epoch + 1, val_precision_avg))
        print('{"chart": "val_loss", "x": %d, "y": %.3f}' % (epoch + 1, val_loss_avg))
        # print('{"chart": "val_precision_swa", "x": %d, "y": %.3f}' % (epoch + 1, val_precision_swa_avg))
        # print('{"chart": "val_loss_swa", "x": %d, "y": %.3f}' % (epoch + 1, val_loss_swa_avg))
        print('{"chart": "sgdr_reset", "x": %d, "y": %.3f}' % (epoch + 1, sgdr_reset_count))
        # print('{"chart": "swa_update", "x": %d, "y": %.3f}' % (epoch + 1, swa_update_count))

        if epoch - epoch_of_last_improval >= train_abort_epochs_without_improval:
            print("early abort")
            break

    optim_summary_writer.close()
    train_summary_writer.close()
    val_summary_writer.close()
    # val_swa_summary_writer.close()

    train_end_time = time.time()
    print()
    print("Train time: %s" % str(datetime.timedelta(seconds=train_end_time - train_start_time)))

    eval_start_time = time.time()

    print()
    print("evaluation of the training model")
    model.load_state_dict(torch.load("{}/model.pth".format(output_dir), map_location=device))
    mask_threshold_global, mask_threshold_per_cc = analyze(model, train_data.val_set_df)

    eval_end_time = time.time()
    print()
    print("Eval time: %s" % str(datetime.timedelta(seconds=eval_end_time - eval_start_time)))

    print()
    print("submission preparation")

    submission_start_time = time.time()

    test_data = TestData(input_dir)

    test_set = TestDataset(test_data.df, image_size_target)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    test_data.df["predictions"] = predict(model, test_data_loader)
    test_data.df["prediction_masks"] = [np.int32(p > mask_threshold_global) for p in test_data.df.predictions]

    test_data.df["predictions_cc"] = test_data.df.prediction_masks.map(calculate_coverage_class)
    test_data.df["prediction_masks_cc"] = [np.int32(p > mask_threshold_per_cc[cc]) for p, cc in
                                           zip(test_data.df.predictions, test_data.df.predictions_cc)]

    test_data.df["prediction_masks_crf"] = crf_batch(test_data.df.images, test_data.df.prediction_masks)

    write_submission(test_data.df, "prediction_masks", "{}/{}".format(output_dir, "submission.csv"))
    write_submission(test_data.df, "prediction_masks_cc", "{}/{}".format(output_dir, "submission_cc.csv"))
    write_submission(test_data.df, "prediction_masks_crf", "{}/{}".format(output_dir, "submission_crf.csv"))

    submission_end_time = time.time()
    print()
    print("Submission time: %s" % str(datetime.timedelta(seconds=submission_end_time - submission_start_time)))


if __name__ == "__main__":
    main()
