import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset import TrainData, TrainDataset
from losses import BCELovaszLoss
from metrics import precision_batch
from models import AlbuNet34
from utils import moving_parameter_average, adjust_learning_rate, freeze, unfreeze

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def evaluate(model, data_loader, criterion):
    loss_sum = 0.0
    precision_sum = 0.0
    step_count = 0

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            images, masks, mask_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            prediction_logits = model(images)
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
    epochs_to_train = 64
    bce_loss_weight_gamma = 0.98
    clr_base_lr = 0.0001
    clr_max_lr = 0.001
    clr_cycle_epochs = 4
    clr_scale_factor = 1.1
    clr_scale_fn = lambda x: 1.0 / (clr_scale_factor ** (x - 1))
    swa_start_epoch = 20
    swa_cycle_epochs = 4
    train_reset_epochs_without_improval = 10
    train_abort_epochs_without_improval = 20

    train_data = TrainData(input_dir)

    train_set = TrainDataset(train_data.train_set_df, image_size_target, augment=True)
    train_set_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = TrainDataset(train_data.val_set_df, image_size_target, augment=False)
    val_set_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = AlbuNet34(num_filters=32, pretrained=True, is_deconv=True).to(device)
    # model.load_state_dict(torch.load("/storage/model.pth", map_location=device))
    model_freezed_layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5]
    for layer in model_freezed_layers:
        freeze(layer)

    swa_model = AlbuNet34(num_filters=32, pretrained=True, is_deconv=True).to(device)
    swa_model.load_state_dict(model.state_dict())
    swa_model_freezed_layers = [swa_model.conv1, swa_model.conv2, swa_model.conv3, swa_model.conv4, swa_model.conv5]
    for layer in swa_model_freezed_layers:
        freeze(layer)

    print("train_set_samples: %d, val_set_samples: %d" % (len(train_set), len(val_set)))

    global_val_precision_overall_avg = float("-inf")
    global_val_precision_best_avg = float("-inf")
    global_val_precision_swa_best_avg = float("-inf")

    epoch_iterations = len(train_set) // batch_size
    clr_step_size = (clr_cycle_epochs // 2) * epoch_iterations

    optimizer = optim.Adam(model.parameters(), lr=clr_base_lr)

    train_summary_writer = SummaryWriter(log_dir="{}/logs/train".format(output_dir))
    val_summary_writer = SummaryWriter(log_dir="{}/logs/val".format(output_dir))
    val_swa_summary_writer = SummaryWriter(log_dir="{}/logs/val_swa".format(output_dir))

    clr_iterations = 0
    swa_update_count = 0
    batch_count = 0
    epoch_since_reset = 0
    epoch_of_last_improval = 0

    print('{"chart": "precision", "axis": "epoch"}')

    for epoch in range(epochs_to_train):
        epoch_start_time = time.time()

        bce_loss_weight = bce_loss_weight_gamma ** epoch_since_reset
        criterion = BCELovaszLoss(bce_loss_weight)

        train_loss_sum = 0.0
        train_precision_sum = 0.0
        train_step_count = 0
        for _, batch in enumerate(train_set_data_loader):
            images, masks, mask_weights = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            clr_cycle = np.floor(1 + clr_iterations / (2 * clr_step_size))
            clr_x = np.abs(clr_iterations / clr_step_size - 2 * clr_cycle + 1)
            lr = clr_base_lr + (clr_max_lr - clr_base_lr) * np.maximum(0, (1 - clr_x)) * clr_scale_fn(clr_cycle)

            adjust_learning_rate(optimizer, lr)

            optimizer.zero_grad()
            prediction_logits = model(images)
            predictions = torch.sigmoid(prediction_logits)
            criterion.weight = mask_weights
            loss = criterion(prediction_logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_precision_sum += np.mean(precision_batch(predictions, masks))
            clr_iterations += 1
            train_step_count += 1
            batch_count += 1

            train_summary_writer.add_scalar("lr", lr, batch_count + 1)

        train_loss_avg = train_loss_sum / train_step_count
        train_precision_avg = train_precision_sum / train_step_count
        val_loss_avg, val_precision_avg = evaluate(model, val_set_data_loader, criterion)

        model_improved = val_precision_avg > global_val_precision_best_avg
        ckpt_saved = False
        if model_improved:
            torch.save(model.state_dict(), "{}/model.pth".format(output_dir))
            global_val_precision_best_avg = val_precision_avg
            ckpt_saved = True

        swa_updated = False
        if epoch + 1 >= swa_start_epoch and (model_improved or (epoch_since_reset + 1) % swa_cycle_epochs == 0):
            swa_update_count += 1
            moving_parameter_average(swa_model, model, 1.0 / swa_update_count)
            swa_updated = True

        val_loss_swa_avg, val_precision_swa_avg = evaluate(swa_model, val_set_data_loader, criterion)

        swa_model_improved = val_precision_swa_avg > global_val_precision_swa_best_avg
        swa_ckpt_saved = False
        if swa_model_improved:
            torch.save(swa_model.state_dict(), "{}/swa_model.pth".format(output_dir))
            global_val_precision_swa_best_avg = val_precision_swa_avg
            swa_ckpt_saved = True

        epoch_end_time = time.time()
        epoch_duration_time = epoch_end_time - epoch_start_time

        train_summary_writer.add_scalar("loss", train_loss_avg, epoch + 1)
        train_summary_writer.add_scalar("precision", train_precision_avg, epoch + 1)

        val_summary_writer.add_scalar("loss", val_loss_avg, epoch + 1)
        val_summary_writer.add_scalar("precision", val_precision_avg, epoch + 1)

        if swa_updated:
            val_swa_summary_writer.add_scalar("loss", val_loss_swa_avg, epoch + 1)
            val_swa_summary_writer.add_scalar("precision", val_precision_swa_avg, epoch + 1)

        if global_val_precision_best_avg > global_val_precision_overall_avg \
                or global_val_precision_swa_best_avg > global_val_precision_overall_avg:
            global_val_precision_overall_avg = max(global_val_precision_best_avg, global_val_precision_swa_best_avg)
            epoch_of_last_improval = epoch

        if min(epoch - epoch_of_last_improval, epoch_since_reset) >= train_reset_epochs_without_improval:
            clr_iterations = 0
            epoch_since_reset = 0
            if len(model_freezed_layers) > 0:
                unfreeze(model_freezed_layers.pop())
            if len(swa_model_freezed_layers) > 0:
                unfreeze(swa_model_freezed_layers.pop())
            trainig_reset = True
        else:
            epoch_since_reset += 1
            trainig_reset = False

        print(
            "[%03d/%03d] %ds, lr: %.6f, loss: %.3f, val_loss: %.3f|%.3f, prec: %.3f, val_prec: %.3f|%.3f, swa: %d, ckpt: %d|%d, rst: %d" % (
                epoch + 1,
                epochs_to_train,
                epoch_duration_time,
                lr,
                train_loss_avg,
                val_loss_avg,
                val_loss_swa_avg,
                train_precision_avg,
                val_precision_avg,
                val_precision_swa_avg,
                int(swa_updated),
                int(ckpt_saved),
                int(swa_ckpt_saved),
                int(trainig_reset)),
            flush=True)

        print('{"chart": "precision", "x": %d, "y": %.3f}' % (epoch + 1, global_val_precision_overall_avg))

        if min(epoch - epoch_of_last_improval, epoch_since_reset) >= train_abort_epochs_without_improval:
            print("early abort")
            break

    train_summary_writer.close()
    val_summary_writer.close()
    val_swa_summary_writer.close()


if __name__ == "__main__":
    main()
