import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from fvcore.nn import FlopCountAnalysis


from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.apps import DecathlonDataset, MedNISTDataset, download_and_extract

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.msd import get_datasets
from loss import EDiceLoss
from loss import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, save_checkpoint_latest,  save_checkpoint_every, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_pred, post_label, dice_metric, \
    dice_metric_batch, generate_segmentations_monai
from vtunet.vision_transformer import VTUNet as ViT_seg
from datetime import datetime

from logger import msnetLogger

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='VTUNET MSD 2021 Training with AdamW')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
                    
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/msnet_base.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

device = torch.device("cuda")

def main(args):
    # setup

    args.exp_name = "logs_base"
    args.output_folder = pathlib.Path(f"./runs/{args.exp_name}/model1_2w")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.output_folder / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.output_folder = args.output_folder.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.output_folder))
    args.checkpoint_folder = pathlib.Path(f"./runs/{args.exp_name}/model1_2w")

    # Logging
    timestamp = datetime.now()
    log_file = os.path.join(str(args.output_folder), "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                            (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                            timestamp.second))
    logger = msnetLogger(log_file)

    # Create model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = get_config(args)
    model = ViT_seg(config, num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda()
    model.load_from(config)
    logger.print_to_log_file(f"Created а model")
    logger.print_to_log_file(f"total number of trainable parameters {count_parameters(model)}")

    start_epoch = 0
    if args.resume:
        args.checkpoint = args.checkpoint_folder / "model_best.pth.tar"
        start_epoch = reload_ckpt_bis(args.checkpoint, model)
        logger.print_to_log_file(f"resuming the training from epoch {start_epoch}")
        

    model = model.cuda()
    input = torch.rand(1, 4, 128, 128, 128).cuda()
    flops = FlopCountAnalysis(model, input)
    logger.print_to_log_file(f"total number of flops {flops.total()}")

    model_file = args.output_folder / "model.txt"
    with model_file.open("w") as f:
        print(model, file=f)

    criterion = EDiceLoss().cuda()
    criterian_val = EDiceLoss().cuda()
    # criterian_val = EDiceLoss_Val().cuda()
    metric = criterian_val.metric
    logger.print_to_log_file(metric)
    params = model.parameters()

    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    full_train_dataset, l_val_dataset, bench_dataset = get_datasets(args.seed, fold_number=args.fold)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=True,
                                             pin_memory=True, num_workers=args.workers)
    
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    # train_loader, val_loader = get_training_dataloaders(args.batch_size, args.workers)
    # bench_loader = get_test_dataloader(args.batch_size, args.workers)

    logger.print_to_log_file("Train dataset number of batch:", len(train_loader))
    logger.print_to_log_file("Val dataset number of batch:", len(val_loader))
    logger.print_to_log_file("Bench Test dataset number of batch:", len(bench_loader))

    # Actual Train loop
    best_1 = 0.0
    patients_perf = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    logger.print_to_log_file("start training now!")

    for epoch in range(start_epoch, args.epochs):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

            mode = "train" 
            batch_per_epoch = len(train_loader)
            logger.print_to_log_file("Batches per epoch: ", batch_per_epoch)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]", 
                logger=logger)

            end = time.perf_counter()
            metrics = []
            for i, batch in enumerate(zip(train_loader)):
            # for batch in train_loader:
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                # inputs, labels = (
                #     batch[0]["image"].to(device),
                #     batch[0]["label"].to(device),
                # )

                inputs, labels = Variable(batch[0]["image"].float()).cuda(), Variable(batch[0]["label"].float()).cuda()
                # inputs, labels = Variable(inputs), Variable(labels)
                # inputs, labels = inputs.cuda(), labels.cuda()

                # logger.print_to_log_file("inputs.shape", inputs.shape)
                # logger.print_to_log_file("label.shape", labels.shape)

                optimizer.zero_grad()
                outputs = model(inputs)
                # logger.print_to_log_file("outputs.shape", outputs.shape)

                loss_ = criterion(outputs, labels)

                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    logger.print_to_log_file("NaN in model loss !")

                # compute gradient and do SGD step
                loss_.backward()
                optimizer.step()

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)

                if scheduler is not None:
                    scheduler.step()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)

            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)

            te = time.perf_counter()
            logger.print_to_log_file(f"Train Epoch done in {te - ts} s")
            model_dict = model.state_dict()
            save_checkpoint_latest(
                dict(
                    epoch=epoch,
                    state_dict=model_dict,
                    optimizer=optimizer.state_dict(),
                ),
                save_folder=args.output_folder, )
            torch.cuda.empty_cache()

            if (epoch + 1) % 20 == 0:
                save_checkpoint_every(
                dict(
                    epoch=epoch,
                    state_dict=model_dict,
                    optimizer=optimizer.state_dict(),
                ),
                save_folder=args.output_folder, epoch=epoch + 1)


            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                validation_loss_1, validation_dice = step(val_loader, model, criterian_val, metric, epoch, t_writer_1,
                                                          save_folder=args.output_folder,
                                                          patients_perf=patients_perf, logger=logger)

                t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)
                t_writer_1.add_scalar(f"SummaryDice", validation_dice, epoch)

                if validation_dice > best_1:
                    logger.print_to_log_file(f"Saving the model with DSC {validation_dice}")
                    best_1 = validation_dice
                    model_dict = model.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.output_folder, )

                ts = time.perf_counter()
                logger.print_to_log_file(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            logger.print_to_log_file("Stopping training loop, doing benchmark")
            generate_segmentations_monai(bench_loader, model, t_writer_1, args)
            break
    generate_segmentations_monai(bench_loader, model, t_writer_1, args)


def step(data_loader, model, criterion: EDiceLoss_Val, metric, epoch, writer, logger, save_folder=None, patients_perf=None):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]",
        logger=logger)

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["id"]

        model.eval()
        with torch.no_grad():
            # val_inputs, val_labels = (
            #     val_data["image"].cuda(),
            #     val_data["label"].cuda(),
            # )

            val_inputs, val_labels = Variable(val_data["image"].float()).cuda(), Variable(val_data["label"].float()).cuda()

            val_outputs = inference(val_inputs, model)
            # val_outputs_1 = post_pred(val_outputs)
            val_outputs_1 = [post_pred(i) for i in decollate_batch(val_outputs)]

            # logger.print_to_log_file("val_outputs.shape ", val_outputs.shape)
            # logger.print_to_log_file("val_outputs_1.shape ", val_outputs.shape)

            # # val_labels_1 = post_label(val_labels)
            # logger.print_to_log_file("val_labels.shape ", val_labels.shape)
            # logger.print_to_log_file("val_labels_1.shape ", val_labels.shape)

            loss_ = criterion(val_outputs, val_labels)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if patients_perf is not None:
            patients_perf.append(
                # dict(epoch=epoch, split=mode, loss=loss_.item())
                dict(id=patient_id, epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            logger.print_to_log_file("NaN in model loss !!")

        metric_ = metric(val_outputs, val_labels)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, logger, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    return losses.avg, dice_values


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
