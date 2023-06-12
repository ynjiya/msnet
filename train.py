import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import yaml
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
from monai.data import decollate_batch

from config import *
from model import MSNet 
from logger import msnetLogger
from dataset.msd import get_datasets
from loss import EDiceLoss
from loss import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, save_checkpoint_latest,  save_checkpoint_every, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args, inference, post_pred, dice_metric, \
    dice_metric_batch, generate_segmentations_metrics

parser = argparse.ArgumentParser(description='MSNET MSD Training')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', 
                    help='number of total epochs to run (default: 200)')
parser.add_argument('-b', '--batch-size', default=3, type=int, metavar='N', 
                    help='batch size (default: 3)')
parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, metavar='LR', 
                    help='initial learning rate (default: 1e-4)', dest='lr')
parser.add_argument('-wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('-w', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--val', default=1, type=int, 
                    help="how often to perform validation step (default: 1)")
parser.add_argument('-nc', '--num_classes', type=int, default=3,
                    help='output channel of network (default: 3)')
parser.add_argument('--resume', default=False, type=bool, 
                    help='resume from the latest saved checkpoint (default: False)')
parser.add_argument('--input', type=str, default=MAIN_DATA_FOLDER_MSD, metavar="FOLDER",
                    help='path to input data folder')
parser.add_argument('--cfg', type=str, default="configs/msnet_base.yaml", metavar="FILE",
                    help='path to model config file')

device = torch.device("cuda")
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

def main(args):
    # output folder paths
    args.output_folder = pathlib.Path(f"./train_results/model")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.output_folder / "output"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.output_folder = args.output_folder.resolve()
    save_args(args)
    t_writer = SummaryWriter(str(args.output_folder))
    args.checkpoint_folder = pathlib.Path(f"./train_results/model")

    # logger init
    timestamp = datetime.now()
    log_file = os.path.join(str(args.output_folder), "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                            (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                            timestamp.second))
    logger = msnetLogger(log_file)

    # init and load model from yaml_cfg file
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = MSNet(num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda()
    model.load_from(yaml_cfg.get("MODEL").get("PRETRAIN_CKPT"))
    logger.print_to_log_file(f"Created а model")
    logger.print_to_log_file(f"Total number of trainable parameters {count_parameters(model)}")

    # calculate total number of FLOPS
    model = model.cuda()
    input = torch.rand(1, 4, 128, 128, 128).cuda()
    flops = FlopCountAnalysis(model, input)
    logger.print_to_log_file(f"Total number of flops {flops.total()}")

    # resume training
    start_epoch = 0
    if args.resume:
        args.checkpoint = args.checkpoint_folder / "model_best.pth.tar"
        start_epoch = reload_ckpt_bis(args.checkpoint, model)
        logger.print_to_log_file(f"Resuming the training from epoch {start_epoch}")

    # save model architecture in txt file
    model_file = args.output_folder / "model.txt"
    with model_file.open("w") as f:
        print(model, file=f)

    # init training hyperparams
    criterion = EDiceLoss().cuda()
    criterian_val = EDiceLoss().cuda()
    metric = criterian_val.metric
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # get dataloaders
    train_dataset, val_dataset, bench_dataset = get_datasets(on="train", mainpath=args.input)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True,
                                             pin_memory=True, num_workers=args.workers)
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    logger.print_to_log_file("Train dataset number of batch:", len(train_loader))
    logger.print_to_log_file("Val dataset number of batch:", len(val_loader))
    logger.print_to_log_file("Benchtest dataset number of batch:", len(bench_loader))

    # training loop
    best = 0.0
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
            for i, batch in enumerate(zip(train_loader)):
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                inputs, labels = Variable(batch[0]["image"].float()).cuda(), Variable(batch[0]["label"].float()).cuda()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_ = criterion(outputs, labels)

                t_writer.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    logger.print_to_log_file("NaN in model loss !")

                # compute gradient and do Adam step
                loss_.backward()
                optimizer.step()

                t_writer.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)
                if scheduler is not None:
                    scheduler.step()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                progress.display(i)

            t_writer.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)
            te = time.perf_counter()
            logger.print_to_log_file(f"Train Epoch done in {te - ts} s")

            # save state_dict as the latest
            model_dict = model.state_dict()
            save_checkpoint_latest(
                dict(
                    epoch=epoch,
                    state_dict=model_dict,
                    optimizer=optimizer.state_dict(),
                ),
                save_folder=args.output_folder, )
            torch.cuda.empty_cache()

            # save every 20th state_dict
            if (epoch + 1) % 20 == 0:
                save_checkpoint_every(
                dict(
                    epoch=epoch,
                    state_dict=model_dict,
                    optimizer=optimizer.state_dict(),
                ),
                save_folder=args.output_folder, epoch=epoch + 1)


            # validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                validation_loss_1, validation_dice = validation_step(
                    val_loader, model, criterian_val, metric, epoch, t_writer,
                    save_folder=args.output_folder, logger=logger
                    )

                t_writer.add_scalar(f"SummaryLoss", validation_loss_1, epoch)
                t_writer.add_scalar(f"SummaryDice", validation_dice, epoch)

                if validation_dice > best:
                    logger.print_to_log_file(f"Saving the model with DSC {validation_dice}")
                    best = validation_dice
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
            generate_segmentations_metrics(bench_loader, model, t_writer, args)
            break
    generate_segmentations_metrics(bench_loader, model, t_writer, args)


def validation_step(data_loader, model, criterion: EDiceLoss_Val, metric, epoch, writer, logger, save_folder=None):
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

    for i, data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = Variable(data["image"].float()).cuda(), Variable(data["label"].float()).cuda()

            val_outputs = inference(val_inputs, model)
            val_outputs_processed = [post_pred(i) for i in decollate_batch(val_outputs)]

            loss_ = criterion(val_outputs, val_labels)
            dice_metric(y_pred=val_outputs_processed, y=val_labels)

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
        # display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, logger, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    return losses.avg, dice_values


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
