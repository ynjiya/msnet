import os
import pathlib
import pprint

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric 
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
    )
from numpy import logical_and as l_and, logical_not as l_not
from torch import distributed as dist
from skimage.metrics import structural_similarity as ssim
from medpy.metric import binary


def save_args(args):
    """Save parsed arguments to config file.
    """
    config = vars(args).copy()
    # del config['output_folder']
    # del config['seg_folder_1']
    config_file = args.output_folder / ("settings.yaml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)



def save_checkpoint(state: dict, save_folder: pathlib.Path):
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_best.pth.tar'
    torch.save(state, best_filename)

def save_checkpoint_latest(state: dict, save_folder: pathlib.Path):
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_latest.pth.tar'
    torch.save(state, best_filename)

def save_checkpoint_every(state: dict, save_folder: pathlib.Path, epoch: int):
    """Save Training state."""
    best_filename = f'{str(save_folder)}/model_{epoch}.pth.tar'
    torch.save(state, best_filename)   


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.print_to_log_file('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def reload_ckpt(args, model, optimizer, device=torch.device("cuda:0")):
    if os.path.isfile(args):
        print("=> loading checkpoint '{}'".format(args))
        checkpoint = torch.load(args, map_location=device)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args))


def reload_ckpt_bis(ckpt, model, device=torch.device("cuda:0")):
    if os.path.isfile(ckpt):
        print(f"=> loading checkpoint {ckpt}")
        try:
            checkpoint = torch.load(ckpt, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            return start_epoch
        except RuntimeError:
            # TO account for checkpoint from Alex nets
            print("Loading model Alex style")
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    else:
        raise ValueError(f"=> no checkpoint found at '{ckpt}'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(preds, targets, patient):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []
    avg_dice = 0
    avg_hd = 0
    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            ssim_m = np.nan

        else:
            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            ssim_m = ssim(preds[i], targets[i])

            dice = 2 * tp / (2 * tp + fp + fn)

        metrics[HAUSSDORF] = hd(preds[i], targets[i])
        avg_hd += metrics[HAUSSDORF]
        metrics[DICE] = dice
        avg_dice += metrics[DICE]
        metrics[SENS] = sens
        metrics[SPEC] = spec
        metrics[SSIM] = ssim_m
        pp.pprint(metrics)
        metrics_list.append(metrics)
    avg_hd = avg_hd/3
    avg_dice = avg_dice/3
    return metrics_list, avg_hd, avg_dice, 

def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return hd95
    else:
        return 0

def save_metrics(epoch, metrics, writer, current_epoch, logger, teacher=False, save_folder=None):
    metrics = list(zip(*metrics))
    # print(metrics)
    # TODO check if doing it directly to numpy work
    metrics = [torch.tensor(dice, device="cpu").numpy() for dice in metrics]
    # print(metrics)
    labels = ("ET", "TC", "WT")
    metrics = {key: value for key, value in zip(labels, metrics)}
    # print(metrics)
    fig, ax = plt.subplots()
    ax.set_title("Dice metrics")
    ax.boxplot(metrics.values(), labels=metrics.keys())
    ax.set_ylim(0, 1)
    writer.add_figure(f"val/plot", fig, global_step=epoch)
    logger.print_to_log_file(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
          [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    with open(f"{save_folder}/val{'_teacher' if teacher else ''}.txt", mode="a") as f:
        print(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()], file=f)
        logger.print_to_log_file(f"Epoch {current_epoch} :{'val' + '_teacher :' if teacher else 'Val :'}",
              [f"{key} : {np.nanmean(value)}" for key, value in metrics.items()])
    for key, value in metrics.items():
        tag = f"val{'_teacher' if teacher else ''}{''}/{key}_Dice"
        writer.add_scalar(tag, np.nanmean(value), global_step=epoch)


dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_pred = Compose(
    # [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    # [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.3)]
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

post_label = Compose([AsDiscrete(to_onehot=3)])

VAL_AMP = True


# define inference method
def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# for training, validation and benchtest
def generate_segmentations_metrics(data_loader, model, writer_1, args):
    metrics_list = []
    model.eval()

    avg_dice_all = 0
    avg_hd_all = 0
    avg_dice_et = avg_hd_et = 0
    avg_dice_tc = avg_hd_tc = 0
    avg_dice_wt = avg_hd_wt = 0

    for i, val_data in enumerate(data_loader):
        patient_id = val_data["id"][0]
        ref_path = val_data["seg_path"][0]
        crops_idx = val_data["crop_indexes"]
        val_inputs = val_data["image"].cuda()
        args.logger.print_to_log_file(f"Test case id: {patient_id}")

        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)

        with torch.no_grad():
            val_outputs = inference(val_inputs, model)
            val_outputs_processed = [post_pred(i) for i in decollate_batch(val_outputs)]

        segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs_processed[0]
        segs = segs[0].numpy() > 0

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))

        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 3
        labelmap[net] = 2
        labelmap[ed] = 1
        labelmap = sitk.GetImageFromArray(labelmap)

        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 3
        refmap_tc = np.logical_or(refmap_et, ref_seg == 2)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 1)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])

        patient_metric_list, avg_hd, avg_dice = calculate_metrics(segs, refmap, patient_id)
        args.logger.print_to_log_file(f"{patient_metric_list}")
        args.logger.print_to_log_file(f"avg_hd: {avg_hd}")
        args.logger.print_to_log_file(f"avg_dice: {avg_dice}")

        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)

        avg_hd_all += avg_hd
        avg_dice_all += avg_dice

        avg_hd_et += patient_metric_list[0]["haussdorf"]
        avg_dice_et += patient_metric_list[0]["dice"]

        avg_hd_tc += patient_metric_list[1]["haussdorf"]
        avg_dice_tc += patient_metric_list[1]["dice"]

        avg_hd_wt += patient_metric_list[2]["haussdorf"]
        avg_dice_wt += patient_metric_list[2]["dice"]

        args.logger.print_to_log_file(f"Writing {args.seg_folder}/BRATS_{'{:03d}'.format(patient_id)}.nii.gz \n")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/BRATS_{'{:03d}'.format(patient_id)}.nii.gz")

    loader_len = len(data_loader)

    avg_hd_et /= loader_len
    avg_dice_et /= loader_len

    avg_hd_tc /= loader_len
    avg_dice_tc /= loader_len

    avg_hd_wt /= loader_len
    avg_dice_wt /= loader_len

    avg_hd_all /= loader_len
    avg_dice_all /= loader_len

    args.logger.print_to_log_file(f"\n avg_dice_all {avg_dice_all}\n avg_hd_all {avg_hd_all}\n avg_dice_et {avg_dice_et}\n avg_dice_tc {avg_dice_tc}\n avg_dice_wt {avg_dice_wt}\n avg_hd_et {avg_hd_et}\n avg_hd_tc {avg_hd_tc}\n avg_hd_wt {avg_hd_wt}")

    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    writer_1.add_figure("benchmark/overlap_measures", overlap_figure)
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    writer_1.add_figure("benchmark/distance_measure", haussdorf_figure)
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()
    for metric, label_values in summary.items():
        for label, score in label_values.items():
            writer_1.add_scalar(f"benchmark_{metric}/{label}", score)
    df.to_csv((args.output_folder / 'results.csv'), index=False)


# for evaluation
def generate_segmentations(data_loader, model, args):
    model.eval()

    for i, val_data in enumerate(data_loader):
        patient_id = val_data["id"][0]
        crops_idx = val_data["crop_indexes"]
        og_size = val_data["og_size"]

        args.logger.print_to_log_file(f"Evaluation case id: {patient_id}")

        val_inputs = val_data["image"].cuda()
        # val_inputs = torch.tensor(val_data["image"], dtype=torch.float16)

        with torch.no_grad():
            val_outputs = inference(val_inputs, model)
            val_outputs_processed = [post_pred(i) for i in decollate_batch(val_outputs)]

        segs = torch.zeros((1, 3, og_size[1] ,og_size[2], og_size[3]))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs_processed[0]
        segs = segs[0].numpy() > 0

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))

        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 3
        labelmap[net] = 2
        labelmap[ed] = 1
        labelmap = sitk.GetImageFromArray(labelmap)

        args.logger.print_to_log_file(f"Writing {args.seg_folder}/BRATS_{'{:03d}'.format(patient_id)}.nii.gz \n")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/BRATS_{'{:03d}'.format(patient_id)}.nii.gz")


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
SSIM = "ssim"
METRICS = [HAUSSDORF, DICE, SENS, SPEC, SSIM]
