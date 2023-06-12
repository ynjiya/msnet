import argparse
import os
import pathlib
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter
from dataset.msd import get_datasets
# from vtunet.vision_transformer import VTUNet as ViT_seg
from model import MSNet 
from config import *
from utils import reload_ckpt_bis, save_args, generate_segmentations
from logger import msnetLogger
from datetime import datetime

parser = argparse.ArgumentParser(description='MSNET MSD Evaluation')
parser.add_argument('-w', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('-nc', '--num_classes', type=int, default=3, 
                    help='output channel of network')
parser.add_argument('--input', type=str, default=EVAL_DATA_FOLDER_IN, metavar="FOLDER",
                    help='path to input data folder')
parser.add_argument('--output', type=str, default=EVAL_DATA_FOLDER_OUT, metavar="FOLDER",
                    help='path to output data folder')
parser.add_argument('--model', type=str, default=TRAINED_MODEL, 
                    metavar="FILE", help='path to trained model file')
parser.add_argument('--cfg', type=str, default="configs/msnet_base.yaml", metavar="FILE",
                    help='path to model config file')

device = torch.device("cpu")

def main(args):
    # output folder paths
    args.output_folder = pathlib.Path(f"./{args.output}")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.output_folder / "output"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.output_folder = args.output_folder.resolve()
    save_args(args)
    # args.checkpoint = pathlib.Path(f"./train_results/model3_200/model_best.pth.tar") 
    args.checkpoint = pathlib.Path(f"{args.model}") 


    timestamp = datetime.now()
    log_file = os.path.join(str(args.output_folder), "evaluation_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                        (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                        timestamp.second))
    args.logger = msnetLogger(log_file)

    # init model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = MSNet(num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda()
    model.load_from(yaml_cfg.get("MODEL").get("PRETRAIN_CKPT"))
    model = model.cuda()

    # get dataloader
    bench_dataset = get_datasets(on="eval", evalpath=args.input)
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)
    args.logger.print_to_log_file("Evaluation dataset number of batch:", len(bench_loader))
    args.logger.print_to_log_file("start inference now!")

    # load trained model from .pth.tar 
    reload_ckpt_bis(f'{args.checkpoint}', model, device)

    # get results
    generate_segmentations(bench_loader, model, args)


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
