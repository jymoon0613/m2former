import logging
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.mvit_model import MViT, smooth_CE
from config.defaults import get_cfg

from utils.data_utils import get_loader

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def setup(args):
    # Prepare model
    # config = CONFIGS[args.model_type]
    # config.split = args.split
    # config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089
    elif args.dataset == "air":
        num_classes = 100

    cfg = get_cfg()

    cfg.merge_from_file('config/MVITv2_B.yaml')

    cfg.MVIT.CLS_EMBED_ON = True
    cfg.MODEL.NUM_CLASSES = 19168

    model = MViT(cfg)
    
    model.num_classes = num_classes

    cfg.DATA.TRAIN_CROP_SIZE = 448
    cfg.DATA.TEST_CROP_SIZE = 448
    cfg.MODEL.NUM_CLASSES = num_classes

    base = MViT(cfg)

    model.head1 = base.head1
    model.head2 = base.head2
    model.head3 = base.head3
    model.head4 = base.head4
    model.head5 = base.head5

    for b, z in zip(model.blocks, base.blocks):
        b.attn.rel_pos_h = z.attn.rel_pos_h
        b.attn.rel_pos_w = z.attn.rel_pos_w
        
    check = torch.load(args.pretrained_model)['model']

    model.load_state_dict(check)
    
    model.to(args.device)
    model = torch.nn.DataParallel(model)
    num_params = count_parameters(model)

    print("Training parameters: %s" % args)
    print("Total Parameter: \t%2.1fM" % num_params)
    
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def valid(args, model):
    # Validation!
    args.train_batch_size = 1
    args.eval_batch_size = args.batch_size

    # Prepare dataset
    _, test_loader = get_loader(args)

    print("***** Running Validation *****")
    print("  Num steps = %d" % len(test_loader))
    print("  Batch size = %d" % args.eval_batch_size)

    model.eval()
    
    all_label = []
    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_preds4 = []
    all_preds5 = []
    all_preds6 = []
    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            x1, x2, x3, x4, x5 = model(x)
            x1 = x1.reshape(y.shape[0],-1)
            x2 = x2.reshape(y.shape[0],-1)
            x3 = x3.reshape(y.shape[0],-1)
            x4 = x4.reshape(y.shape[0],-1)
            x5 = x5.reshape(y.shape[0],-1)

            preds1 = torch.argmax(x1, dim=-1)
            preds2 = torch.argmax(x2, dim=-1)
            preds3 = torch.argmax(x3, dim=-1)
            preds4 = torch.argmax(x4, dim=-1)
            preds5 = torch.argmax(x5, dim=-1)
            preds6 = torch.argmax((x1 + x2 + x3 + x4 + x5), 1)

        if len(all_preds1) == 0:
            all_preds1.append(preds1.detach().cpu().numpy())
            all_preds2.append(preds2.detach().cpu().numpy())
            all_preds3.append(preds3.detach().cpu().numpy())
            all_preds4.append(preds4.detach().cpu().numpy())
            all_preds5.append(preds5.detach().cpu().numpy())
            all_preds6.append(preds6.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds1[0] = np.append(
                all_preds1[0], preds1.detach().cpu().numpy(), axis=0
            )
            all_preds2[0] = np.append(
                all_preds2[0], preds2.detach().cpu().numpy(), axis=0
            )
            all_preds3[0] = np.append(
                all_preds3[0], preds3.detach().cpu().numpy(), axis=0
            )
            all_preds4[0] = np.append(
                all_preds4[0], preds4.detach().cpu().numpy(), axis=0
            )
            all_preds5[0] = np.append(
                all_preds5[0], preds5.detach().cpu().numpy(), axis=0
            )
            all_preds6[0] = np.append(
                all_preds6[0], preds6.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            
    all_label = all_label[0]    
    all_preds1 = all_preds1[0]
    all_preds2 = all_preds2[0]
    all_preds3 = all_preds3[0]
    all_preds4 = all_preds4[0]
    all_preds5 = all_preds5[0]
    all_preds6 = all_preds6[0]
    val_accuracy1 = simple_accuracy(all_preds1, all_label)
    val_accuracy2 = simple_accuracy(all_preds2, all_label)
    val_accuracy3 = simple_accuracy(all_preds3, all_label)
    val_accuracy4 = simple_accuracy(all_preds4, all_label)
    val_accuracy5 = simple_accuracy(all_preds5, all_label)
    val_accuracy6 = simple_accuracy(all_preds6, all_label)

    print("\n")
    print("Validation Results")
    print("ValAcc1: %2.5f ValAcc2: %2.5f ValAcc3: %2.5f ValAcc4: %2.5f ValAccCon: %2.5f ValAccTot: %2.5f" % (val_accuracy1, val_accuracy2, val_accuracy3, val_accuracy4, val_accuracy5, val_accuracy6))
        
    return val_accuracy6

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    
    parser.add_argument("--gpu_ids", required=True,
                        help="GPU IDs.")
    parser.add_argument("--pretrained_model", required=True,
                        help="Where to search for pretrained models.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017", "air"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='../Dataset/01.CUB-200-2011')
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for eval.")

    args = parser.parse_args()

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    args, model = setup(args)

    valid(args, model)

if __name__ == "__main__":
    main()