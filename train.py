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

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)

    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    # config = CONFIGS[args.model_type]
    # config.split = args.split
    # config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "nabirds":
        num_classes = 555

    cfg = get_cfg()

    cfg.merge_from_file('config/MVITv2_B.yaml')

    cfg.MVIT.CLS_EMBED_ON = True
    cfg.MODEL.NUM_CLASSES = 19168

    model = MViT(cfg)
    
    model.num_classes = num_classes

    model.load_state_dict(torch.load(args.pretrained_dir)['model_state'], strict=False)

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
    
    model.to(args.device)
    model = torch.nn.DataParallel(model)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

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
            
            eval_loss = smooth_CE(x1, y, 0.6) + smooth_CE(x2, y, 0.7) + smooth_CE(x3, y, 0.8) + smooth_CE(x4, y, 0.9) + smooth_CE(x5, y, 1)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())

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

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("ValAcc1: %2.5f ValAcc2: %2.5f ValAcc3: %2.5f ValAcc4: %2.5f ValAccCon: %2.5f ValAccTot: %2.5f" % (val_accuracy1, val_accuracy2, val_accuracy3, val_accuracy4, val_accuracy5, val_accuracy6))

    writer.add_scalar("test/accuracy", scalar_value=val_accuracy6, global_step=global_step)
        
    return val_accuracy6

def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.eval_batch_size = args.eval_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    start_time = time.time()
    while True:
        model.train()
        all_label = []
        all_preds1 = []
        all_preds2 = []
        all_preds3 = []
        all_preds4 = []
        all_preds5 = []
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch

            loss1, x1, loss2, x2, loss3, x3, loss4, x4, loss5, x5 = model(x, y)
            loss = loss1.mean() + loss2.mean() + loss3.mean() + loss4.mean() + loss5.mean()

            preds1 = torch.argmax(x1, dim=-1)
            preds2 = torch.argmax(x2, dim=-1)
            preds3 = torch.argmax(x3, dim=-1)
            preds4 = torch.argmax(x4, dim=-1)
            preds5 = torch.argmax(x5, dim=-1)

            if len(all_preds1) == 0:
                all_preds1.append(preds1.detach().cpu().numpy())
                all_preds2.append(preds2.detach().cpu().numpy())
                all_preds3.append(preds3.detach().cpu().numpy())
                all_preds4.append(preds4.detach().cpu().numpy())
                all_preds5.append(preds5.detach().cpu().numpy())
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
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                
                if global_step % args.eval_every == 0:
                    with torch.no_grad():
                        accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    logger.info("best accuracy so far: %f" % best_acc)
                    model.train()

                if global_step % t_total == 0:
                    break
                
        all_label = all_label[0]
        all_preds1 = all_preds1[0]
        all_preds2 = all_preds2[0]
        all_preds3 = all_preds3[0]
        all_preds4 = all_preds4[0]
        all_preds5 = all_preds5[0]
        train_accuracy1 = simple_accuracy(all_preds1, all_label)
        train_accuracy2 = simple_accuracy(all_preds2, all_label)
        train_accuracy3 = simple_accuracy(all_preds3, all_label)
        train_accuracy4 = simple_accuracy(all_preds4, all_label)
        train_accuracy5 = simple_accuracy(all_preds5, all_label)

        logger.info("TrainAcc1: %f TrainAcc2: %f TrainAcc3: %f TrainAcc4: %f TrainAccCon: %f" % (train_accuracy1, train_accuracy2, train_accuracy3, train_accuracy4, train_accuracy5))
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))
    
def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", type=str, default='sample_run',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--gpu_ids", required=True,
                        help="GPU IDs.")
    
    parser.add_argument("--dataset", choices=["CUB_200_2011", "nabirds"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='../Dataset/01.CUB-200-2011')
    parser.add_argument("--pretrained_dir", type=str, default='weights/MViTv2_B_in21k.pyth',
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s" % (args.device, args.n_gpu))

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)

if __name__ == "__main__":
    main()