import logging
import argparse
import os
import time
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.mvit_model_visual import MViT, smooth_CE
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

def visualize(args, model):
    # Validation!
    args.train_batch_size = 1
    args.eval_batch_size = 1

    # Prepare dataset
    _, test_loader = get_loader(args)
    
    # os.makedirs('./bbox', exist_ok=True)
    # os.makedirs('./att1', exist_ok=True)
    # os.makedirs('./att2', exist_ok=True)
    # os.makedirs('./att3', exist_ok=True)
    # os.makedirs('./att4', exist_ok=True)
    
    p1 = 8
    p2 = 16
    p3 = 32
    p4 = 64

    l1 = 162
    l2 = 54
    l3 = 18
    l4 = 6

    print("***** Running Validation *****")
    print("  Num steps = %d" % len(test_loader))
    print("  Batch size = %d" % args.eval_batch_size)

    model.eval()

    for step, batch in enumerate(test_loader):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            inter, inds, inrs, weights = model(x)
            
#         mask1 = get_mask(x, p1, l1, inrs[0])
#         mask2 = get_mask(x, p2, l2, inrs[1])
#         mask3 = get_mask(x, p3, l3, inrs[2])
#         mask4 = get_mask(x, p4, l4, inrs[3])

#         v1, v2, v3, v4 = run_one_image_save(x, mask1, p1), run_one_image_save(x, mask2, p2), run_one_image_save(x, mask3, p3), run_one_image_save(x, mask4, p4)
#         sample = torch.cat((x.data, v4.data, v3.data, v2.data, v1.data), -1)
#         save_image(sample, "mask/{}.png".format(step), nrow=1, normalize=True)
            
#         bbox_img1 = draw_box(x[0].detach().cpu(), inds[0].detach().cpu(), p1)
#         bbox_img2 = draw_box(x[0].detach().cpu(), inds[1].detach().cpu(), p2)
#         bbox_img3 = draw_box(x[0].detach().cpu(), inds[2].detach().cpu(), p3)
#         bbox_img4 = draw_box(x[0].detach().cpu(), inds[3].detach().cpu(), p4)

#         bbox = Image.fromarray(np.concatenate([tensor_to_array(x[0].detach().cpu()), bbox_img4, bbox_img3, bbox_img2, bbox_img1], axis=1))
#         bbox.save("bbox/{}.png".format(step))

#         weight1 = weights[0][0].mean(dim=0)
#         weight12 = weight1[1:,l1:l1+l2].argsort(dim=1)
#         weight13 = weight1[1:,l2:l2+l3].argsort(dim=1)
#         weight14 = weight1[1:,l3:l3+l4].argsort(dim=1)

#         weight2 = weights[1][0].mean(dim=0)
#         weight21 = weight2[1:,:l1].argsort(dim=1)
#         weight23 = weight2[1:,l2:l2+l3].argsort(dim=1)
#         weight24 = weight2[1:,l3:l3+l4].argsort(dim=1)

#         weight3 = weights[2][0].mean(dim=0)
#         weight31 = weight3[1:,:l1].argsort(dim=1)
#         weight32 = weight3[1:,l1:l1+l2].argsort(dim=1)
#         weight34 = weight3[1:,l3:l3+l4].argsort(dim=1)

#         weight4 = weights[3][0].mean(dim=0)
#         weight41 = weight4[1:,:l1].argsort(dim=1)
#         weight42 = weight4[1:,l1:l1+l2].argsort(dim=1)
#         weight43 = weight4[1:,l2:l2+l3].argsort(dim=1)

#         imgp12 = draw_point(x[0].detach().cpu(), inds[0].detach().cpu(), inds[1].detach().cpu(), p1,  p2, weight12)
#         imgp13 = draw_point(x[0].detach().cpu(), inds[0].detach().cpu(), inds[2].detach().cpu(), p1,  p3, weight13)
#         imgp14 = draw_point(x[0].detach().cpu(), inds[0].detach().cpu(), inds[3].detach().cpu(), p1,  p4, weight14)

#         imgp21 = draw_point(x[0].detach().cpu(), inds[1].detach().cpu(), inds[0].detach().cpu(), p2,  p1, weight21)
#         imgp23 = draw_point(x[0].detach().cpu(), inds[1].detach().cpu(), inds[2].detach().cpu(), p2,  p3, weight23)
#         imgp24 = draw_point(x[0].detach().cpu(), inds[1].detach().cpu(), inds[3].detach().cpu(), p2,  p4, weight24)

#         imgp31 = draw_point(x[0].detach().cpu(), inds[2].detach().cpu(), inds[0].detach().cpu(), p3,  p1, weight31)
#         imgp32 = draw_point(x[0].detach().cpu(), inds[2].detach().cpu(), inds[1].detach().cpu(), p3,  p2, weight32)
#         imgp34 = draw_point(x[0].detach().cpu(), inds[2].detach().cpu(), inds[3].detach().cpu(), p3,  p4, weight34)

#         imgp41 = draw_point(x[0].detach().cpu(), inds[3].detach().cpu(), inds[0].detach().cpu(), p4,  p1, weight41)
#         imgp42 = draw_point(x[0].detach().cpu(), inds[3].detach().cpu(), inds[1].detach().cpu(), p4,  p2, weight42)
#         imgp43 = draw_point(x[0].detach().cpu(), inds[3].detach().cpu(), inds[2].detach().cpu(), p4,  p3, weight43)

#         imgp1 = np.concatenate([imgp12, imgp13[:,448:,:], imgp14[:,448:,:]], axis=1)
#         imgp2 = np.concatenate([imgp21, imgp23[:,448:,:], imgp24[:,448:,:]], axis=1)
#         imgp3 = np.concatenate([imgp31, imgp32[:,448:,:], imgp34[:,448:,:]], axis=1)
#         imgp4 = np.concatenate([imgp41, imgp42[:,448:,:], imgp43[:,448:,:]], axis=1)

#         imgp1 = Image.fromarray(imgp1)
#         imgp2 = Image.fromarray(imgp2)
#         imgp3 = Image.fromarray(imgp3)
#         imgp4 = Image.fromarray(imgp4)
#         imgp1.save("att1/{}.png".format(step))   
#         imgp2.save("att2/{}.png".format(step))   
#         imgp3.save("att3/{}.png".format(step))   
#         imgp4.save("att4/{}.png".format(step))
         
    return

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def run_one_image_save(x, mask, patch_size):
    
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)  # (N, H*W, p*p*3)
    mask = unpatchify(mask, patch_size).detach()  # 1 is removing, 0 is keeping
    
    # masked image
    im_masked = x * (1 - mask)
    
    return im_masked

def draw_box(img, index_tensor, patch_size):
    
    C, H, W = img.shape
    
    img = tensor_to_array(img)

    image = img.copy()
    
    im = img.copy()
    
    image = image / 1.5

    image = image.astype(np.uint8)
    
    alpha = 0.6
    
    row = index_tensor // int(H / patch_size)
    row = row[0]
    
    col = index_tensor % int(H / patch_size)
    col = col[0]
    
    for i in range(len(row)):
        x1 = row[i].item() * patch_size
        y1 = col[i].item() * patch_size
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        cv2.rectangle(image, (y1, x1), (y2, x2), (255,0,0), -1)
        
    image = cv2.addWeighted(image, alpha, im, 1 - alpha, 0)

    return image

def tensor_to_array(tensor_image):
    
    img = (tensor_image.permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    
    img = img * 255   

    img = img.astype(np.uint8)

    return img

def get_mask(x, patch_size, len_keep, ids_restore):
    
    B, C, H, W = x.shape
    
    L = (H / patch_size) * (W / patch_size)
    
    mask = torch.ones([B, int(L)], device=x.device)
    
    mask[:, :len_keep] = 0
    
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return mask

def draw_point(img, q_index_tensor, k_index_tensor, q_patch_size, k_patch_size, score_tensor): 
    
    C, H, W = img.shape
    
    img = tensor_to_array(img)
    
    q_row = q_index_tensor // int(H / q_patch_size)
    q_row = q_row[0]
    
    q_col = q_index_tensor % int(H / q_patch_size)
    q_col = q_col[0]
    
    k_row = k_index_tensor // int(H / k_patch_size)
    k_row = k_row[0]
    
    k_col = k_index_tensor % int(H / k_patch_size)
    k_col = k_col[0]
    
    total = []
    
    im = img.copy()
    alpha = 0.8
    
    for i in range(len(q_row)):
        
        part = []
        
        q_x1 = q_row[i].item() * q_patch_size
        q_y1 = q_col[i].item() * q_patch_size
        q_x2 = q_x1 + q_patch_size
        q_y2 = q_y1 + q_patch_size
        
        # q_xc, q_yc = int((q_x2 + q_x1) / 2), int((q_y2 + q_y1) / 2)
        
        img_q = img.copy()
        
        img_q = img_q / 1.5
        
        img_q = img_q.astype(np.uint8)
        
        cv2.rectangle(img_q, (q_y1, q_x1), (q_y2, q_x2), (255,0,0), -1)
        
        img_q = cv2.addWeighted(img_q, alpha, im, 1 - alpha, 0)
        
        part.append(img_q)
        
        score = score_tensor[i,:]
        
        ref = torch.linspace(0.1, 0.9, len(score))
        
        img_kk = img.copy()

        img_kk = img_kk / 1.5

        img_kk = img_kk.astype(np.uint8)
        
        for j in range(len(k_row)):
            
            img_k = img.copy()
            
            img_k = img_k / 1.5

            img_k = img_k.astype(np.uint8)
            
            k_x1 = k_row[j].item() * k_patch_size
            k_y1 = k_col[j].item() * k_patch_size
            k_x2 = k_x1 + k_patch_size
            k_y2 = k_y1 + k_patch_size
            
            cv2.rectangle(img_k, (k_y1, k_x1), (k_y2, k_x2), (255,102,0), -1)
            
            beta = ref[score[j]].item()
            
            img_k = cv2.addWeighted(img_k, float(beta), im, 1 - float(beta), 0)
            
            img_kk[k_x1:k_x2, k_y1:k_y2, :] = img_k[k_x1:k_x2, k_y1:k_y2, :]
        
        part.append(img_kk)
            
        part = np.concatenate(part, axis = 1)
        
        total.append(part)
        
    total = np.concatenate(total, axis = 0)

    return total

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

    args = parser.parse_args()

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    args, model = setup(args)

    visualize(args, model)

if __name__ == "__main__":
    main()