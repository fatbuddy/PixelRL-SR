import torch
import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from SwinIR.network_swinir import SwinIR as net
import json
import os
from utils.common import *

SCALE = 4
LS_HR_PATHS = sorted_list(f"dataset/test/x{SCALE}/labels")
LS_LR_PATHS = sorted_list(f"dataset/test/x{SCALE}/data")
SIGMA = 0.3 if SCALE == 2 else 0.2

# declaringa a class
class obj:
     
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 grayscale JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 color JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model

def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8

    # 003 real-world image sr
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8

    # 004 grayscale image denoising/ 005 color image denoising
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8

    # 006 JPEG compression artifact reduction
    elif args.task in ['jpeg_car', 'color_jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_gt
        border = 0
        window_size = 7

    return folder, save_dir, border, window_size

def main():
    
    opt = {
        'task': 'classical_sr',
        'scale': SCALE,
        'noise': 15,
        'jpeg': 40,
        'training_patch_size': 48,
        'large_model': True,
        'model_path': 'sr_weight/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
        'folder_gt': f'dataset/test/x{SCALE}/labels',
        'folder_lq': f'dataset/test/x{SCALE}/data',
        'tile': None,
        'tile_overlap': 32
    }
    opt = json.loads(json.dumps(opt), object_hook=obj)
    model = define_model(opt)
    model.eval()
    folder, save_dir, border, window_size = setup(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cuda()

    psnr_sr = np.zeros(len(LS_HR_PATHS))
    # psnr_sr = np.zeros(1)
    ssim_sr = np.zeros(len(LS_HR_PATHS))
    # ssim_sr = np.zeros(1)
    for i in range(len(LS_HR_PATHS)):
    # for i in range(1):
        hr_image_path = LS_HR_PATHS[i]
        lr_image_path = LS_LR_PATHS[i]
        
        hr = read_image(hr_image_path)
        lr = read_image(lr_image_path)
        lr = gaussian_blur(lr, sigma=SIGMA)
        bicubic = upscale(lr, SCALE)

        bicubic = rgb2ycbcr(bicubic)
        lr = rgb2ycbcr(lr)
        hr= rgb2ycbcr(hr)


        bicubic = norm01(bicubic).unsqueeze(0)
        lr = norm01(lr).unsqueeze(0)
        hr = norm01(hr).unsqueeze(0)

        lr = lr.cuda()
        with torch.no_grad():
            print(f"lr.shape: {lr.shape}")
            output = test(lr, model, opt, window_size)
            output = output.cpu()
        
        sr_image = torch.clip(output, 0.0, 1.0)
        psnr_sr[i] = PSNR(hr, sr_image)

        sr_image_np = sr_image.detach().numpy()  # Convert tensor to numpy array
        print(f"HR shape: {hr.shape}")
        print(f"SR shape: {sr_image.shape}")
        ssim_sr[i] = compute_ssim(quantize(hr.detach().numpy()),quantize(sr_image_np))
        sr_image = denorm01(output.squeeze())
        sr_image = sr_image.type(torch.uint8)
        sr_image = ycbcr2rgb(sr_image)
        # write_image("sr.png", sr_image)

    print('Mean PSNR for SR: {}'.format(np.mean(psnr_sr)))
    print('Mean SSIM for SR: {}'.format(np.mean(ssim_sr)))

if __name__ == '__main__':
    main()


