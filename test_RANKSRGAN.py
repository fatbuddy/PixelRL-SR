import torch
import torch.nn as nn
from RANKSRGAN.RankSRGAN_model import SRGANModel
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


def main():
    
    opt = {
        'alpha': 1.0,
        'cuda': True,
        'isHR': True,
        'is_train': False,
        'models': 'sr_weight/RankSRGAN_NIQE.pth',
        'pretrained_model_D': 'sr_weight/RankSRGAN_NIQE.pth',
        'pretrained_model_G': 'sr_weight/RankSRGAN_NIQE.pth',
        'only_y': True,
        'output_folder': 'result/Set5/',
        'save_path': 'save',
        'test_hr_folder': f'dataset/test/x{SCALE}/labels',
        'test_lr_folder': f'dataset/test/x{SCALE}/data',
        'upscale_factor': SCALE,
        'which_model': 'SRResNet',
    }
    opt = json.loads(json.dumps(opt), object_hook=obj)
    RANKSRGAN = SRGANModel(opt)
    if isinstance(RANKSRGAN, nn.DataParallel):
        RANKSRGAN = RANKSRGAN.module
    psnr_sr = np.zeros(len(LS_HR_PATHS))
    ssim_sr = np.zeros(len(LS_HR_PATHS))
    for i in range(len(LS_HR_PATHS)):
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

        if opt.cuda:
            RANKSRGAN = RANKSRGAN.cuda()
            lr = lr.cuda()
        with torch.no_grad():
            RANKSRGAN.feed_data([lr], need_GT=False)
            RANKSRGAN.test()
            visuals = RANKSRGAN.get_current_visuals(need_GT=False)
        
        sr_image = visuals['rlt'].unsqueeze(0)
        psnr_sr[i] = PSNR(hr, sr_image)

        sr_image_np = sr_image.detach().numpy()  # Convert tensor to numpy array
        ssim_sr[i] = compute_ssim(quantize(hr.detach().numpy()),quantize(sr_image_np))

    print('Mean PSNR for SR: {}'.format(np.mean(psnr_sr)))
    print('Mean SSIM for SR: {}'.format(np.mean(ssim_sr)))

if __name__ == '__main__':
    main()
