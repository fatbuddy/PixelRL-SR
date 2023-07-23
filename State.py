import torch
from neuralnet import ESPCN_model, FSRCNN_model, SRCNN_model, VDSR_model
from SwinIR.utils import *
import torch.nn as nn
from PPON.PPON_model import PPONModel
from RANKSRGAN.RankSRGAN_model import SRGANModel
from PPON import networks
from utils.common import exist_value, to_cpu, convert_shape, pad_image_to_factor_of_16
import json
from hat.archs.hat_arch import HAT

# declaringa a class
class obj:
     
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
class State:
    def __init__(self, scale, device):
        self.device = device
        self.lr_image = None
        self.sr_image = None
        self.tensor = None
        self.move_range = 3

        dev = torch.device(device)

        # PPON
        opt = {
            'alpha': 1.0,
            'cuda': True,
            'isHR': True,
            'is_train': False,
            'models': 'sr_weight/PPON_G.pth',
            'pretrained_model_D': 'sr_weight/PPON_D.pth',
            'pretrained_model_G': 'sr_weight/PPON_G.pth',
            'only_y': True,
            'output_folder': 'result/Set5/',
            'save_path': 'save',
            'test_hr_folder': f'dataset/test/x{scale}/labels',
            'test_lr_folder': f'dataset/test/x{scale}/data',
            'upscale_factor': scale,
            'which_model': 'ppon'
        }
        opt = json.loads(json.dumps(opt), object_hook=obj)
        self.PPON = networks.define_G(opt)
        if isinstance(self.PPON, nn.DataParallel):
            self.PPON = self.PPON.module
        model_path = "sr_weight/PPON_G.pth"
        self.PPON.load_state_dict(torch.load(model_path), strict=True)

        # For SwinIR
        opt = {
            'task': 'classical_sr',
            'scale': scale,
            'noise': 15,
            'jpeg': 40,
            'training_patch_size': 48,
            'large_model': True,
            'model_path': 'sr_weight/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
            'folder_gt': f'dataset/test/x{scale}/labels',
            'folder_lq': f'dataset/test/x{scale}/data',
            'tile': None,
            'tile_overlap': 32
        }
        opt = json.loads(json.dumps(opt), object_hook=obj)
        self.SwinIR = define_model(opt)
        self.SwinIR.eval()

        # For HAT
        self.HAT_model = HAT(upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv').to(device)
        model_path = "sr_weight/HAT_SRx4.pth"
        self.HAT_model.load_state_dict(torch.load(model_path)['params_ema'])
        self.HAT_model.eval()

        # RANKSRGAN
        opt = {
            'alpha': 1.0,
            'cuda': True,
            'isHR': True,
            'is_train': False,
            'models': 'sr_weight/mmsr_RankSRGAN_NIQE.pth',
            'pretrained_model_D': 'sr_weight/mmsr_RankSRGAN_NIQE.pth',
            'pretrained_model_G': 'sr_weight/mmsr_RankSRGAN_NIQE.pth',
            'only_y': True,
            'output_folder': 'result/Set5/',
            'save_path': 'save',
            'test_hr_folder': f'dataset/test/x{scale}/labels',
            'test_lr_folder': f'dataset/test/x{scale}/data',
            'upscale_factor': scale,
            'which_model': 'SRResNet',
        }
        opt = json.loads(json.dumps(opt), object_hook=obj)
        self.RANKSRGAN = SRGANModel(opt)
        if isinstance(self.RANKSRGAN, nn.DataParallel):
            self.RANKSRGAN = self.RANKSRGAN.module

    def reset(self, lr, bicubic):
        self.lr_image = lr 
        self.sr_image = bicubic
        b, _, h, w = self.sr_image.shape
        previous_state = torch.zeros(size=(b, 64, h, w), dtype=self.lr_image.dtype)
        self.tensor = torch.concat([self.sr_image, previous_state], dim=1)

    def set(self, lr, bicubic):
        self.lr_image = lr
        self.sr_image = bicubic
        self.tensor[:,0:3,:,:] = self.sr_image

    def step(self, act, inner_state):
        act = to_cpu(act)
        inner_state = to_cpu(inner_state)
        ppon = self.sr_image.clone()
        swinir = self.sr_image.clone()
        hat = self.sr_image.clone()
        ranksrgan = self.sr_image.clone()

        neutral = (self.move_range - 1) / 2
        move = act.type(torch.float32)
        move = (move - neutral) / 255
        moved_image = self.sr_image.clone()
        for i in range(0, self.sr_image.shape[1]):
            moved_image[:,i] += move[0]

        self.lr_image = self.lr_image.to(self.device)
        self.sr_image = self.sr_image.to(self.device)

        with torch.no_grad():
            if exist_value(act, 3):
                # change ESPCN to PPON
                self.PPON.cuda()
                out_c, out_s, out_p = self.PPON(self.lr_image)
                out_c, out_s, out_p = out_c.cpu(), out_s.cpu(), out_p.cpu()
                out_img_p = out_c.detach().numpy().squeeze()
                ppon = torch.from_numpy(out_img_p)
            if exist_value(act, 4):
                # change SRCNN to SwinIR
                self.SwinIR.cuda()
                self.lr_image.cuda()
                output = self.SwinIR(self.lr_image)
                output = output.cpu()
                output = output.detach().numpy().squeeze()
                # print(out_img_c.shape)
                swinir = torch.from_numpy(output)
            if exist_value(act, 5):
                # change VDSR to HAT
                hat = self.HAT_model(self.lr_image.float())
                hat = to_cpu(hat.int())
            if exist_value(act, 6):
                self.lr_image.cuda()
                with torch.no_grad():
                    self.RANKSRGAN.feed_data([self.lr_image], need_GT=False)
                    self.RANKSRGAN.test()
                    visuals = self.RANKSRGAN.get_current_visuals(need_GT=False)['rlt'].unsqueeze(0)
                ranksrgan = visuals

        self.lr_image = to_cpu(self.lr_image)
        self.sr_image = moved_image
        act = act.unsqueeze(1)
        act = torch.concat([act, act, act], 1)
  
        self.sr_image = torch.where(act==3, ppon,  self.sr_image)
        self.sr_image = torch.where(act==4, swinir,  self.sr_image)
        self.sr_image = torch.where(act==5, hat,  self.sr_image)
        self.sr_image = torch.where(act==6, ranksrgan,  self.sr_image)

        self.tensor[:,0:3,:,:] = self.sr_image
        self.tensor[:,-64:,:,:] = inner_state
