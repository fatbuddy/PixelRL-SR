import torch
import torchvision.io as io
import torchvision.transforms as transforms
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import json
import yaml

def read_image(filepath):
    image = io.read_image(filepath, io.ImageReadMode.RGB)
    return image

def write_image(filepath, src):
    io.write_png(src, filepath)
    
def pad_image_to_factor_of_16(image_tensor):
    # Get the original dimensions of the image tensor
    original_height, original_width = image_tensor.size(-2), image_tensor.size(-1)

    # Calculate the padding amounts needed on x and y dimensions
    pad_x = (16 - (original_width % 16)) % 16
    pad_y = (16 - (original_height % 16)) % 16

    # Decide whether to pad from left or both sides based on whether the original width is odd or even
    pad_left = pad_x // 2 if original_width % 2 == 0 else pad_x
    pad_right = pad_x // 2 if original_width % 2 == 0 else pad_x - pad_left

    # Decide whether to pad from top or both sides based on whether the original height is odd or even
    pad_top = pad_y // 2 if original_height % 2 == 0 else pad_y
    pad_bottom = pad_y // 2 if original_height % 2 == 0 else pad_y - pad_top

    # Pad the image tensor
    padded_image_tensor = torch.nn.functional.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_image_tensor

# from PPON library
def convert_shape(img):
    img = np.transpose((img * 255.0).round(), (1, 2, 0))
    img = np.uint8(np.clip(img, 0, 255))
    return img

def shave(im, border):
    border = [border, border]
    im = im[border[0]:-border[0], border[1]:-border[1], ...]
    return im

def quantize(img):
    return img.clip(0, 255).round().astype(np.uint8)

def compute_mse(im1, im2):
    return mse(im1, im2)

def compute_psnr(im1, im2):
    p = psnr(im1, im2)
    return p

def compute_ssim(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, win_size=min(im1.shape[0], im1.shape[1], 7), gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

# https://www.researchgate.net/publication/284923134
def rgb2ycbcr(src):
    R = src[0]
    G = src[1]
    B = src[2]
    
    ycbcr = torch.zeros(size=src.shape)
    # *Intel IPP
    # ycbcr[0] = 0.257 * R + 0.504 * G + 0.098 * B + 16
    # ycbcr[1] = -0.148 * R - 0.291 * G + 0.439 * B + 128
    # ycbcr[2] = 0.439 * R - 0.368 * G - 0.071 * B + 128
    # *Intel IPP specific for the JPEG codec
    ycbcr[0] =  0.299 * R + 0.587 * G + 0.114 * B
    ycbcr[1] =  -0.16874 * R - 0.33126 * G + 0.5 * B + 128
    ycbcr[2] =  0.5 * R - 0.41869 * G - 0.08131 * B + 128
    
    # Y in range [16, 235]
    ycbcr[0] = torch.clip(ycbcr[0], 16, 235)
    # Cb, Cr in range [16, 240]
    ycbcr[[1, 2]] = torch.clip(ycbcr[[1, 2]], 16, 240)
    ycbcr = ycbcr.type(torch.uint8)
    return ycbcr

# https://www.researchgate.net/publication/284923134
def ycbcr2rgb(src):
    Y = src[0]
    Cb = src[1]
    Cr = src[2]

    rgb = torch.zeros(size=src.shape)
    # *Intel IPP
    # rgb[0] = 1.164 * (Y - 16) + 1.596 * (Cr - 128)
    # rgb[1] = 1.164 * (Y - 16) - 0.813 * (Cr - 128) - 0.392 * (Cb - 128)
    # rgb[2] = 1.164 * (Y - 16) + 2.017 * (Cb - 128)
    # *Intel IPP specific for the JPEG codec
    rgb[0] = Y + 1.402 * Cr - 179.456
    rgb[1] = Y - 0.34414 * Cb - 0.71414 * Cr + 135.45984
    rgb[2] = Y + 1.772 * Cb - 226.816

    rgb = torch.clip(rgb, 0, 255)
    rgb = rgb.type(torch.uint8)
    return rgb

# list all file in dir and sort
def sorted_list(dir):
    ls = os.listdir(dir)
    ls.sort()
    for i in range(0, len(ls)):
        ls[i] = os.path.join(dir, ls[i])
    return ls

def resize_bicubic(src, h, w):
    image = transforms.Resize((h, w), transforms.InterpolationMode.BICUBIC)(src)
    return image

def gaussian_blur(src, ksize=3, sigma=0.5):
    blur_image = transforms.GaussianBlur(kernel_size=ksize, sigma=sigma)(src)
    return blur_image
    
def upscale(src, scale):
    h = int(src.shape[1] * scale)
    w = int(src.shape[2] * scale)
    image = resize_bicubic(src, h, w)
    return image

def downscale(src, scale):
    h = int(src.shape[1] / scale)
    w = int(src.shape[2] / scale)
    image = resize_bicubic(src, h, w)
    return image

def make_lr(src, scale=3):
    h = src.shape[1]
    w = src.shape[2]
    lr_image = downscale(src, scale)
    lr_image = resize_bicubic(lr_image, h, w)
    return lr_image

def norm01(src):
    return src / 255

def denorm01(src):
    return src * 255

def exists(path):
    return os.path.exists(path)

def exist_value(tensor, value):
    num_elements = tensor.shape[0]
    for i in range(0, num_elements):
        sum_values = torch.sum(tensor[i] == value)
        if sum_values > 0:
            return True
    return False

def PSNR(y_true, y_pred, max_val=1.0):
    y_true = y_true.type(torch.float32)
    y_pred = y_pred.type(torch.float32)
    MSE = torch.mean(torch.square(y_true - y_pred))
    return 10 * torch.log10(max_val * max_val / MSE)

def random_crop(src, h, w):
    crop = transforms.RandomCrop([h, w])(src)
    return crop

def random_transform(src):
    _90_left, _90_right, _180 = 1, 3, 2
    operations = {
        0 : (lambda x : x                                       ),
        1 : (lambda x : torch.rot90(x, k=_90_left,  dims=(1, 2))),
        2 : (lambda x : torch.rot90(x, k=_90_right, dims=(1, 2))),
        3 : (lambda x : torch.rot90(x, k=_180,      dims=(1, 2))),
        4 : (lambda x : torch.fliplr(x)                         ),
        5 : (lambda x : torch.flipud(x)                         ),
    }
    idx = np.random.choice([0, 1, 2, 3, 4, 5])
    image_transform = operations[idx](src)
    return image_transform

def shuffle(X, Y):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of elements")
    indices = np.arange(0, X.shape[0])
    np.random.shuffle(indices)
    X = torch.index_select(X, dim=0, index=torch.as_tensor(indices))
    Y = torch.index_select(Y, dim=0, index=torch.as_tensor(indices))
    return X, Y

# todo: draw action map based on 2d action matrix
def draw_action_map(actions, color_table):
    h = actions.shape[0]
    w = actions.shape[1]
    action_map = torch.zeros((3, h, w), dtype=torch.uint8)
    for i in range(0, h):
        for j in range(0, w):
            action_map[:, i, j] = color_table[actions[i, j]]
    return action_map

def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_cpu(tensor):
    return tensor.detach().cpu()
