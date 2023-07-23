import torch
import RANKSRGAN.archs.SRResNet_arch as SRResNet_arch
import RANKSRGAN.archs.discriminator_vgg_arch as SRGAN_arch
import RANKSRGAN.archs.RRDBNet_arch as RRDBNet_arch
import RANKSRGAN.archs.RankSRGAN_arch as RankSRGAN_arch


# Generator
def define_G(opt):
    # opt_net = opt['network_G']
    which_model = opt.which_model

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=3, out_nc=3, nf=64, nb=16)
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=3,out_nc=3, nf=64, nb=16)
    elif which_model == 'SRResNet':
        netG = RankSRGAN_arch.SRResNet(in_nc=3,out_nc=3, nf=64, nb=16)
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_vgg_296':
        netD = RankSRGAN_arch.Discriminator_VGG_296(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

# Define network used for rank-content loss
def define_R(opt):
    opt_net = opt['network_R']
    which_model = opt_net['which_model_R']

    if which_model == 'Ranker_VGG12':
        netR = RankSRGAN_arch.Ranker_VGG12_296(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Ranker model [{:s}] is not recognized'.format(which_model))

    return netR
