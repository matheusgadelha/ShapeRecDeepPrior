import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import skimage.io as io

import argparse
import os
import sys

from models.skip import skip

from volumetocube import write_obj_from_array
import binvox_rw
from tools.Ops import radon
from tools.Ops import tvloss

from skimage.measure import compare_ssim as ssim


def add_sp_noise(img, p=0.2):
    randv = np.random.rand(*(img.shape)).flatten()
    flip = randv > (1-p)
    noisy_img = img.copy().flatten()
    is_white = noisy_img > 0
    noisy_img[is_white & flip] = 0
    noisy_img[~is_white & flip] = 1.0

    final_img = noisy_img.copy()

    return final_img.reshape((img.shape[0], img.shape[1], -1))


def add_gaussian_noise(img, sigma=1.0):
    randv = torch.randn(*(img.shape)).cuda()
    return img + sigma*randv


def save_img(path, img):
    imgdata = img.data.cpu().numpy()
    io.imsave(path, imgdata/imgdata.max())


if __name__ == '__main__':
    #net = ImageUNet(256, batch_size=1, noise_reg=1.0/3.0).cuda()
    input_depth = 3
    input_noise = torch.randn(1,input_depth,256,256).cuda()
    input_noise.detach()
    net = skip(
        input_depth, 3, 
        num_channels_down = [8, 16, 32, 64, 128], 
        num_channels_up   = [8, 16, 32, 64, 128],
        num_channels_skip = [0, 0, 0, 4, 4], 
        upsample_mode='bilinear',
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
    net.cuda()

    #optimizer = optim.Adam(net.parameters(), lr=0.01)

    theta_size = 30
    #rec_image = (io.imread('data/slice_058.png')[:, :, 0]/255.0).astype('float32')
    rec_image = (io.imread('data/slice_038.png')/255.0).astype('float32')
    sinogram = radon(torch.tensor(rec_image),
            thetas=torch.linspace(0, np.pi, steps=theta_size)).cuda()
    noisy_sinogram = add_gaussian_noise(sinogram, sigma=1.0)
    
    mse = nn.L1Loss()
    sigmoid = nn.Sigmoid()

    folder = 'results/radon'
    os.makedirs(folder, exist_ok=True)
    save_img('results/radon/sinogram.png', sinogram)
    save_img('results/radon/noisy_sinogram.png', noisy_sinogram)
    io.imsave('results/radon/rec_image.png', rec_image)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for i in range(10000):
        optimizer.zero_grad()
        
        out_rec = net(input_noise)[0, 0, :, :]
        out_sinogram = radon(out_rec, thetas=torch.linspace(0, np.pi, steps=theta_size))
        loss = mse(out_sinogram, noisy_sinogram)
        gtloss = mse(out_sinogram, sinogram)
        noisyloss = mse(noisy_sinogram, sinogram)
        loss.backward()
        print("{} | {} | {}".format(loss.item(), gtloss.item(), noisyloss.item()))

        if i % 100 == 0 and i > 0:
            save_img('results/radon/rec_denoised{}.png'.format(str(i).zfill(5)),
                    out_rec)

        optimizer.step()

    #print "SSIM: {}".format(ssim(denoised.data.cpu().numpy(), rec_image, 
    #    data_range=rec_image.max()-rec_image.min()))
