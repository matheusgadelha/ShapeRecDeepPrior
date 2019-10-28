import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import skimage.io as io

import argparse
import os
import sys
import time

# Allow python3 to search for modules outside of this directory
sys.path.append("../")

from models.skip import skip3d

from volumetocube import write_bin_from_array
from volumetocube import write_obj_from_array
import binvox_rw
from tools.Ops import radon
from tools.Ops import tvloss
from tools.Ops import tvloss3d
from tools.Ops import load_binvox
from tools.Ops import volume_proj
from tools.Ops import rotate_volume
from tools.Ops import inv_rotate_volume

from skimage.measure import compare_ssim as ssim

parser = argparse.ArgumentParser(description='Reconstruciton using deep prior.')
parser.add_argument("-m", "--method", type=str, help="Prior to be used in the reconstruction (deep | tv | carve)", default="deep")
parser.add_argument("-b", "--binvox", type=str, help="Path to the binvox file.", default="../data/bunny.binvox")
parser.add_argument("-p", "--projection", type=str, help="Type of projection to be used (depth | binary)", default="depth")
parser.add_argument("-n", "--nproj", type=int, help="Number of projections.", default=8)
parser.add_argument("-s", "--sigma", type=float, help="Amount of variance in the gaussian noise.", default=0.0)
parser.add_argument("-k", "--kappa", type=float, help="Dispersion rate of Von Mises noise.", default=4.0)
parser.add_argument("-v", "--viewWeight", type=float, help="Weight of the viewpoint regularization.", default=1.0)


def add_gaussian_noise(img, sigma=1.0):
    randv = torch.randn(*(img.shape)).cuda()
    return img + sigma*randv


if __name__ == '__main__':
    args = parser.parse_args()
    use_tv = args.method == 'tv'
    use_dp = args.method == 'deep'
    kappa = args.kappa
    view_weight = args.viewWeight

    binvoxname = args.binvox.split('/')[-1].split('.')[0]
    fullname = "prob_{}_{}_{}_{}_{}_vw{}_k{}".format(binvoxname, args.method, args.projection,
            args.nproj, args.sigma, view_weight, kappa)

    input_depth = 3
    input_noise = torch.randn(1, input_depth, 128, 128, 128).cuda()
    net = skip3d(
        input_depth, 1, 
        num_channels_down = [8, 16, 32, 64, 128], 
        num_channels_up   = [8, 16, 32, 64, 128],
        num_channels_skip = [0, 0, 0, 4, 4], 
        upsample_mode='trilinear',
        need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU')
    net.cuda()
    net(input_noise)

    out_volume = torch.zeros(1, 1, 128, 128, 128).cuda()
    out_volume.requires_grad = True

    nviews = args.nproj
    method = args.projection

    views = torch.FloatTensor(np.random.rand(nviews, 3) * 2*np.pi)
    noisy_views = torch.FloatTensor(np.random.vonmises(views, kappa, size=(nviews,3)))
    pred_views = nn.Parameter(noisy_views.detach().clone())

    if use_dp:
        optimizer = optim.Adam(list(net.parameters()) + [pred_views], lr=0.01)
    elif use_tv:
        optimizer = optim.Adam([out_volume] + [pred_views], lr=0.01)
    
    padder = nn.ConstantPad3d(10, 0.0)
    
    volume = padder(load_binvox(args.binvox).cuda())
    gtprojs = volume_proj(volume, method=method, views=views).cuda()
    noisyprojs = gtprojs.detach().clone()
    noisyprojs.requires_grad = False

    results_dir = os.path.join("results", fullname)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    mse = nn.L1Loss()
    sigmoid = nn.Sigmoid()

    #Space carving
    if args.method == 'carve':
        gtprojs = volume_proj(volume, method=method, views=views).cuda()
        gtprojs.requires_grad = False
        noisyprojs = gtprojs.clone()
        noisyprojs.requires_grad = False

        carve = torch.ones(*(volume.size())).cuda()
        for i in range(nviews):
            carve = rotate_volume(carve, x=noisy_views[i,0], y=noisy_views[i,1], z=noisy_views[i,2])
            p = gtprojs[:, :, i] < 1e-2
            coords = np.argwhere(p)
            carve[coords[0, :], :, coords[1, :]] = 0.0
            carve = inv_rotate_volume(carve, x=noisy_views[i,0], y=noisy_views[i,1], z=noisy_views[i,2])

        projs = volume_proj(carve, method=method, views=views).cuda()
        for i in range(noisyprojs.size()[2]):
            io.imsave(results_dir+"/carve{}.png".format(i), torch.clamp(projs[:, :, i], -1, 1))
            io.imsave(results_dir+"/carvegt{}.png".format(i), torch.clamp(gtprojs[:, :, i], -1, 1))
        write_bin_from_array("results/{}/data.npy".format(fullname), carve.data.cpu().numpy())

        exit(0)

    gt_curve = []
    noisygt_curve = []

    n_iter = 500
    out_rec = None
    out_projs = None
    pred_views_log = []
    noisy_views_log = []
    gt_views_log = []
    print('EXPERIMENT {}'.format(fullname))
    for i in range(n_iter):
        optimizer.zero_grad()

        if use_dp:
            out_rec = net(input_noise)[0, 0, :, :, :]
            out_projs = volume_proj(out_rec, method=method, views=pred_views)
            loss = mse(out_projs, noisyprojs)
            loss -= view_weight * torch.cos(pred_views - noisy_views).mean().cuda()
        elif use_tv:
            out_rec = sigmoid(out_volume[0, 0, :, :, :])
            out_projs = volume_proj(out_rec, method=method, views=views)
            loss = mse(out_projs, noisyprojs) + tvloss3d(out_rec, weight=1e-7)#
        else:
            raise ValueError("Unkown method")

        pred_views_log.append(pred_views.data.detach().cpu().numpy())
        noisy_views_log.append(noisy_views.data.detach().cpu().numpy())
        gt_views_log.append(views.data.detach().cpu().numpy())

        predloss = mse(out_projs, noisyprojs)
        gtloss = torch.abs(out_projs - gtprojs).mean()
        noisyloss = torch.abs(noisyprojs - gtprojs).mean()
        print("\r({}/{}) Pred->Noisy: {} | Pred->GT: {} | Noisy->GT: {}".format(
                str(i).zfill(4), n_iter, predloss.item(), gtloss.item(), noisyloss.item()),
        gt_curve.append(gtloss.item()))
        noisygt_curve.append(noisyloss.item())

        loss.backward()
        optimizer.step()

    results_dir = os.path.join("results", fullname)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    write_bin_from_array("results/{}/databin.npy".format(fullname), 
            out_rec.data.cpu().detach().numpy())

    np.save("results/{}/data.npy".format(fullname), 
            out_rec.data.cpu().detach().numpy())

    for i in range(out_projs.size()[2]):
        print("Saved {}".format("results/{}/proj{}".format(fullname, i)))
        io.imsave("results/{}/proj{}.png".format(fullname, i), 
                out_projs.data.cpu().detach().numpy()[:, :, i])
        io.imsave("results/{}/gt{}.png".format(fullname, i), 
                torch.clamp(gtprojs[:, :, i], -1, 1).data.cpu().detach().numpy())

    np.save("results/{}/gtviews.npy".format(fullname), np.array(gt_views_log))
    np.save("results/{}/noisyviews.npy".format(fullname), np.array(noisy_views_log))
    np.save("results/{}/predviews.npy".format(fullname), np.array(pred_views_log))

