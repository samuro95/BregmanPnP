import os
import numpy as np
from argparse import ArgumentParser
from prox_PnP_restoration import PnP_restoration
from utils.utils_restoration import single2uint, crop_center, imread_uint, imsave
from natsort import os_sorted
from utils.utils_restoration import psnr, array2tensor, tensor2array
import torch

def denoise():

    parser = ArgumentParser()
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # Denoising specific hyperparameters
    hparams.degradation_mode = 'denoising'

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path, hparams.dataset_name)
    input_path = os.path.join(input_path, os.listdir(input_path)[0])
    input_paths = os_sorted([os.path.join(input_path, p) for p in os.listdir(input_path)])

    # Output images and curves paths
    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
        exp_out_path = 'denoise'
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)

    print('\n Prox-DRUNet denoising with image sigma:{:.3f} \n'.format(hparams.noise_level_img))

    psnr_list = []

    for i in range(min(len(input_paths), hparams.n_images)): # For each image

        print('__ image__', i)

        # load image
        input_im_uint = imread_uint(input_paths[i])
        if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
            input_im_uint = crop_center(input_im_uint, hparams.patch_size, hparams.patch_size)
        input_im = np.float32(input_im_uint / 255.)
        # Degrade image
        np.random.seed(seed=0)
        if hparams.noise_model == 'gaussian' :
            noise = np.random.normal(0, hparams.noise_level_img / 255., input_im.shape)
            noise_im = input_im + noise
            noise_im_tensor = array2tensor(noise_im).float()
            noise_im_tensor = noise_im_tensor.to(PnP_module.device)
            y = noise_im_tensor
            sigma_in = hparams.noise_level_img / 255.
        elif hparams.noise_model == 'poisson' :
            noise_im = np.random.poisson(np.maximum(hparams.alpha_poisson*input_im,0))
            noise_im_tensor = array2tensor(noise_im).float()
            noise_im_tensor = noise_im_tensor.to(PnP_module.device)
            y = noise_im_tensor/hparams.alpha_poisson
        elif hparams.noise_model == 'inv_gamma' :
            x = array2tensor(input_im).float().to(PnP_module.device).clamp(0.001,10)
            gamma = hparams.noise_level_img
            sigma_in = gamma
            m = torch.distributions.gamma.Gamma(gamma-1, gamma*x)
            z = m.sample().to(PnP_module.device)
            y = 1/z
            noise_im = tensor2array(y.cpu())

        Dx, g, Dg = PnP_module.denoise(y, sigma_in)
        
        denoise_img = tensor2array(Dx.cpu())
        psnri = psnr(denoise_img, input_im)

        psnr_list.append(psnri)

        if hparams.extract_images:
            # Save images
            save_im_path = os.path.join(exp_out_path, 'images')
            if not os.path.exists(save_im_path):
                os.mkdir(save_im_path)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_noise.png'), single2uint(noise_im))
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_denoise.png'), single2uint(denoise_img))
            print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_denoise.png'))

    avg_psnr = np.mean(np.array(psnr_list))
    print('avg RGB psnr for sigma={}: {:.2f}dB'.format(hparams.noise_level_img, avg_psnr))


if __name__ == '__main__':
    denoise()
