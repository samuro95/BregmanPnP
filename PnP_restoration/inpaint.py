import os
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from prox_PnP_restoration import PnP_restoration
from utils.utils_restoration import (
    single2uint,
    crop_center,
    matlab_style_gauss2D,
    imread_uint,
    imsave,
)
from natsort import os_sorted


def inpaint():

    parser = ArgumentParser()
    parser.add_argument("--prop_mask", type=float, default=0.5)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # Inpainting specific hyperparameters
    hparams.degradation_mode = "inpainting"
    hparams.PnP_algo = "DRS"
    hparams.sigma_denoiser = 15
    hparams.tau = 2
    hparams.lamb = 0.5
    hparams.noise_level_img = 0
    hparams.n_init = 10
    hparams.maxitr = 500
    hparams.use_backtracking = False
    hparams.inpainting_init = True
    hparams.relative_diff_Psi_min = 1e-20

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path, hparams.dataset_name)
    input_path = os.path.join(input_path, os.listdir(input_path)[0])
    input_paths = os_sorted(
        [os.path.join(input_path, p) for p in os.listdir(input_path)]
    )

    # Output images and curves paths
    den_out_path = "inpaint"
    if not os.path.exists(den_out_path):
        os.mkdir(den_out_path)
    exp_out_path = os.path.join(den_out_path, hparams.PnP_algo)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    kout_path = os.path.join(exp_out_path, "prop_" + str(hparams.prop_mask))
    if not os.path.exists(kout_path):
        os.mkdir(kout_path)

    test_results = OrderedDict()
    test_results["psnr"] = []

    if hparams.extract_curves:
        PnP_module.initialize_curves()

    psnr_list = []
    psnrY_list = []
    F_list = []

    # for i in range(min(len(input_paths), hparams.n_images)): # For each image
    for i in range(0, 1):

        print("__ image__", i)

        # load image
        input_im_uint = imread_uint(input_paths[i])
        if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
            input_im_uint = crop_center(
                input_im_uint, hparams.patch_size, hparams.patch_size
            )
        input_im = np.float32(input_im_uint / 255.0)
        # Degrade image
        mask = np.random.binomial(
            n=1, p=hparams.prop_mask, size=(input_im.shape[0], input_im.shape[1])
        )
        mask = np.expand_dims(mask, axis=2)
        mask_im = input_im * mask + (0.5) * (1 - mask)

        np.random.seed(seed=0)
        mask_im += np.random.normal(0, hparams.noise_level_img / 255.0, mask_im.shape)

        init_im = mask_im

        # PnP restoration
        if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
            inpainted_im, output_psnr, output_psnrY, x_list, z_list, y_list, Dg_list, psnr_tab, Dx_list, g_list, F_list, Psi_list = PnP_module.restore(
                mask_im.copy(),
                init_im.copy(),
                input_im.copy(),
                mask,
                extract_results=True,
            )
        else:
            inpainted_im, output_psnr, output_psnrY = PnP_module.restore(
                mask_im, init_im, input_im, mask
            )

        print("PSNR: {:.2f}dB".format(output_psnr))
        psnr_list.append(output_psnr)
        psnrY_list.append(output_psnrY)

        if hparams.extract_curves:
            # Create curves
            PnP_module.update_curves(
                x_list, Dx_list, psnr_tab, Dg_list, g_list, F_list, Psi_list
            )

        if hparams.extract_images:
            # Save images
            save_im_path = os.path.join(kout_path, "images")
            if not os.path.exists(save_im_path):
                os.mkdir(save_im_path)

            imsave(
                os.path.join(save_im_path, "img_" + str(i) + "_input.png"),
                input_im_uint,
            )
            imsave(
                os.path.join(save_im_path, "img_" + str(i) + "_inpainted.png"),
                single2uint(inpainted_im),
            )
            imsave(
                os.path.join(save_im_path, "img_" + str(i) + "_masked.png"),
                single2uint(mask_im * mask),
            )

            print("output images saved at ", save_im_path)

    if hparams.extract_curves:
        # Save curves
        save_curves_path = os.path.join(kout_path, "curves")
        if not os.path.exists(save_curves_path):
            os.mkdir(save_curves_path)
        PnP_module.save_curves(save_curves_path)
        print("output curves saved at ", save_curves_path)

    avg_k_psnr = np.mean(np.array(psnr_list))
    print("avg RGB psnr : {:.2f}dB".format(avg_k_psnr))
    avg_k_psnrY = np.mean(np.array(psnrY_list))
    print("avg Y psnr : {:.2f}dB".format(avg_k_psnrY))


if __name__ == "__main__":
    psnr = inpaint()
