import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import (
    rescale,
    array2tensor,
    tensor2array,
    get_poisson_noise_parameters,
    create_out_dir,
    single2uint,
    crop_center,
    matlab_style_gauss2D,
    imread_uint,
    imsave,
)
from natsort import os_sorted
from prox_PnP_restoration import PnP_restoration
import wandb
from utils.utils_sr import Wiener_filter
import torch
import cv2


def deblur():

    parser = ArgumentParser()
    parser.add_argument("--kernel_path", type=str)
    parser.add_argument("--kernel_indexes", nargs="+", type=int)
    parser.add_argument("--image_path", type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    hparams.degradation_mode = "deblurring"

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    if hparams.image_path is not None:  # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[
            0
        ]
    else:  # if not given, we aply on the whole dataset name given in argument
        input_path = os.path.join(hparams.dataset_path, hparams.dataset_name, "0")
        input_paths = os_sorted(
            [os.path.join(input_path, p) for p in os.listdir(input_path)]
        )

    psnr_list = []
    F_list = []

    if (
        hparams.kernel_path is not None
    ):  # if a specific kernel saved in hparams.kernel_path as np array is given
        k_list = [np.load(hparams.kernel_path)]
        k_index_list = [0]
    else:
        k_list = []
        # If no specific kernel is given, load the 8 motion blur kernels
        kernel_path = os.path.join("kernels", "Levin09.mat")
        kernels = hdf5storage.loadmat(kernel_path)["kernels"]
        # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
        for k_index in range(10):
            if k_index == 8:  # Uniform blur
                k = np.float32((1 / 81) * np.ones((9, 9)))
            elif k_index == 9:  # Gaussian blur
                k = np.float32(matlab_style_gauss2D(shape=(25, 25), sigma=1.6))
            else:  # Motion blur
                k = np.float32(kernels[0, k_index])
            k_list.append(k)

        if hparams.kernel_indexes is not None:
            k_index_list = hparams.kernel_indexes
        else:
            k_index_list = range(len(k_list))

    if hparams.use_wandb:
        wandb.init(project=hparams.degradation_mode)

    data = []

    for k_index in k_index_list:  # For each kernel

        psnr_k_list = []
        n_it_list = []

        k = k_list[k_index]

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for PnP_algo in hparams.PnP_algos:

            PnP_module.hparams.PnP_algo = PnP_algo

            PnP_module.hparams.lamb_pre_init, PnP_module.hparams.sigma_denoiser_pre_init, PnP_module.hparams.stepsize_pre_init, PnP_module.hparams.lim_init, PnP_module.hparams.sigma_denoiser, PnP_module.hparams.lamb, PnP_module.hparams.stepsize, PnP_module.hparams.alpha = get_poisson_noise_parameters(
                    hparams.PnP_algo,
                    hparams.bregman_potential,
                    hparams.alpha_poisson,
                    hparams.use_GSDRUNET_ckpt,
                )

            if (
                hparams.extract_images
                or hparams.extract_curves
                or hparams.print_each_step
            ):
                exp_out_path = create_out_dir(
                    hparams.degradation_mode, hparams.dataset_name, PnP_algo
                )

            np.random.seed(seed=0)

            for i in range(min(hparams.n_images, len(input_paths))):  # For each image
                print(
                    "Deblurring of image {}, kernel index {}, with PnP-{} algorithm".format(
                        i, k_index, PnP_algo
                    )
                )

                # load image
                input_im_uint = imread_uint(input_paths[i])
                if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
                    input_im_uint = crop_center(input_im_uint, hparams.patch_size, hparams.patch_size)
                input_im = np.float32(input_im_uint / 255.0)

                if hparams.grayscale:
                    input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
                    input_im = np.expand_dims(input_im, axis=2)
                # Degrade image
                blur_im = ndimage.convolve(input_im, np.expand_dims(k, axis=2), mode="wrap")

                np.random.seed(seed=0)
                blur_im = (
                        np.random.poisson(
                            np.maximum(hparams.alpha_poisson * blur_im, 0)
                        )
                        / hparams.alpha_poisson
                    )
                blur_im = np.float32(blur_im)
                if hparams.use_Wiener_init:  # Wiener filter initialzation for Poisson noise
                    init_im = Wiener_filter(
                        array2tensor(blur_im),
                        torch.tensor(k),
                        1 / hparams.alpha_poisson,
                        1,
                    )
                    init_im = tensor2array(torch.clamp(init_im, 0, 1))
                else:
                    init_im = blur_im

                # PnP restoration
                if (
                    hparams.extract_images
                    or hparams.extract_curves
                    or hparams.print_each_step
                ):
                    deblur_im, init_im, output_psnr, n_it, x_list, z_list, y_list, Dg_list, psnr_tab, g_list, F_list, Psi_list, f_list, Df_list = PnP_module.restore(
                        blur_im.copy(),
                        init_im.copy(),
                        input_im.copy(),
                        k,
                        extract_results=True,
                    )
                else:
                    deblur_im, init_im, output_psnr, n_it = PnP_module.restore(
                        blur_im, init_im, input_im, k
                    )

                print("PSNR: {:.2f}dB".format(output_psnr))
                print(f"N iterations: {n_it}")

                psnr_k_list.append(output_psnr)
                psnr_list.append(output_psnr)
                n_it_list.append(n_it)

                if hparams.extract_curves:
                    # Create curves
                    PnP_module.update_curves(
                        x_list,
                        psnr_tab,
                        Dg_list,
                        g_list,
                        F_list,
                        Psi_list,
                        f_list,
                        Df_list,
                    )

                if hparams.extract_images:
                    # Save images
                    save_im_path = os.path.join(exp_out_path, "images")
                    if not os.path.exists(save_im_path):
                        os.mkdir(save_im_path)
                    imsave(
                        os.path.join(save_im_path, "img_" + str(i) + "_input.png"),
                        input_im_uint,
                    )
                    imsave(
                        os.path.join(save_im_path, "img_" + str(i) + "_deblur.png"),
                        single2uint(rescale(deblur_im)),
                    )
                    imsave(
                        os.path.join(save_im_path, "img_" + str(i) + "_blur.png"),
                        single2uint(rescale(blur_im)),
                    )
                    imsave(
                        os.path.join(save_im_path, "img_" + str(i) + "_init.png"),
                        single2uint(rescale(init_im)),
                    )
                    print(
                        "output image saved at ",
                        os.path.join(save_im_path, "img_" + str(i) + "_deblur.png"),
                    )

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(exp_out_path, "curves")
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print("output curves saved at ", save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print("avg RGB psnr on kernel {}: {:.2f}dB".format(k_index, avg_k_psnr))

        data.append([k_index, avg_k_psnr, np.mean(np.mean(n_it_list))])

    data = np.array(data)

    if hparams.use_wandb:
        table = wandb.Table(data=data, columns=["k", "psnr", "n_it"])
        for i, metric in enumerate(["psnr", "n_it"]):
            wandb.log(
                {
                    f"{metric}_plot": wandb.plot.scatter(
                        table, "k", metric, title=f"{metric} vs. k"
                    ),
                    f"average_{metric}": np.mean(data[:, i + 1]),
                },
                step=0,
            )

    return np.mean(np.array(psnr_list))


if __name__ == "__main__":
    deblur()
