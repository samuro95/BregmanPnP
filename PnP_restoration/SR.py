import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import modcrop, rescale, array2tensor, tensor2array, get_gaussian_noise_parameters, get_poisson_noise_parameters, create_out_dir, single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from natsort import os_sorted
from prox_PnP_restoration import PnP_restoration
from utils.utils_sr import numpy_degradation, shift_pixel
import wandb
import cv2

def SR():
    parser = ArgumentParser()
    parser.add_argument('--sf', nargs='+', type=int)
    parser.add_argument('--kernel_path', type=str)
    parser.add_argument('--kernel_indexes', nargs='+', type=int)
    parser.add_argument('--image_path', type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()


    # SR specific hyperparameters
    hparams.degradation_mode = 'SR'

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name,'0')
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    psnr_list = []
    F_list = []

    if hparams.kernel_path is not None : # if a specific kernel saved in hparams.kernel_path as np array is given 
        k_list = [np.load(hparams.kernel_path)]
    else : 
        # If no specific kernel is given, load the SR kernels used in the paper
        kernel_path = os.path.join('kernels','kernels_12.mat')
        k_list = hdf5storage.loadmat(kernel_path)['kernels'][0][:4]

    if hparams.kernel_indexes is not None : 
        k_index_list = hparams.kernel_indexes
    else :
        k_index_list = range(len(k_list))


    if hparams.sf is not None : # if SR scales are given 
        sf_list = hparams.sf
    else :
        sf_list = [2,3]


    # wandb.init(
    #     project=hparams.degradation_mode
    #     )        

    data = []

    for k_index in k_index_list : # For each kernel

        psnr_k_list = []
        psnrY_k_list = []
        n_it_list = []

        k = k_list[k_index]

        for sf in sf_list :

            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                exp_out_path = 'SR'
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path)
                exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path)
                exp_out_path = os.path.join(exp_out_path, str(hparams.noise_level_img))
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path)
                exp_out_path = os.path.join(exp_out_path, 'kernel_' + str(k_index))
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path) 
                exp_out_path = os.path.join(exp_out_path, 'sf_' + str(sf))
                if not os.path.exists(exp_out_path):
                    os.mkdir(exp_out_path) 
        
            if hparams.extract_curves:
                PnP_module.initialize_curves()

            for PnP_algo in hparams.PnP_algos : 

                PnP_module.hparams.PnP_algo = PnP_algo

                if hparams.noise_model == 'gaussian':
                    if PnP_algo == 'PGD_RED' or  PnP_algo == 'GD':
                        PnP_module.use_GSDRUNET_ckpt = True
                    else:
                        PnP_module.use_GSDRUNET_ckpt = False
                    PnP_module.initialize_denoiser()

                if hparams.default_params:
                    if hparams.noise_model == 'gaussian':
                        PnP_module.hparams.lamb, PnP_module.hparams.sigma_denoiser, PnP_module.hparams.alpha, PnP_module.hparams.alpha_pd = get_gaussian_noise_parameters(hparams.PnP_algo, hparams.noise_level_img, k_index=k_index, degradation_mode='SR')
                    else:
                        raise ValueError('Noise model not implemented')
                

                if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                    exp_out_path = create_out_dir(hparams.degradation_mode, hparams.dataset_name, PnP_algo)

                np.random.seed(seed=0)
            
                    
                for i in range(min(len(input_paths),hparams.n_images)): # For each image
                #for i in range(1,2): # For each image

                    print('SR of image {}, sf={}, kernel index {}, with PnP-{} algorithm'.format(i, sf, k_index,PnP_algo))

                    ## load image
                    input_im_uint = imread_uint(input_paths[i])
                    if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
                        input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
                    input_im = np.float32(input_im_uint / 255.)                    
                    if hparams.grayscale : 
                        input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
                        input_im = np.expand_dims(input_im, axis = 2)

                    # Degrade image
                    np.random.seed(seed=0)
                    blur_im = modcrop(input_im, sf)
                    blur_im = numpy_degradation(input_im, k, sf)
                    if hparams.noise_model == 'gaussian' :
                        noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
                        blur_im += noise
                    else:
                        raise ValueError('Noise model not implemented')
                    
                    # Initialize the algorithm
                    if not hparams.use_linear_init:
                        init_im = cv2.resize(blur_im, (int(blur_im.shape[1] * sf), int(blur_im.shape[0] * sf)),interpolation=cv2.INTER_CUBIC)
                        init_im = shift_pixel(init_im, sf)
                    else:
                        init_im = blur_im
                    PnP_module.sf = sf

                    # PnP restoration
                    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                        deblur_im, init_im, output_psnr, n_it, x_list, z_list, y_list, Dg_list, psnr_tab, g_list, F_list, Psi_list, f_list, Df_list = PnP_module.restore(blur_im,init_im,input_im, k, extract_results=True, sf=sf)
                    else :
                        deblur_im, init_im, output_psnr, n_it = PnP_module.restore(blur_im,init_im,input_im,k, sf=sf)

                    print('PSNR: {:.2f}dB'.format(output_psnr))
                print(f'N iterations: {n_it}')
               
                psnr_k_list.append(output_psnr)
                psnr_list.append(output_psnr)
                n_it_list.append(n_it)

                if hparams.extract_curves:
                    # Create curves
                    PnP_module.update_curves(x_list, psnr_tab, Dg_list, g_list, F_list, Psi_list, f_list, Df_list)

                if hparams.extract_images:
                    # Save images
                    save_im_path = os.path.join(exp_out_path, 'images')
                    if not os.path.exists(save_im_path):
                        os.mkdir(save_im_path)
                    imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
                    imsave(os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'), single2uint(rescale(deblur_im)))
                    imsave(os.path.join(save_im_path, 'img_'+str(i)+'_blur.png'), single2uint(rescale(blur_im)))
                    imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(rescale(init_im)))
                    print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(exp_out_path,'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        print('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))

        data.append([k_index, avg_k_psnr, np.mean(np.mean(n_it_list))])

    data = np.array(data)

    if hparams.use_wandb :
        table = wandb.Table(data=data, columns=['k', 'psnr', 'n_it'])
        for i, metric in enumerate(['psnr', 'n_it']):
            wandb.log({
                f'{metric}_plot': wandb.plot.scatter(
                    table, 'k', metric,
                    title=f'{metric} vs. k'),
                f'average_{metric}': np.mean(data[:,i+1])
            },
            step = 0)

    return np.mean(np.array(psnr_list))

if __name__ == '__main__':
    SR()
