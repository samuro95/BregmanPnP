# Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems

Implementation of the [paper](https://arxiv.org/pdf/2306.03466.pdf)] "Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems" presented at Neurips 2023. 

[Samuel Hurault]([https://www.math.u-bordeaux.fr/~shurault/](https://samuelhurault.netlify.app/)), [Arthur Leclaire](https://www.math.u-bordeaux.fr/~aleclaire/), [Nicolas Papadakis](https://www.math.u-bordeaux.fr/~npapadak/). \
[Institut de Mathématiques de Bordeaux](https://www.math.u-bordeaux.fr/imb/spip.php), France.


## Prerequisites


The code was computed with Python 3.8.10, PyTorch Lightning 1.2.6, PyTorch 1.7.1

```
pip install -r requirements.txt
```

### Training of the Inverse Gamma noise Bregman denoiser.

- Download [training dataset](https://plmbox.math.cnrs.fr/f/4f56db2f0f7d49a88663/?dl=1) and unzip ```DRUNET``` in the ```datasets``` folder
- Realize training
```
cd GS_denoising
python main_train.py --name experiment_name --log_folder logs
```
The model is trained to denoiser Inverse Gamma noise with parameter $\gamma$. The inverse $1 / \gamma$ is randomly unformly sampled during training between ```min_invgamma_train``` and ```max_invgamma_train``` . By default these paremeters are respectively set to $0$ and $0.1$.

Checkpoints, tensorboard events and hyperparameters will be saved in the ```GS_denoising/logs/experiment_name``` subfolder. 

### Testing of the denoiser

- For directly testing the model, download  [pretrained checkpoint](https://plmbox.math.cnrs.fr/f/c5574b42bdc146d08844/?dl=1) and save it as ```GS_denoising/ckpts/inv_gamma.ckpt```
- For denoising a set of (clean) images. Place your images in a directory of the form ```datasets/DATASET_NAME/0/```. 
- For testing the denoiser on these images at different noise levels $\gamma$ (here $50$, $75$ and $100$).
```
cd GS_denoising
python main_test.py --gamma_list_test 50, 75, 100 -- dataset_name DATASET_NAME
```
Datasets CBSD68, CBSD10, set3c are already present in the directory. Default value is CBSD10. 


### Bregman PnP and RED for Poisson image deblurring 

- Download  [pretrained checkpoint](https://plmbox.math.cnrs.fr/f/c5574b42bdc146d08844/?dl=1) and save it as ```GS_denoising/ckpts/inv_gamma.ckpt``` .

- For deblurring an input (clean) image ```IMAGE_PATH```, that we blur with kernel saved at ```KERNEL_PATH``` (saved as ```.npy```) and with Inverse Gamma noise at level ```NOISE_LEVEL``` 
```
cd PnP_restoration
python deblur.py --image_path IMAGE_PATH --kernel_path KERNEL_PATH --noise_level_img NOISE_LEVEL
```

By default, without specifying ```--kernel_path ```, deblurring will be performed on the 10 kernels evaluated in the paper. You can specify  ```--kernel_index``` to choose a specific kernel in this list. 

You can also specify ```--dataset_name``` to treat a set of images places in directory ```datasets/DATASET_NAME``` 

Add the argument ```--extract_curves``` and ```--extract_images``` the save convergence curves and the output images.






## Acknowledgments

This repo contains parts of code taken from : 
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 
- Gradient Step Denoiser for convergent Plug-and-Play (GS-PnP) : https://github.com/samuro95/GSPnP

## Citation 
```
@article{hurault2023convergent,
  title={Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems},
  author={Hurault, Samuel and Kamilov, Ulugbek and Leclaire, Arthur and Papadakis, Nicolas},
  journal={arXiv preprint arXiv:2306.03466},
  year={2023}
}

```
