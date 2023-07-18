import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim import lr_scheduler
import random
from argparse import ArgumentParser
import cv2
import torchvision
import numpy as np
from test_utils import test_mode
import matplotlib.pyplot as plt
from GS_utils import normalize_min_max
from models.network_unet import UNetRes
from models import DNCNN
from models.FFDNET import FFDNet
from GS_utils import psnr, burg_bregman_divergence, KL_divergence, normalize_min_max
import sys

sys.path.append("..")
from PnP_restoration.utils.utils_restoration import (
    tensor2array,
    imsave,
    single2uint,
    rescale,
)


class StudentGrad(pl.LightningModule):
    """
    Standard Denoiser model
    """

    def __init__(
        self,
        model_name,
        pretrained,
        pretrained_checkpoint,
        act_mode,
        DRUNET_nb,
        residual_learning,
        grayscale,
        constant_noise_map,
    ):
        super().__init__()
        self.model_name = model_name
        self.residual_learning = residual_learning
        self.constant_noise_map = constant_noise_map
        if self.model_name == "DRUNET":
            if grayscale:
                in_nc = 2
                out_nc = 1
            else:
                in_nc = 4
                out_nc = 3
                if not constant_noise_map:
                    in_nc = 6
            self.model = UNetRes(
                in_nc=in_nc,
                out_nc=out_nc,
                nc=[64, 128, 256, 512],
                nb=DRUNET_nb,
                act_mode=act_mode,
                downsample_mode="strideconv",
                upsample_mode="convtranspose",
            )
        elif self.model_name == "DNCNN":
            self.model = DNCNN.dncnn(3, 3, 17, "C", act_mode, False)
        elif self.model_name == "FFDNET":
            self.model = FFDNet(3, 3, 64, 15, act_mode=act_mode)
        self.model.to(self.device)
        if pretrained:
            checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for key, val in state_dict.items():
                new_state_dict[key[6:]] = val
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x, sigma, constant_noise_map=True):
        if self.model_name == "FFDNET":
            n = self.model(x, torch.full((x.shape[0], 1, 1, 1), sigma).type_as(x))
        else:
            if self.model_name == "DRUNET":
                if self.constant_noise_map:
                    noise_level_map = (
                        torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3))
                        .fill_(sigma)
                        .to(self.device)
                    )
                else:
                    noise_level_map = sigma.to(self.device)
                x = torch.cat((x, noise_level_map), 1)
            n = self.model(x)
        if self.residual_learning:
            return x - n
        else:
            return n


class GradMatch(pl.LightningModule):
    """
    Gradient Step Denoiser
    """

    def __init__(self, hparams):
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters(hparams)
        self.student_grad = StudentGrad(
            self.hparams.model_name,
            self.hparams.pretrained_student,
            self.hparams.pretrained_checkpoint,
            self.hparams.act_mode,
            self.hparams.DRUNET_nb,
            self.hparams.residual_learning,
            self.hparams.grayscale,
            self.hparams.constant_noise_map,
        )

    @torch.enable_grad()
    @torch.inference_mode(False)
    def calculate_grad(self, x, sigma):
        """
        Calculate Dg(x) the gradient of the regularizer g at input x
        :param x: torch.tensor Input image
        :param sigma: Denoiser level (std)
        :return: Dg(x) gradient of the regularizer g at x
        :return: N(x) value of the model N at x
        :return: g(x) value of the regularizer g at x
        """
        x = x.float().clone().requires_grad_()
        if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
            N = self.student_grad.forward(x, sigma)
        else:
            current_model = lambda v: self.student_grad.forward(v, sigma)
            N = test_mode(current_model, x, mode=5, refield=64, min_size=256)
        # The argument `bregman_div_g` defines which Bregman divergence is used to define the regularizer g.
        if self.hparams.bregman_div_g == "L2":  # g(x) = 0.5*||x-N(x)||^2
            g = 0.5 * torch.sum((x - N).reshape((x.shape[0], -1)) ** 2)
        elif self.hparams.bregman_div_g == "L2_N":  # g(x) = 0.5*||N(x)||^2
            g = 0.5 * torch.sum(N.reshape((x.shape[0], -1)) ** 2)
        elif (
            self.hparams.bregman_div_g == "burg"
        ):  # g(x) = D_h(N(x),x) with h(x) = - \log(x)
            g = burg_bregman_divergence(N, x)
        elif self.hparams.bregman_div_g == "KL":  # g(x) = KL(x,N(x))
            g = KL_divergence(x, N)
        Dg = torch.autograd.grad(
            g, x, torch.ones_like(g), create_graph=True, only_inputs=True
        )[0]
        return Dg, N, g

    def forward(self, x, sigma):
        """
        Denoising with Gradient Step Denoiser
        :param x:  torch.tensor input image
        :param sigma: Denoiser level
        :return: Denoised image x_hat, Dg(x) gradient of the regularizer g at x
        :return: Dg(x) gradient of the regularizer g at x
        :return: g(x) value of the regularizer g at x
        """
        if self.hparams.grad_matching:  # If gradient step denoising
            Dg, N, g = self.calculate_grad(x, sigma)
            weight_Dg = self.hparams.weight_Dg
            if (
                self.hparams.bregman_denoiser == "burg"
            ):  # Bremgan gradient descent step with Burg's entropy Bregman divergence
                if self.hparams.sigma_step:
                    weight_Dg = weight_Dg * (1 / sigma)
                x_hat = x - weight_Dg * (x ** 2) * Dg
            elif self.hparams.bregman_denoiser == "L2":  # Standart gradient step
                if self.hparams.sigma_step:
                    weight_Dg = weight_Dg * sigma
                x_hat = x - weight_Dg * Dg
        else:  # If denoising with standard forward CNN
            x_hat = self.student_grad.forward(x, sigma)
            N = x_hat
            Dg = x - x_hat
            g = 0.5 * torch.sum((x - N).reshape((x.shape[0], -1)) ** 2)
        return x_hat, Dg, g

    def lossfn(self, x, y):
        """
        Loss function for training the denoiser
        :param x: Estimated image
        :param y: Ground Truth image
        :return: Loss value
        """
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)
        if self.hparams.bregman_div_loss == "L2":
            loss = nn.MSELoss(reduction="none")(x, y).mean(dim=1)
        elif self.hparams.bregman_div_loss == "burg":
            loss = burg_bregman_divergence(x, y) / (x.shape[1])
        elif self.hparams.bregman_div_loss == "KL":
            loss = KL_divergence(x, y) / (x.shape[1])
        return loss

    def training_step(self, batch, batch_idx):
        y, _ = batch
        y = torch.clamp(
            y, 0.001, y.max()
        )  # Burg's entropy defined for positive values.
        invgamma = random.uniform(
            self.hparams.min_invgamma_train, self.hparams.max_invgamma_train
        )  # noise level gamma
        gamma = 1 / invgamma
        if not self.hparams.constant_noise_map:
            sigma_model = y * (gamma / (gamma - 2)) * np.sqrt(1 / (gamma - 3))
        else:
            sigma_model = invgamma
        if (
            self.hparams.gaussian_noise_pre
        ):  # Add Gaussian noise before inverse gamma noise
            sigma_gauss = (
                random.uniform(
                    self.hparams.min_sigma_gauss, self.hparams.max_sigma_gauss
                )
                / 255
            )
            u = torch.randn(y.size(), device=self.device)
            y = (y + u * sigma_gauss).clamp(0.001, 10)
        m = torch.distributions.gamma.Gamma(gamma - 1, gamma * y)
        z = m.sample().to(self.device)  # z follows gamma distribution
        x = 1 / z  # x follows inverse gamma distribution
        if (
            self.hparams.gaussian_noise_post
        ):  # Add Gaussian noise after inverse gamma noise
            sigma_gauss = (
                random.uniform(
                    self.hparams.min_sigma_gauss, self.hparams.max_sigma_gauss
                )
                / 255
            )
            u = torch.randn(y.size(), device=self.device)
            x = (x + u * sigma_gauss).clamp(0.001, 10)

        x_hat, _, _ = self.forward(x, sigma_model)
        loss = self.lossfn(x_hat, y)
        train_PSNR = psnr(x_hat, y)

        loss = loss.mean()

        self.log("train/train_loss", loss.detach(), on_epoch=True)
        self.log("train/train_psnr", train_PSNR.detach(), prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(
                normalize_min_max(x_hat.detach())[:1]
            )
            self.logger.experiment.add_image(
                "train/noisy", noisy_grid, self.current_epoch
            )
            self.logger.experiment.add_image(
                "train/denoised", denoised_grid, self.current_epoch
            )

        return loss

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        torch.manual_seed(0)
        y, _ = batch
        gamma_list = self.hparams.gamma_list_test
        for i, gamma in enumerate(gamma_list):
            y = torch.clamp(y, 0.001, y.max())
            if not self.hparams.constant_noise_map:
                sigma_model = y * (gamma / (gamma - 2)) * np.sqrt(1 / (gamma - 3))
            else:
                sigma_model = 1 / gamma
            m = torch.distributions.gamma.Gamma(gamma - 1, gamma * y)
            z = m.sample().to(self.device)
            x = 1 / z
            if (
                self.hparams.use_sigma_model
            ):  # Possibility to test with sigma model different than input sigma
                sigma_model = self.hparams.sigma_model
            if self.hparams.grad_matching:  # GS denoise
                torch.set_grad_enabled(True)
                x_hat = x
                for n in range(self.hparams.n_step_eval):  # 1 step in practice
                    current_model = lambda v: self.forward(v, sigma_model)
                    x_hat, Dg, g = current_model(x_hat)
                l = self.lossfn(x_hat, y).mean()
                p = psnr(x_hat, y).mean()
                Dg_norm = torch.norm(Dg, p=2)
            else:
                for n in range(self.hparams.n_step_eval):
                    current_model = lambda v: self.forward(v, sigma_model)[0]
                    x_hat = x
                    if x.size(2) % 8 == 0 and x.size(3) % 8 == 0:
                        x_hat, Dg, g = current_model(x_hat)
                    elif x.size(2) % 8 != 0 or x.size(3) % 8 != 0:
                        x_hat, Dg, g = test_mode(
                            current_model, x_hat, refield=64, mode=5
                        )
                Dg_norm = torch.norm(Dg, p=2)
                l = self.lossfn(x_hat, y)
                p = psnr(x_hat, y)

            if self.hparams.test_convexity:
                min_convexity_gap, mean_convexity_gap = self.test_convexity(
                    y,
                    sigma_model,
                    interpolation=True,
                    noise_model=self.hparams.test_convexity_noise_model,
                )
                self.log(
                    "min_convexity_gap_" + str(gamma),
                    min_convexity_gap,
                    reduce_fx=torch.min,
                )
                self.log("mean_convexity_gap_" + str(gamma), mean_convexity_gap)

            self.log("psnr_" + str(gamma), p.detach())
            self.log("loss_" + str(gamma), l.detach())
            self.log("g_" + str(gamma), g.detach())
            self.log("Dg_norm_" + str(gamma), Dg_norm.detach())

            if self.hparams.save_images:
                print("psnr noisy", gamma, psnr(x, y))
                print("psnr denoised", gamma, psnr(x_hat, y))
                save_dir = "images/" + self.hparams.name
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                    os.mkdir(save_dir + "/noisy")
                    os.mkdir(save_dir + "/denoised")
                    os.mkdir(save_dir + "/denoised_no_noise")
                    os.mkdir(save_dir + "/clean")
                for i in range(len(x)):
                    clean = tensor2array(y[i].detach().cpu())
                    noisy = tensor2array(x[i].detach().cpu())
                    denoised = tensor2array(x_hat[i].detach().cpu())
                    imsave(
                        save_dir
                        + "/denoised/"
                        + str(batch_idx)
                        + "_sigma_"
                        + str(gamma)
                        + ".png",
                        single2uint(rescale(denoised)),
                    )
                    imsave(
                        save_dir
                        + "/clean/"
                        + str(batch_idx)
                        + "_sigma_"
                        + str(gamma)
                        + ".png",
                        single2uint(rescale(clean)),
                    )
                    imsave(
                        save_dir
                        + "/noisy/"
                        + str(batch_idx)
                        + "_sigma_"
                        + str(gamma)
                        + ".png",
                        single2uint(rescale(noisy)),
                    )

        if batch_idx == 0:  # logging for tensorboard
            clean_grid = torchvision.utils.make_grid(normalize_min_max(y.detach())[:1])
            noisy_grid = torchvision.utils.make_grid(normalize_min_max(x.detach())[:1])
            denoised_grid = torchvision.utils.make_grid(
                normalize_min_max(x_hat.detach())[:1]
            )
            self.logger.experiment.add_image(
                "val/clean", clean_grid, self.current_epoch
            )
            self.logger.experiment.add_image(
                "val/noisy", noisy_grid, self.current_epoch
            )
            self.logger.experiment.add_image(
                "val/denoised", denoised_grid, self.current_epoch
            )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optim_params = []
        for k, v in self.student_grad.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                print("Params [{:s}] will not optimize.".format(k))
        optimizer = Adam(optim_params, lr=self.hparams.optimizer_lr, weight_decay=0)
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, self.hparams.scheduler_milestones, self.hparams.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def test_convexity(
        self,
        y,
        sigma,
        n_test=10,
        interpolation=False,
        noise_model="gaussian",
        verbose=False,
    ):
        if noise_model == "gaussian":
            sigma_in = sigma / 255.0
            noise_in = torch.randn(y.size(), device=self.device) * sigma_in
            x = y + noise_in
        elif noise_model == "inv_gamma":
            y = torch.clamp(y, 0.01, y.max())
            m = torch.distributions.gamma.Gamma(sigma - 1, sigma * y)
            z = m.sample().to(self.device)
            x = 1 / z
        elif noise_model == "poisson":
            y = torch.clamp(y, 0.01, y.max())
            x = torch.poisson(sigma * y) / sigma
        if interpolation:
            x_hat, _, _ = self.forward(x, sigma)
            eta = torch.rand(y.size(0), 1, 1, 1, requires_grad=True).to(self.device)
            x = eta * y.detach() + (1 - eta) * x_hat.detach()
            x = x.to(self.device)
        with torch.inference_mode(False):
            x = x.clone().requires_grad_()
            x_hat, Dg, g = self.forward(x, sigma)
            operator = lambda vec: torch.autograd.grad(
                Dg,
                x,
                grad_outputs=vec,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            n_fail = 0
            sup_x = (1 - 2 * x * Dg) / (x ** 2)
            error_min = []
            error_mean = []
            for i in range(n_test):
                d = torch.rand_like(y)
                ps = (operator(d) * d * (x ** 4)).view(y.shape[0], -1).sum(dim=-1)
                sup = (sup_x * (d ** 2) * (x ** 4)).view(y.shape[0], -1).sum(dim=-1)
                gap = sup - ps
                error_min.append(gap.min().detach().cpu().item())
                error_mean.append(gap.mean().detach().cpu().item())
                for k in range(len(ps)):
                    if ps[k] > sup[k]:
                        n_fail += 1
            min_error = torch.min(torch.tensor(error_min))
            mean_error = torch.mean(torch.tensor(error_mean))
            if verbose:
                if n_fail > 0:
                    print(
                        f"FAIL : image convexity test for sigma={sigma:0.01f} on {n_fail} random d out of {n_test*len(y)}"
                    )
                else:
                    print(
                        f"SUCCESS : batch convexity test for sigma={sigma:0.01f} with min_error={min_error.mean():0.0001f} / mean_error={mean_error.mean():0.0001f}"
                    )
            return min_error, mean_error

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name", type=str, default="DRUNET")
        parser.add_argument(
            "--start_from_checkpoint", dest="start_from_checkpoint", action="store_true"
        )
        parser.set_defaults(start_from_checkpoint=False)
        parser.add_argument(
            "--resume_from_checkpoint",
            dest="resume_from_checkpoint",
            action="store_true",
        )
        parser.set_defaults(resume_from_checkpoint=False)
        parser.add_argument(
            "--pretrained_checkpoint", type=str, default="ckpts/inv_gamma.ckpt"
        )
        parser.add_argument(
            "--pretrained_student", dest="pretrained_student", action="store_true"
        )
        parser.set_defaults(pretrained_student=False)
        parser.add_argument("--n_channels", type=int, default=3)
        parser.add_argument("--nc_in", type=int, default=3)
        parser.add_argument("--nc_out", type=int, default=3)
        parser.add_argument("--nc", type=int, default=64)
        parser.add_argument("--nb", type=int, default=20)
        parser.add_argument("--act_mode", type=str, default="s")
        parser.add_argument("--no_bias", dest="no_bias", action="store_false")
        parser.set_defaults(use_bias=True)
        parser.add_argument("--DRUNET_nb", type=int, default=2)
        parser.set_defaults(power_method_mean_correction=False)
        parser.add_argument(
            "--no_grad_matching", dest="grad_matching", action="store_false"
        )
        parser.set_defaults(grad_matching=True)
        parser.add_argument("--weight_Dg", type=float, default=1.0)
        parser.add_argument(
            "--residual_learning", dest="residual_learning", action="store_true"
        )
        parser.set_defaults(residual_learning=False)
        parser.add_argument("--grayscale", dest="grayscale", action="store_true")
        parser.set_defaults(grayscale=False)
        return parser

    @staticmethod
    def add_optim_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--optimizer_type", type=str, default="adam")
        parser.add_argument("--optimizer_lr", type=float, default=1e-4)
        parser.add_argument("--gradient_clip_val", type=float, default=1e-2)
        parser.add_argument("--scheduler_type", type=str, default="MultiStepLR")
        parser.add_argument(
            "--scheduler_milestones", type=int, nargs="+", default=[300, 600, 900, 1200]
        )
        parser.add_argument("--scheduler_gamma", type=float, default=0.5)
        parser.add_argument("--early_stopping_patiente", type=int, default=5)
        parser.add_argument("--val_check_interval", type=float, default=1.0)
        parser.add_argument("--min_invgamma_train", type=float, default=0)
        parser.add_argument("--max_invgamma_train", type=float, default=0.1)
        parser.add_argument(
            "--gamma_list_test", type=float, nargs="+", default=[50, 75, 100]
        )
        parser.add_argument("--sigma_step", dest="sigma_step", action="store_true")
        parser.set_defaults(sigma_step=False)
        parser.add_argument(
            "--test_convexity", dest="test_convexity", action="store_true"
        )
        parser.set_defaults(test_convexity=False)
        parser.add_argument(
            "--test_convexity_noise_model", type=str, default="gaussian"
        )
        parser.add_argument("--n_step_eval", type=int, default=1)
        parser.add_argument(
            "--use_post_forward_clip", dest="use_post_forward_clip", action="store_true"
        )
        parser.set_defaults(use_post_forward_clip=False)
        parser.add_argument(
            "--use_sigma_model", dest="use_sigma_model", action="store_true"
        )
        parser.set_defaults(use_sigma_model=False)
        parser.add_argument("--sigma_model", type=int)
        parser.add_argument("--bregman_div_g", type=str, default="L2")
        parser.add_argument("--bregman_div_loss", type=str, default="L2")
        parser.add_argument("--bregman_denoiser", type=str, default="burg")
        parser.add_argument(
            "--no_constant_noise_map", dest="constant_noise_map", action="store_false"
        )
        parser.set_defaults(constant_noise_map=True)
        parser.add_argument(
            "--gaussian_noise_pre", dest="gaussian_noise_pre", action="store_true"
        )
        parser.set_defaults(gaussian_noise_pre=False)
        parser.add_argument(
            "--gaussian_noise_post", dest="gaussian_noise_post", action="store_true"
        )
        parser.set_defaults(gaussian_noise_post=False)
        parser.add_argument("--max_sigma_gauss", type=float, default=7.65)
        parser.add_argument("--min_sigma_gauss", type=float, default=7.65)
        return parser
