import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
import torch
from argparse import ArgumentParser
from utils.utils_restoration import burg_bregman_divergence, burg_bregman_divergence_np, psnr, array2tensor, tensor2array
import sys
from matplotlib.ticker import MaxNLocator


class PnP_restoration():

    def __init__(self, hparams):

        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_denoiser()

    def initialize_denoiser(self):
        '''
        Initialize the denoiser model with the given pretrained ckpt
        '''
        sys.path.append('../GS_denoising/')
        from lightning_denoiser import GradMatch
        parser2 = ArgumentParser(prog='utils_restoration.py')
        parser2 = GradMatch.add_model_specific_args(parser2)
        parser2 = GradMatch.add_optim_specific_args(parser2)
        hparams = parser2.parse_known_args()[0]
        if self.hparams.bregman_potential == 'burg' :
            if not self.hparams.grad_matching:
                pretrained_checkpoint = '../GS_denoising/ckpts/no_grad_matching.ckpt'
            elif self.hparams.grayscale:
                pretrained_checkpoint = '../GS_denoising/ckpts/burg_gray.ckpt'
                hparams.grayscale = True
                self.hparams.inv_gamma = False
            else :
                pretrained_checkpoint = '../GS_denoising/ckpts/inv_gamma.ckpt'
                self.hparams.inv_gamma = True
        else: 
            self.hparams.use_GSDRUNET_ckpt = True
        if self.hparams.use_GSDRUNET_ckpt:
            self.hparams.inv_gamma = False
            pretrained_checkpoint = '../GS_denoising/ckpts/GS-DRUnet_S.ckpt'
        if self.hparams.pretrained_checkpoint is not None : 
            pretrained_checkpoint = self.hparams.pretrained_checkpoint
        hparams.grad_matching = self.hparams.grad_matching
        hparams.bregman_div_g = self.hparams.bregman_div_g
        hparams.act_mode = 's'
        self.denoiser_model = GradMatch(hparams)
        checkpoint = torch.load(pretrained_checkpoint, map_location=self.device)
        self.denoiser_model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.denoiser_model.eval()
        for i, v in self.denoiser_model.named_parameters():
            v.requires_grad = False
        self.denoiser_model = self.denoiser_model.to(self.device)

    def denoise(self, x, sigma, weight=1.):
        '''
        Denoising step
        :param x: Input image
        :param sigma: Noise level
        :param weight: multiplicative weight in front of the gradient in the gradient descent step. 
        :return: Denoised image, regularization function value, regularization function gradient.
        '''
        if self.hparams.grad_matching # gradient step denoising
            torch.set_grad_enabled(True)
            if self.hparams.inv_gamma : # if the denoiser is trained with the inverse of gamma as second input
                Dg, N, g = self.denoiser_model.calculate_grad(x, 1/sigma)
            else :
                Dg, N, g = self.denoiser_model.calculate_grad(x, sigma)
            torch.set_grad_enabled(False)
            Dg = Dg.detach()
            N = N.detach()
            if self.hparams.sigma_step : 
                Dg = (self.hparams.weight_Dg/sigma)*Dg
            if self.hparams.bregman_potential == 'L2':
                Dx = x - weight * Dg
            elif self.hparams.bregman_potential == 'burg':
                Dx = x - weight * (x**2) * Dg
            else :
                ValueError('noise model not treated')
        else : # if no gradient step denoising, just apply the denoising network
            if self.hparams.inv_gamma :
                N = self.denoiser_model.student_grad.forward(x, 1/sigma)
            else :
                N = self.denoiser_model.student_grad.forward(x, sigma)
            Dx = N
            Dg = x - N
            g = 0.5 * (torch.norm(x - N, p=2) ** 2)
        return Dx, g, Dg

    def N(self, x, sigma):
        N = self.denoiser_model.student_grad.forward(x, sigma)
        return N

    def JNx_z(self, x, z, sigma):
        '''
        Calculate the Jacobian-vector product of with the Jacobian of N(x) with z. 
        '''
        torch.set_grad_enabled(True)
        x = x.requires_grad_()
        Nx = self.N(x,sigma)
        JNz = torch.autograd.grad(Nx, x, grad_outputs=z, create_graph=True, only_inputs=True)[0]
        torch.set_grad_enabled(False)
        return JNz


    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.hparams.degradation_mode == 'deblurring' or self.hparams.degradation_mode == 'SR':
            k = degradation
            self.k_tensor = torch.tensor(k).to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(img, self.k_tensor, self.hparams.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            self.M = array2tensor(degradation).to(self.device)


    def data_fidelity_prox_step(self, x, y, stepsize):
        '''
        Calculation of the proximal step on the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            if self.hparams.degradation_mode == 'deblurring' or self.hparams.degradation_mode == 'SR':
                px = utils_sr.prox_solution_L2(x, self.FB, self.FBC, self.F2B, self.FBFy, stepsize, self.hparams.sf)
            elif self.hparams.degradation_mode == 'inpainting':
                if self.hparams.noise_level_img > 1e-2:
                    px = (stepsize*self.M*y + x)/(stepsize*self.M+1)
                else :
                    px = self.M*y + (1-self.M)*x
            else:
                ValueError('Degradation not treated')
        elif self.hparams.noise_model == 'poisson':
            ValueError('Prox not implemented for Poisson noise')
        else :  
            ValueError('noise model not treated')
        return px

    def data_fidelity_grad(self, x, y):
        '''
        Calculation of the gradient of the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            return utils_sr.grad_solution_L2(x, y, self.k_tensor, self.hparams.sf)
        elif self.hparams.noise_model == 'poisson':
            return utils_sr.grad_solution_KL(x, y, self.k_tensor, self.hparams.sf, self.hparams.alpha_poisson)
        else:
            raise ValueError('noise model not implemented')


    def data_fidelity_grad_step(self, x, y, stepsize):
        '''
        Calculation of the gradient descent step on the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            grad = utils_sr.grad_solution_L2(x, y, self.k_tensor, self.hparams.sf)
        elif self.hparams.noise_model == 'poisson':
            grad = utils_sr.grad_solution_KL(x, y, self.k_tensor, self.hparams.sf, self.hparams.alpha_poisson)
        else:
            raise ValueError('noise model not implemented')
        if self.hparams.bregman_potential == 'L2':
            return x - stepsize*grad, grad
        elif self.hparams.bregman_potential == 'burg':
            return x / (1 + x*stepsize*grad), grad
        else :
            raise ValueError('Bregman potential not implemented')


    def A(self,y):
        '''
        Calculation A*x with A the linear degradation operator 
        '''
        if self.hparams.degradation_mode == 'deblurring':
            y = utils_sr.G(y, self.k_tensor, sf=1)
        elif self.hparams.degradation_mode == 'SR':
            y = utils_sr.G(y, self.k_tensor, sf=self.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            y = self.M * y
        else:
            raise ValueError('degradation not implemented')
        return y  

    def At(self,y):
        '''
        Calculation A*x with A the linear degradation operator 
        '''
        if self.hparams.degradation_mode == 'deblurring':
            y = utils_sr.Gt(y, self.k_tensor, sf=1)
        elif self.hparams.degradation_mode == 'SR':
            y = utils_sr.Gt(y, self.k_tensor, sf=self.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            y = self.M * y
        else:
            raise ValueError('degradation not implemented')
        return y  


    def calulate_data_term(self,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
        deg_y = self.A(y)
        if self.hparams.noise_model == 'gaussian':
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.noise_model == 'poisson':
            f = (img*torch.log(img/deg_y + 1e-15) + deg_y - img).sum()
        return f

    def calculate_regul(self,y, x, g=None, prox_pnp = True):
        '''
        Calculation of the regularization phi_sigma(y)
        :param y: Point where to evaluate
        :param x: D^{-1}(y)
        :param g: Precomputed regularization function value at x
        :return: regul(y)
        '''
        if g is None:
            _,g,_ = self.denoise(x, self.sigma_denoiser)
        if not prox_pnp : 
            return self.alpha * g 
        else :
            if self.hparams.bregman_potential == 'L2':
                return self.alpha * g - (1 / 2) * torch.norm(x - y, p=2) ** 2
            elif self.hparams.bregman_potential == 'burg':
                return self.alpha * g - burg_bregman_divergence(y,x)

    def calculate_F(self, y, x, img, g = None, prox_pnp = True):
        '''
        Calculation of the objective function value lamb*f(y) + phi_sigma(y)
        :param y: Point where to evaluate F
        :param x: D^{-1}(y)
        :param img: Degraded image
        :param g: Precomputed regularization function value at x
        :return: F(y)
        '''
        regul = self.calculate_regul(y, x, g, prox_pnp = prox_pnp)
        if self.hparams.no_data_term:
            F = regul
            f = torch.zeros_like(F)
        else:
            f = self.calulate_data_term(y,img)
            F = self.lamb * f + regul
        return f.item(), F.item()


    def calculate_lyapunov_DRS(self,y,z,x,g,img):
        '''
            Calculation of the Lyapunov function value Psi(x)
            :param x: Point where to evaluate F
            :param y,z: DRS iterations initialized at x
            :param g: Precomputed regularization function value at x
            :param img: Degraded image
            :return: Psi(x)
        '''
        regul = self.calculate_regul(y,x,g)
        f = self.calulate_data_term(z, img)
        Psi = regul + f + (1 / self.lamb) * (torch.sum(torch.mul(y-x,y-z)) + (1/2) * torch.norm(y - z, p=2) ** 2)
        return Psi.item()


    def restore(self, img, init_im, clean_img, degradation, extract_results=False, sf=1):
        '''
        Compute Prox-PnP restoration algorithm
        :param img: Degraded image
        :param init_im: Initialization of the algorithm
        :param clean_img: ground-truth clean image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        :param extract_results: Extract information for subsequent image or curve saving
        :param sf: Super-resolution factor
        '''
        
        self.hparams.sf = sf

        if extract_results:
            y_list, z_list, x_list, Dg_list, psnr_tab, g_list, f_list, Df_list, F_list, Psi_list = [], [], [], [], [], [], [], [], [], []

        i = 0 # iteration counter

        img_tensor = array2tensor(img).to(self.device)

        # some prox and grad calculus that can be done outside of the loop
        self.initialize_prox(img_tensor, degradation)  

        # Initialization of the algorithm
        x0 = array2tensor(init_im).to(self.device)
        if self.hparams.use_linear_init : 
            x0 = self.At(x0)
        if self.hparams.bregman_potential == 'burg' : # Burg entropy is only define with strictly positive values
            x0 = torch.clamp(x0, 0.001, x0.max())
        if self.hparams.use_hard_constraint:
            x0 = torch.clamp(x0, 0, 1)
        x = x0
        y = x0

        if extract_results:  # extract numpy images and PSNR values
            out_x = tensor2array(x0.cpu())
            current_x_psnr = psnr(clean_img, out_x)
            if self.hparams.print_each_step:
                print('current x PSNR : ', current_x_psnr)
            if self.hparams.lim_init == 0 : # if no initialization phase, save the initialization
                psnr_tab.append(current_x_psnr)
                x_list.append(out_x)

        # Initialize function values
        diff_Psi = 1
        Psi_old = 1
        Psi = Psi_old
        F = 1
        f = 1
        self.backtracking_check = True

        if self.hparams.lim_init == 0 : # if no initialization phase, initialize the parameters with the algorithm parameter values.
            init = False # init is True if we are in the initialization phase at the current iteration
            prev_init = False # prev_init is True if we were in the initialization phase at the previous iteration
            self.tau = self.hparams.stepsize
            self.lamb = self.hparams.lamb
            self.sigma_denoiser = self.hparams.sigma_denoiser
            self.alpha = self.hparams.alpha
        else : # if initialization phase, initialize the parameters with the initialization parameter values.
            init = True
            prev_init = True
            self.tau = self.hparams.stepsize_pre_init
            self.lamb = self.hparams.lamb_pre_init
            self.sigma_denoiser = self.hparams.sigma_denoiser_pre_init
            self.alpha = self.hparams.alpha_pre_init

        while  i < self.hparams.maxitr: # for each iteration

            if (not init and prev_init): # if initialization phase is over, set the parameters to the algorithm parameter values.
                init = False
                self.tau = self.hparams.stepsize
                self.lamb = self.hparams.lamb
                self.sigma_denoiser = self.hparams.sigma_denoiser
                self.alpha = self.hparams.alpha

            if init : # if initialization phase, no backtracking nor early stopping
                use_backtracking = False
                early_stopping = False
            else :
                use_backtracking = self.hparams.use_backtracking
                early_stopping = self.hparams.early_stopping
            
            prev_init = init
            x_old = x
            y_old = y
            Psi_old = Psi
            F_old = F
            f_old = f

            if self.hparams.PnP_algo == 'GD_RED': 
                # Gradient of the data fidelity term
                Df = self.data_fidelity_grad(x_old, img_tensor)
                # Gradient of the regularization term
                _,g,Dg = self.denoise(x_old, self.sigma_denoiser)
                # Gradient step
                if self.hparams.bregman_potential == 'L2':
                    x = x_old - self.tau * (self.lamb*Df + Dg)
                elif self.hparams.bregman_potential == 'burg':
                    den = x_old / (1 + self.tau * x_old * (self.lamb*Df + Dg) )
                    x = (den)*(0<=den)*(den<=self.hparams.C_max) + self.hparams.C_max*(den<0) + self.hparams.C_max*(den>self.hparams.C_max)
                else :
                    raise ValueError('Bregman potential not implemented')
                y = z = x
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0.01,1)
                # Calculate Objective
                f, F = self.calculate_F(x, x_old, img_tensor, prox_pnp=False)
                Psi = F # no Lyapunov function for GD
                diff_Psi = Psi-Psi_old

            elif self.hparams.PnP_algo == 'PGD':
                # Gradient step
                z, Df = self.data_fidelity_grad_step(x_old, img_tensor, self.lamb)
                # Denoising step
                Dz,g,Dg = self.denoise(z, self.sigma_denoiser, weight=self.alpha)
                x = Dz
                y = x
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                f, F = self.calculate_F(x, z, img_tensor, g)
                Psi = F
                diff_Psi = Psi-Psi_old

            elif self.hparams.PnP_algo == 'alphaPGD':
                # Pre-Gradient step
                q = (1-self.hparams.alpha_pd)*y_old + self.hparams.alpha_pd*x_old
                # Gradient step
                z, Df = self.data_fidelity_grad_step(q, img_tensor, self.lamb)
                # Denoising step
                Dz,g,Dg = self.denoise(z, self.sigma_denoiser, weight=self.alpha)
                x = (1 - self.hparams.alpha)*z + self.hparams.alpha*Dz
                y = (1-self.hparams.alpha_pd)*y + self.hparams.alpha_pd*x
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                f, F = self.calculate_F(x, z, img_tensor, g)
                Psi = F
                diff_Psi = Psi-Psi_old

            elif self.hparams.PnP_algo == 'PGD_RED':
                # Gradient of the regularization term
                _,g,Dg = self.denoise(x_old, self.sigma_denoiser)
                # Gradient step
                if self.hparams.bregman_potential == 'L2':
                    z = x_old - self.tau * Dg
                else:
                    raise ValueError('Bregman potential not implemented')
                # Proximal step
                x = self.data_fidelity_prox_step(z, img_tensor, self.lamb*self.tau)
                y = z # output image is the output of the denoising step
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                f, F = self.calculate_F(y, x_old, img_tensor, prox_pnp=False)
                Psi = F
                diff_Psi = Psi-Psi_old

            elif self.hparams.PnP_algo == 'DRS':
                # Denoising step
                Dx,g,Dg = self.denoise(x_old, self.sigma_denoiser)
                y = (1 - self.hparams.alpha)*x_old + self.hparams.alpha*Dx
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    y = torch.clamp(y,0,1)
                # Proximal step
                z = self.data_fidelity_prox_step(2*y-x_old, img_tensor, self.lamb)
                # Calculate Lyapunov
                Psi = self.calculate_lyapunov_DRS(y,z,x,g,img_tensor)
                # Calculate Objective
                f,F = self.calculate_F(y, x, img_tensor, g)
                # Final step
                x = x_old + (z-y)
                diff_Psi = Psi-Psi_old

            elif self.hparams.PnP_algo == 'DRSdiff':
                # Proximal step
                y = self.data_fidelity_prox_step(x_old, img_tensor, self.lamb)
                y2 = 2*y-x_old
                # Denoising step
                Dy2,g,Dg = self.denoise(y2, self.sigma_denoiser)
                z = (1 - self.hparams.alpha) * y2 + self.hparams.alpha * Dy2
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    z = torch.clamp(z, 0, 1)
                # Calculate Lyapunov
                Psi = self.calculate_lyapunov_DRS(y,z,x,g,img_tensor)
                # Calculate Objective
                F = self.calculate_F(y, x, img_tensor, g)
                # Final step
                x = x_old + (z-y)
            else :
                raise ValueError('algo not implemented')

            # Backtracking
            if i>self.hparams.lim_init+1 and use_backtracking :
                if self.hparams.noise_model == 'gaussian' :
                    diff_x = (torch.norm(x - x_old, p=2) ** 2)
                elif self.hparams.noise_model == 'poisson' : 
                    diff_x = burg_bregman_divergence(x, x_old).sum()
                diff_F = F_old - F
                if diff_F < (self.hparams.gamma_backtracking / self.tau) * diff_x :
                    self.tau = self.hparams.eta_backtracking * self.tau
                    self.backtracking_check = False
                    print('backtracking : tau =', self.tau, 'diff_F=', diff_F)
                else : 
                    self.backtracking_check = True

            if self.backtracking_check : # if the backtracking condition is satisfied
                # Logging
                if extract_results:
                    out_y = tensor2array(y.cpu())
                    out_z = tensor2array(z.cpu())
                    out_x = tensor2array(x.cpu())
                    current_y_psnr = psnr(clean_img, out_y)
                    current_z_psnr = psnr(clean_img, out_z)
                    current_x_psnr = psnr(clean_img, out_x)
                    if self.hparams.print_each_step:
                        print('iteration : ', i)
                        print('current y PSNR : ', current_y_psnr)
                        print('current z PSNR : ', current_z_psnr)
                        print('current x PSNR : ', current_x_psnr)
                    if not init :
                        y_list.append(out_y)
                        x_list.append(out_x)
                        z_list.append(out_z)
                        g_list.append(g.cpu().item())
                        Dg_list.append(torch.norm(Dg).cpu().item())
                        psnr_tab.append(current_x_psnr)
                        F_list.append(F)
                        Psi_list.append(Psi)
                        f_list.append(f)
                        try :
                            Df_list.append(torch.norm(Df).cpu().item())
                        except: 
                            pass

                # check decrease of data_fidelity 
                diff_f = f_old - f
                if init and i>self.hparams.lim_init : 
                    print('init phase is over')
                    init = False
                if i>self.hparams.lim_init+1 and early_stopping : 
                    if self.hparams.crit_conv == 'cost':
                        if abs(diff_Psi)/abs(Psi_old) < self.hparams.thres_conv:
                            print(f'Convergence reached at iteration {i}')
                            break
                    elif self.hparams.crit_conv == 'residual':
                        diff_x = torch.norm(x - x_old, p=2)
                        if diff_x/torch.norm(x) < self.hparams.thres_conv:
                            print(f'Convergence reached at iteration {i}')
                            break
                
                # next iteration
                i += 1
            else :
                x = x_old
                y = y_old
                Psi = Psi_old
                F = F_old

        output_img = tensor2array(y.cpu())
        output_psnr = psnr(clean_img, output_img)

        if extract_results:
            return output_img, tensor2array(x0.cpu()), output_psnr, i, x_list, np.array(z_list), np.array(y_list), np.array(Dg_list), np.array(psnr_tab), np.array(g_list), np.array(F_list), np.array(Psi_list), np.array(f_list), np.array(Df_list)
        else:
            return output_img, tensor2array(x0.cpu()), output_psnr, i

    def initialize_curves(self):

        self.conv = []
        self.conv_F = []
        self.conv_Dh = []
        self.PSNR = []
        self.g = []
        self.Dg = []
        self.Df = []
        self.F = []
        self.Psi = []
        self.f = []
        self.lip_algo = []
        self.lip_D = []
        self.lip_Dg = []

    def update_curves(self, x_list, psnr_tab, Dg_list, g_list, F_list, Psi_list, f_list, Df_list):

        self.F.append(F_list)
        #self.conv_F.append(np.array([(np.linalg.norm(F_list[k + 1] - F_list[k])) for k in range(len(x_list) - 1)]) / np.sum(np.abs(F_list[0])))
        self.Psi.append(Psi_list)
        self.f.append(f_list)
        self.g.append(g_list)
        self.Dg.append(Dg_list)
        self.Df.append(Df_list)
        self.PSNR.append(psnr_tab)
        self.conv.append(np.array([(np.linalg.norm(x_list[k + 1] - x_list[k]) ** 2) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))
        self.lip_algo.append(np.sqrt(np.array([np.sum(np.abs(x_list[k + 1] - x_list[k]) ** 2) for k in range(1, len(x_list) - 1)]) / np.array([np.sum(np.abs(x_list[k] - x_list[k - 1]) ** 2) for k in range(1, len(x_list[:-1]))])))
        self.conv_Dh.append(np.array([(burg_bregman_divergence_np(x_list[k + 1],x_list[k])) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))

    def save_curves(self, save_path):

        import matplotlib
        matplotlib.rcParams.update({'font.size': 17})
        matplotlib.rcParams['lines.linewidth'] = 2
        matplotlib.style.use('seaborn-darkgrid')
        use_tex = matplotlib.checkdep_usetex(True)
        if use_tex:
            plt.rcParams['text.usetex'] = True

        marker_dict = {}
        marker_dict['GD'] = '>'
        marker_dict['PGD'] = 'v'
        marker_dict['DRS'] = '^'
        marker_dict['alphaPGD'] = '*'
        marker_dict['PGD_RED'] = 'o'

        color_dict = {}
        color_dict['GD'] = 'red'
        color_dict['PGD'] = 'orange'
        color_dict['DRS'] = 'green'
        color_dict['alphaPGD'] = 'blue'
        color_dict['PGD_RED'] = 'purple'

        label_dict = {}
        label_dict['PGD'] = 'Prox-PnP-PGD'
        label_dict['DRS'] = 'Prox-PnP-DRS'
        label_dict['alphaPGD'] = r'Prox-PnP-$\alpha$PGD'
        label_dict['PGD_RED'] = 'GS-RED-PGD'

        plt.figure(0)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.g[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'g.png'),bbox_inches="tight")

        plt.figure(1)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.PSNR[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'PSNR.png'),bbox_inches="tight")

        plt.figure(2)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.Psi[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'Liapunov.png'), bbox_inches="tight")

        plt.figure(22)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.F[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'F.png'), bbox_inches="tight")



        plt.figure(23)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.f[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'f.png'), bbox_inches="tight")

        plt.figure(4)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            Ds_norm = [np.linalg.norm(np.array(self.Dg[i][j])) for j in range(len(self.Dg[i]))]
            plt.plot(Ds_norm, label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'Dg.png'), bbox_inches="tight")

        plt.figure(4)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            Ds_norm = [np.linalg.norm(np.array(self.Df[i][j])) for j in range(len(self.Df[i]))]
            plt.plot(Ds_norm, label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'Df.png'), bbox_inches="tight")

        # conv_DPIR = np.load('conv_DPIR2.npy')
        conv_rate = [self.conv[i][0]*np.array([(1/k) for k in range(1,len(self.conv[i]))]) for i in range(len(self.conv))]
        plt.figure(5)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.conv[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        # plt.plot(conv_DPIR[:self.hparams.maxitr], marker=marker_list[-1], markevery=10, label='DPIR')
        # plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.legend()
        plt.semilogy()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log.png'), bbox_inches="tight")

        self.conv2 = [[np.min(self.conv[i][:k]) for k in range(1, len(self.conv[i]))] for i in range(len(self.conv))]
        plt.figure(6)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.conv2[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        # plt.plot(conv_DPIR[:self.hparams.maxitr], marker=marker_list[-1], markevery=10, label='DPIR')
        # plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.semilogy()
        plt.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log2.png'), bbox_inches="tight")

        self.conv_sum = [[np.sum(self.conv[i][:k]) for k in range(1, len(self.conv[i]))] for i in range(len(self.conv))]

        plt.figure(7)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.conv_sum[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(os.path.join(save_path, 'conv_log_sum.png'), bbox_inches="tight")

        conv_rate = [self.conv[i][0]*np.array([(1/k) for k in range(1,len(self.conv[i]))]) for i in range(len(self.conv))]
        plt.figure(55)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.conv_Dh[i], label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        # plt.plot(conv_DPIR[:self.hparams.maxitr], marker=marker_list[-1], markevery=10, label='DPIR')
        # plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.legend()
        plt.semilogy()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log_Dh.png'), bbox_inches="tight")


        plt.figure(8)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i,algo in enumerate(self.hparams.PnP_algos):
            plt.plot(self.lip_algo[i],  label = label_dict[algo], color = color_dict[algo], marker = marker_dict[algo])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'lip_algo.png'))



    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_path', type=str, default='../datasets')
        parser.add_argument('--pretrained_checkpoint', type=str, default='../GS_denoising/ckpts/Prox_DRUNet.ckpt')
        parser.add_argument('--use_GSDRUNET_ckpt', dest='use_GSDRUNET_ckpt', action='store_true')
        parser.set_defaults(use_GSDRUNET_ckpt=False)
        parser.add_argument('--PnP_algos', type=str,  nargs='+', required=True)
        parser.add_argument('--noise_model', type=str,  default='gaussian')
        parser.add_argument('--alpha_poisson', type=float)
        parser.add_argument('--dataset_name', type=str, default='set3c')
        parser.add_argument('--sigma_k_denoiser', type=float)
        parser.add_argument('--noise_level_img', type=float)
        parser.add_argument('--maxitr', type=int, default=1000)
        parser.add_argument('--alpha', type=float, default = 1.)
        parser.add_argument('--stepsize', type=float, default=1.)
        parser.add_argument('--lamb', type=float)
        parser.add_argument('--sigma_denoiser', type=float)
        parser.add_argument('--n_images', type=int, default=68)
        parser.add_argument('--crit_conv', type=str, default='cost')
        parser.add_argument('--thres_conv', type=float, default=1e-7)
        parser.add_argument('--use_backtracking', dest='use_backtracking', action='store_true')
        parser.set_defaults(use_backtracking=False)
        parser.add_argument('--eta_backtracking', type=float, default=0.9)
        parser.add_argument('--gamma_backtracking', type=float, default=0.1)
        parser.add_argument('--inpainting_init', dest='inpainting_init', action='store_true')
        parser.set_defaults(inpainting_init=False)
        parser.add_argument('--precision', type=str, default='simple')
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--patch_size', type=int, default=256)
        parser.add_argument('--extract_curves', dest='extract_curves', action='store_true')
        parser.set_defaults(extract_curves=False)
        parser.add_argument('--extract_images', dest='extract_images', action='store_true')
        parser.set_defaults(extract_images=False)
        parser.add_argument('--print_each_step', dest='print_each_step', action='store_true')
        parser.set_defaults(print_each_step=False)
        parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        parser.set_defaults(grad_matching=True)
        parser.add_argument('--no_data_term', dest='no_data_term', action='store_true')
        parser.set_defaults(no_data_term=False)
        parser.add_argument('--use_hard_constraint', dest='use_hard_constraint', action='store_true')
        parser.set_defaults(use_hard_constraint=False)
        parser.add_argument('--alpha_pd', type=float)
        parser.add_argument('--early_stopping_patience', type=int)
        parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')
        parser.set_defaults(use_wandb=False)
        parser.add_argument('--use_Wiener_init', dest='use_Wiener_init', action='store_true')
        parser.set_defaults(use_Wiener_init=False)
        parser.add_argument('--no_linear_init', dest='use_linear_init', action='store_false')
        parser.set_defaults(use_linear_init=True)
        parser.add_argument('--bregman_potential', type=str, default='L2')
        parser.add_argument('--C_max', type=float, default=1.)
        parser.add_argument('--grayscale', dest='grayscale', action='store_true')
        parser.set_defaults(grayscale=False)
        parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false')
        parser.set_defaults(early_stopping=True)
        parser.add_argument('--weight_Dg', type=float, default=1.)
        parser.add_argument('--sigma_step', dest='sigma_step', action='store_true')
        parser.set_defaults(sigma_step=False)
        parser.add_argument('--bregman_div_g', type=str, default='L2')
        parser.add_argument('--no_inv_gamma', dest='inv_gamma', action='store_true')
        parser.set_defaults(inv_gamma=False)
        parser.add_argument('--lim_init', type=int, default=0)
        parser.add_argument('--lamb_pre_init', type=float, default=1.)
        parser.add_argument('--sigma_denoiser_pre_init', type=float, default=50.)
        parser.add_argument('--stepsize_pre_init', type=float, default=1.)
        parser.add_argument('--alpha_pre_init', type=float, default=1.)
        parser.add_argument('--no_default_params', dest='default_params', action='store_false')
        parser.set_defaults(default_params=True)
        return parser
