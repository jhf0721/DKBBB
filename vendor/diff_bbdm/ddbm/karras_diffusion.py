"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
from inspect import isfunction
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from piq import LPIPS
from PIL import Image
from tqdm.autonotebook import tqdm
from .nn import mean_flat, append_dims, append_zero

from functools import partial
import torch
from .utils import extract, default
def vp_logsnr(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return - th.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
    
def vp_logs(t, beta_d, beta_min):
    t = th.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

class KarrasDenoiser(nn.Module):
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0., # 0 for uncorrelated, sigma_data**2 / 2 for  C_skip=1/2 at sigma_max
        rho=7.0,
        image_size=64,
        weight_schedule="karras",
        pred_mode='both',
        loss_norm="lpips",
        num_timesteps=1000,
        objective = 'grad',
            loss_type='l1',

    ):
        super().__init__()
        self.loss_type=loss_type
        self.sigma_data = sigma_data
        
        self.sigma_max = sigma_max 
        self.sigma_min = sigma_min 

        self.beta_d = beta_d
        self.beta_min = beta_min


        self.sigma_data_end = self.sigma_data
        self.cov_xy = cov_xy
            
        self.c = 1

        self.weight_schedule = weight_schedule
        self.pred_mode = pred_mode
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = num_timesteps
        self.image_size = image_size
        self.objective = objective
        self.mt_type = "linear"
        self.max_var = 1.0
        self.skip_sample=True
        self.sample_type='linear'
        self.sample_step=3
        self.eta=1.0
        self.register_schedule()

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)
    def get_snr(self, sigmas):
        if self.pred_mode.startswith('vp'):
            return vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas


    def get_weightings(self, sigma):
        snrs = self.get_snr(sigma)
        
        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data**2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.pred_mode == 've':
                A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
                weightings = A / ((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )
            
            elif self.pred_mode == 'vp':
                
                logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

                A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
                weightings = A / (a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * self.c**2 * c_t )
                
            elif self.pred_mode == 'vp_simple' or  self.pred_mode == 've_simple':

                weightings = th.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = th.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = th.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings


    def get_bridge_scalings(self, sigma):
        if self.pred_mode == 've':
            A = sigma**4 / self.sigma_max**4 * self.sigma_data_end**2 + (1 - sigma**2 / self.sigma_max**2)**2 * self.sigma_data**2 + 2*sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * self.cov_xy + self.c **2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2)
            c_in = 1 / (A) ** 0.5
            c_skip = ((1 - sigma**2 / self.sigma_max**2) * self.sigma_data**2 + sigma**2 / self.sigma_max**2 * self.cov_xy)/ A
            c_out =((sigma/self.sigma_max)**4 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * sigma**2 * (1 - sigma**2/self.sigma_max**2) )**0.5 * c_in
            return c_skip, c_out, c_in
        
    
        elif self.pred_mode == 'vp':

            logsnr_t = vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
            b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -th.expm1(logsnr_T - logsnr_t) * (2*logs_t - logsnr_t).exp()

            A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2*a_t * b_t * self.cov_xy + self.c**2 * c_t
            
            
            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy)/ A
            c_out =(a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 *  self.c **2 * c_t )**0.5 * c_in
            return c_skip, c_out, c_in
            
    
        elif self.pred_mode == 've_simple' or self.pred_mode == 'vp_simple':
            
            c_in = th.ones_like(sigma)
            c_out = th.ones_like(sigma) 
            c_skip = th.zeros_like(sigma)
            return c_skip, c_out, c_in
        

    def training_bridge_losses0(self, model, x_start, sigmas, model_kwargs=None, noise=None, vae=None):
        
        assert model_kwargs is not None
        xT = model_kwargs['xT']
        if noise is None:
            noise = th.randn_like(x_start) 
        sigmas =th.minimum(sigmas, th.ones_like(sigmas)* self.sigma_max)
        terms = {}

        dims = x_start.ndim
        def bridge_sample(x0, xT, t):
            t = append_dims(t, dims)
            # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
            if self.pred_mode.startswith('ve'):
                std_t = t* th.sqrt(1 - t**2 / self.sigma_max**2)
                mu_t= t**2 / self.sigma_max**2 * xT + (1 - t**2 / self.sigma_max**2) * x0
                samples = (mu_t +  std_t * noise )
            elif self.pred_mode.startswith('vp'):
                logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
                logs_t = vp_logs(t, self.beta_d, self.beta_min)
                logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                std_t = (-th.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
                
                samples= a_t * xT + b_t * x0 + std_t * noise
                
                
            return samples

        import matplotlib.pyplot as plt
        # img = Image.fromarray(x_start[0][0].detach().cpu().numpy())
        # plt.imshow(img.copy())
        # plt.axis('off')
        # plt.show()
        # img0 = Image.fromarray(xT[0][0].detach().cpu().numpy())
        # plt.imshow(img0.copy())
        # plt.axis('off')
        # plt.show()
        x_t = bridge_sample(x_start, xT, sigmas)
        # img1=Image.fromarray(x_t[0][0].detach().cpu().numpy())
        # plt.imshow(img1.copy())
        # plt.axis('off')
        # plt.show()
        model_output, denoised = self.denoise(model, x_t, sigmas,  **model_kwargs)
        # img2=Image.fromarray(denoised[0][0].detach().cpu().numpy())
        # plt.imshow(img2.copy())
        # plt.axis('off')
        # plt.show()
        weights = self.get_weightings(sigmas)
        
        weights =  append_dims((weights), dims)
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def training_bridge_losses(self, model, x_start, sigmas, model_kwargs=None, noise=None, vae=None):

        assert model_kwargs is not None
        xT = model_kwargs['xT']
        terms = {}
        b, c,h, w,  device, img_size, = *x_start.shape, x_start.device, self.image_size
        assert h == img_size and w == img_size , f'height and width of image must be {img_size}'
        # assert h == img_size and w == img_size and d==img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if noise is None:
            noise = th.randn_like(x_start)
        x_t, objective = self.q_sample(x_start, xT , t, noise)
        context = None
        objective_recon = model(x_t, t,**model_kwargs)


        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        terms["loss"]=recloss
        terms["xs_mse"] = mean_flat((objective - objective_recon) ** 2)
        return terms
    def q_sample(self, x0, y, t, noise=None):
        if noise is None:
            noise = th.randn_like(x0)
        device = x0.device
        self.m_t = self.m_t.to(device)
        b, *_ = t.shape
        out = self.m_t.gather(-1, t)
        m_t = out.reshape(b, *((1,) * (len(x0.shape) - 1)))
        self.variance_t = self.variance_t.to(device)
        out1= self.variance_t.gather(-1, t)
        var_t = out1.reshape(b, *((1,) * (len(x0.shape) - 1)))
        # var_t = self.extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def denoise(self, model, x_t, sigmas ,**model_kwargs):

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas)
        ]
               
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised
    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon
    @torch.no_grad()
    def p_sample(self, model, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = model(x_t, t)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i + 1], device=x_t.device, dtype=torch.long)

            objective_recon = model(x_t, t)



            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            device = x_t.device
            self.m_t = self.m_t.to(device)
            self.variance_t=self.variance_t.to(device)
            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta
            # sigma_t = torch.sqrt(var_t) * self.eta
            # sigma2_t=0

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon
            # return x_tminus_mean , x0_recon

    @torch.no_grad()
    def p_sample_loop(self,model, y, context=None, clip_denoised=True, sample_mid_step=False):

        context = None


        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(model, x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(model, x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, model, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(model, y, context, clip_denoised, sample_mid_step)
def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    guidance=1,
):
    # assert sampler in ["heun", ], 'only heun sampler is supported currently'
    #
    # sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)
    #
    #
    # sample_fn = {
    #     "heun": partial(sample_heun, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min),
    # }[sampler]
    #
    # sampler_args = dict(
    #         pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
    #     )
    #
    #
    #
    # def denoiser(x_t, sigma, x_T=None):
    #     _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
    #
    #     if clip_denoised:
    #         denoised = denoised.clamp(-1, 1)
    #
    #     return denoised
    #
    # x_0, path, nfe = sample_fn(
    #     denoiser,
    #     x_T,
    #     sigmas,
    #     progress=progress,
    #     callback=callback,
    #     guidance=guidance,
    #     **sampler_args,
    # )
    # print('nfe:', nfe)
    #
    # return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], nfe
    x_0=diffusion.sample(model,x_T, clip_denoised=False)
    return x_0


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_bridge_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, eps=1e-4, device="cpu"):
    
    sigma_t_crit = sigma_max / np.sqrt(2)
    min_start_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_t_crit ** (1 / rho)
    sigmas_second_half = (max_inv_rho + th.linspace(0, 1, n//2 ) * (min_start_inv_rho - max_inv_rho)) ** rho
    sigmas_first_half = sigma_max - ((sigma_max - sigma_t_crit)  ** (1 / rho) + th.linspace(0, 1, n - n//2 +1 ) * (eps  ** (1 / rho)  - (sigma_max - sigma_t_crit)  ** (1 / rho))) ** rho
    sigmas = th.cat([sigmas_first_half.flip(0)[:-1], sigmas_second_half])
    sigmas_bridge = sigmas**2 *(1-sigmas**2/sigma_max**2)
    return append_zero(sigmas).to(device)#, append_zero(sigmas_bridge).to(device)


def to_d(x, sigma, denoised, x_T, sigma_max,   w=1, stochastic=False):
    """Converts a denoiser output to a Karras ODE derivative."""
    grad_pxtlx0 = (denoised - x) / append_dims(sigma**2, x.ndim)
    grad_pxTlxt = (x_T - x) / (append_dims(th.ones_like(sigma)*sigma_max**2, x.ndim) - append_dims(sigma**2, x.ndim))
    gt2 = 2*sigma
    d = - (0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d


def get_d_vp(x, denoised, x_T, std_t,logsnr_t, logsnr_T, logs_t, logs_T, s_t_deriv, sigma_t, sigma_t_deriv, w, stochastic=False):
    
    a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
    b_t = -th.expm1(logsnr_T - logsnr_t) * logs_t.exp()
    
    mu_t = a_t * x_T + b_t * denoised 
    
    grad_logq = - (x - mu_t)/std_t**2 / (-th.expm1(logsnr_T - logsnr_t))
    # grad_logpxtlx0 = - (x - logs_t.exp()*denoised)/std_t**2 
    grad_logpxTlxt = -(x - th.exp(logs_t-logs_T)*x_T) /std_t**2  / th.expm1(logsnr_t - logsnr_T)

    f = s_t_deriv * (-logs_t).exp() * x
    gt2 = 2 * (logs_t).exp()**2 * sigma_t * sigma_t_deriv 
    # breakpoint()

    d = f -  gt2 * ((0.5 if not stochastic else 1)* grad_logq - w * grad_logpxTlxt)
    # d = f - (0.5 if not stochastic else 1) * gt2 * (grad_logpxtlx0 - w * grad_logpxTlxt* (0 if stochastic else 1))
    if stochastic:
        return d, gt2
    else:
        return d

@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    pred_mode='both',
    progress=False,
    callback=None,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    churn_step_ratio=0.,
    guidance=1,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    path = [x]
    
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    nfe = 0
    assert churn_step_ratio < 1

    if pred_mode.startswith('vp'):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (beta_min + beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)

        logs = lambda t: -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min
        
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        logsnr = lambda t :  - 2 * th.log(vp_snr_sqrt_reciprocal(t))

        logsnr_T = logsnr(th.as_tensor(sigma_max))
        logs_T = logs(th.as_tensor(sigma_max))
    
    for j, i in enumerate(indices):
        
        if churn_step_ratio > 0:
            # 1 step euler
            sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
            
            denoised = denoiser(x, sigmas[i] * s_in, x_T)
            if pred_mode == 've':
                d_1, gt2 = to_d(x, sigmas[i] , denoised, x_T, sigma_max,  w=guidance, stochastic=True)
            elif pred_mode.startswith('vp'):
                d_1, gt2 = get_d_vp(x, denoised, x_T, std(sigmas[i]),logsnr(sigmas[i]), logsnr_T, logs(sigmas[i] ), logs_T, s_deriv(sigmas[i] ), vp_snr_sqrt_reciprocal(sigmas[i] ), vp_snr_sqrt_reciprocal_deriv(sigmas[i] ), guidance, stochastic=True)
            
            dt = (sigma_hat - sigmas[i]) 
            x = x + d_1 * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
            
            nfe += 1
            
            path.append(x.detach().cpu())
        else:
            sigma_hat =  sigmas[i]
        
        # heun step
        denoised = denoiser(x, sigma_hat * s_in, x_T)
        if pred_mode == 've':
            # d =  (x - denoised ) / append_dims(sigma_hat, x.ndim)
            d = to_d(x, sigma_hat, denoised, x_T, sigma_max, w=guidance)
        elif pred_mode.startswith('vp'):
            d = get_d_vp(x, denoised, x_T, std(sigma_hat),logsnr(sigma_hat), logsnr_T, logs(sigma_hat), logs_T, s_deriv(sigma_hat), vp_snr_sqrt_reciprocal(sigma_hat), vp_snr_sqrt_reciprocal_deriv(sigma_hat), guidance)
            
        nfe += 1
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            
            x = x + d * dt 
            
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
            if pred_mode == 've':
                # d_2 =  (x_2 - denoised_2) / append_dims(sigmas[i + 1], x.ndim)
                d_2 = to_d(x_2,  sigmas[i + 1], denoised_2, x_T, sigma_max, w=guidance)
            elif pred_mode.startswith('vp'):
                d_2 = get_d_vp(x_2, denoised_2, x_T, std(sigmas[i + 1]),logsnr(sigmas[i + 1]), logsnr_T, logs(sigmas[i + 1]), logs_T, s_deriv(sigmas[i + 1]),
                                vp_snr_sqrt_reciprocal(sigmas[i + 1]), vp_snr_sqrt_reciprocal_deriv(sigmas[i + 1]), guidance)
            
            d_prime = (d + d_2) / 2

            # noise = th.zeros_like(x) if 'flow' in pred_mode or pred_mode == 'uncond' else generator.randn_like(x)
            x = x + d_prime * dt #+ noise * (sigmas[i + 1]**2 - sigma_hat**2).abs() ** 0.5
            nfe += 1
        # loss = (denoised.detach().cpu() - x0).pow(2).mean().item()
        # losses.append(loss)

        path.append(x.detach().cpu())
        
    return x, path, nfe

@th.no_grad()
def forward_sample(
    x0,
    y0,
    sigma_max,
    ):

    ts = th.linspace(0, sigma_max, 120)
    x = x0
    # for t, t_next in zip(ts[:-1], ts[1:]):
    #     grad_pxTlxt = (y0 - x) / (append_dims(th.ones_like(ts)*sigma_max**2, x.ndim) - append_dims(t**2, x.ndim))
    #     dt = (t_next - t) 
    #     gt2 = 2*t
    #     x = x + grad_pxTlxt * dt + th.randn_like(x) *((dt).abs() ** 0.5)*gt2.sqrt()
    path = [x]
    for t in ts:
        std_t = th.sqrt(t)* th.sqrt(1 - t / sigma_max)
        mu_t= t / sigma_max * y0 + (1 - t / sigma_max) * x0
        xt = (mu_t +  std_t * th.randn_like(x0) )
        path.append(xt)

    path.append(y0)

    return path


