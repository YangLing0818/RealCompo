import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from utils.loss_function import get_loss
from utils.attentionmap import aggregate_attention_each_step


class PLMSSampler(object):
    def __init__(self, 
                 diffusion, 
                 layout_unet, 
                 text_unet, 
                 controller, 
                 scale_factor,
                 scale_range,
                 schedule="linear", 
                 alpha_generator_func=None, 
                 set_alpha_scale=None):
        super().__init__()
        self.diffusion = diffusion
        self.layout_unet = layout_unet
        self.text_unet = text_unet
        self.controller = controller
        self.scale_factor = scale_factor
        self.scale_range = scale_range
        self.device = diffusion.betas.device
        self.ddpm_num_timesteps = diffusion.num_timesteps #1000
        self.schedule = schedule
        self.alpha_generator_func = alpha_generator_func
        self.set_alpha_scale = set_alpha_scale

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            attr = attr.to(self.device)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=False):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.diffusion.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('betas', to_torch(self.diffusion.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.diffusion.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
    

    def get_attention_maps(self, t, latent, input):
        input2 = input.copy()
        input2["x"] = latent
        input2["timesteps"] = t 
        e_t_layout, attn_layout = self.layout_unet(input2) 
        e_t_text = self.text_unet(input2["x"], input2["timesteps"], input2["context"]).sample
        attn_text = aggregate_attention_each_step(self.controller, res = 16, from_where = ("up", "down"), is_cross = True, select = 0) # torch.Size([16, 16, 77])
        del e_t_layout, e_t_text
        return attn_layout, attn_text
    
    @torch.no_grad()
    def get_epsilon(self, t, input):
        input2 = input.copy()
        input2["timesteps"] = t
        e_t_layout, attn_layout = self.layout_unet(input2)
        e_t_text = self.text_unet(input2["x"], input2["timesteps"], input2["context"]).sample
        del attn_layout
        return e_t_layout, e_t_text
    
    def compose_epsilon(self, e_layout, e_text, influence_layout, influence_text):
        concat_map = torch.cat([influence_layout, influence_text], dim=1) 
        softmax_map = torch.nn.functional.softmax(input=concat_map, dim=1) 
        confidence_map_layout = softmax_map[:,0,:,:].unsqueeze(1)  
        confidence_map_text = softmax_map[:,1,:,:].unsqueeze(1)
        e_t = confidence_map_layout * e_layout + confidence_map_text * e_text
        return e_t
    
    def get_latent(self, input, i, t, e_t, index):
        x = input["x"].clone() 
        b = x.shape[0]

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(x)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        input["timesteps"] = t 
        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
        return x_prev


    @torch.no_grad()
    def sample(self, S, shape, input, uc=None, guidance_scale=1, mask=None, x0=None):
        self.make_schedule(ddim_num_steps=S)
        return self.plms_sampling(shape, input, uc, guidance_scale, mask=mask, x0=x0)


    @torch.no_grad()
    def plms_sampling(self, shape, input, uc=None, guidance_scale=1, mask=None, x0=None):

        b = shape[0]
        img = input["x"]
        if img == None:     
            img = torch.randn(shape, device=self.device)
            input["x"] = img
        
        torch.manual_seed(31)
        influence_layout = torch.randn((1, 1, 64, 64), device=self.device)
        influence_text = torch.randn((1, 1, 64, 64), device=self.device)

        time_range = np.flip(self.ddim_timesteps) 
        total_steps = self.ddim_timesteps.shape[0] 
        scale_range = np.linspace(self.scale_range[0], self.scale_range[1], len(time_range))    
        old_eps = []

        if self.alpha_generator_func != None:
            alphas = self.alpha_generator_func(len(time_range))

        for i, step in enumerate(tqdm(time_range)):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=self.device, dtype=torch.long)
            ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)
            if self.alpha_generator_func != None:
                self.set_alpha_scale(self.layout_unet, alphas[i])
            with torch.enable_grad():
                influence_layout = influence_layout.clone().detach().requires_grad_(True)
                influence_text = influence_text.clone().detach().requires_grad_(True)
                e_t_layout, e_t_text = self.get_epsilon(ts, input)
                e_t = self.compose_epsilon(e_t_layout, e_t_text, influence_layout, influence_text)

                img = self.get_latent(input, i, ts, e_t, index)
                attn_layout, attn_text = self.get_attention_maps(ts, img, input)
                loss = get_loss(attn_layout, attn_text, input)

                grad_cond_influence_layout = torch.autograd.grad(loss.requires_grad_(True), [influence_layout], retain_graph=True)[0]
                grad_cond_influence_text = torch.autograd.grad(loss.requires_grad_(True), [influence_text], retain_graph=True)[0]

                step_size = self.scale_factor * np.sqrt(scale_range[i])

                # update
                influence_layout = influence_layout - step_size * grad_cond_influence_layout
                influence_text = influence_text - step_size * grad_cond_influence_text

            img, e_t= self.p_sample_plms(input=input,
                                        i=i, 
                                        t=ts, 
                                        influence_layout=influence_layout, 
                                        influence_text = influence_text, 
                                        index=index, 
                                        uc=uc, 
                                        guidance_scale=guidance_scale, 
                                        old_eps=old_eps, 
                                        t_next=ts_next)
            input["x"] = img
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
        return img


    @torch.no_grad()
    def p_sample_plms(self, input, i, t, influence_layout, influence_text, index, guidance_scale=1., uc=None, old_eps=None, t_next=None):
        x = input["x"].clone() 
        b = x.shape[0]

        def get_model_output(input,influence_layout, influence_text, i):
            e_t_layout, attn_layout = self.layout_unet(input) 
            e_t_text = self.text_unet(input["x"], input["timesteps"], input["context"]).sample
            del attn_layout
            if uc is not None and guidance_scale != 1:
                unconditional_input = dict(x=input["x"], timesteps=input["timesteps"], context=uc, inpainting_extra_input=input["inpainting_extra_input"], grounding_extra_input=input['grounding_extra_input'])
                e_t_text_uncond = self.text_unet(input["x"], input["timesteps"], uc ).sample
                e_t_layout_uncond, _ = self.layout_unet( unconditional_input ) 
                e_t_layout = e_t_layout_uncond + guidance_scale * (e_t_layout - e_t_layout_uncond)
                e_t_text = e_t_text_uncond + guidance_scale * (e_t_text - e_t_text_uncond)
            e_t = self.compose_epsilon(e_t_layout, e_t_text, influence_layout, influence_text)
            return e_t

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), self.ddim_alphas[index], device=self.device)
            a_prev = torch.full((b, 1, 1, 1), self.ddim_alphas_prev[index], device=self.device)
            sigma_t = torch.full((b, 1, 1, 1), self.ddim_sigmas[index], device=self.device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.ddim_sqrt_one_minus_alphas[index],device=self.device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(x)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        input["timesteps"] = t 
        e_t = get_model_output(input, influence_layout, influence_text, i)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            input["x"] = x_prev
            input["timesteps"] = t_next
            e_t_next = get_model_output(input, influence_layout, influence_text, i)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)
        
        return x_prev, e_t
