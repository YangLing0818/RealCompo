from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.models import ControlNetModel
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput 
from diffusers.utils import is_compiled_module
from diffusers.pipelines.controlnet import MultiControlNetModel
from keypoint_model.attention import aggregate_attention_each_step
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import PIL 




class StableDiffusionXLControlNet(StableDiffusionXLControlNetPipeline):

    @torch.no_grad()
    def get_noise(self, 
                  latents, 
                  do_classifier_free_guidance, 
                  t, 
                  guess_mode, 
                  prompt_embeds, 
                  controlnet_keep, 
                  controlnet_conditioning_scale, 
                  i, 
                  add_text_embeds, 
                  add_time_ids, 
                  image, 
                  cross_attention_kwargs, 
                  guidance_scale):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latents
            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds

        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        if guess_mode and do_classifier_free_guidance:
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        # predict the noise residual
        noise_pred_ctrl = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        noise_pred_sdxl = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred_ctrl.chunk(2)
            noise_pred_ctrl = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred_sdxl_uncond, noise_pred_sdxl_text = noise_pred_sdxl.chunk(2)
            noise_pred_sdxl = noise_pred_sdxl_uncond + guidance_scale * (noise_pred_sdxl_text - noise_pred_sdxl_uncond)
        return noise_pred_sdxl, noise_pred_ctrl
    
    # @torch.no_grad()
    def get_attn(self, 
                  latents, 
                  do_classifier_free_guidance, 
                  t, 
                  guess_mode, 
                  prompt_embeds, 
                  controlnet_keep, 
                  controlnet_conditioning_scale, 
                  i, 
                  add_text_embeds, 
                  add_time_ids, 
                  image, 
                  cross_attention_kwargs, 
                  controller):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latents
            control_model_input = self.scheduler.scale_model_input(control_model_input, t)
            controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
        else:
            control_model_input = latent_model_input
            controlnet_prompt_embeds = prompt_embeds

        if isinstance(controlnet_keep[i], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
        else:
            cond_scale = controlnet_conditioning_scale * controlnet_keep[i]

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        if guess_mode and do_classifier_free_guidance:
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        # predict the noise residual
        noise_pred_ctrl = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        attn_ctrl = aggregate_attention_each_step(controller, res = 32, from_where = ("up",), is_cross = True, select = 0) # torch.Size([32, 32, 77])
        noise_pred_sdxl = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        attn_sdxl = aggregate_attention_each_step(controller, res = 32, from_where = ("up",), is_cross = True, select = 0) # torch.Size([32, 32, 77])
        return attn_sdxl, attn_ctrl

    def compose_noise(self, noise_sdxl, noise_ctrl, influence_sdxl, influence_ctrl):
        concat_map = torch.cat([influence_sdxl, influence_ctrl], dim=1) 
        softmax_map = torch.nn.functional.softmax(input=concat_map, dim=1) 
        confidence_map_sdxl = softmax_map[:,0,:,:].unsqueeze(1)  
        confidence_map_ctrl = softmax_map[:,1,:,:].unsqueeze(1)
        noise = confidence_map_sdxl * noise_sdxl + confidence_map_ctrl * noise_ctrl
        noise = noise.to(noise_ctrl.dtype)
        return noise

    def get_loss(self, attention_sdxl, attention_ctrl, bboxs, token_location, device):
        loss = torch.tensor(0.0).to(device) 
        sum_box = torch.tensor(0.0)
        sum_all = torch.tensor(0.0)
        for i, (box, location) in enumerate(zip(bboxs, token_location)):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(64*x1), int(64*y1), int(64*x2), int(64*y2)
            sum_box = torch.sum(attention_sdxl[y1:y2, x1:x2, location])
            sum_all = torch.sum(attention_sdxl[:, :, location])
            loss += 1 - sum_box / sum_all
        for i, (box, location) in enumerate(zip(bboxs, token_location)):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(64*x1), int(64*y1), int(64*x2), int(64*y2)
            sum_box = torch.sum(attention_ctrl[y1:y2, x1:x2, location])
            sum_all = torch.sum(attention_ctrl[:, :, location])
            loss += 1 - sum_box / sum_all
        return loss

    @torch.no_grad()
    def __call__(
        self,
        controller,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        device: Optional[Union[str, torch.device]] = None,
        bboxs: Optional[List[List[int]]] = None,
        scale_factor: int = 1000000,
        scale_range: Tuple[float, float] = (1.0, 0.5),
        token_location: Optional[List[int]] = None,
    ):
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if len(keeps) == 1 else keeps)

        original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        # 7.2 Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 7.3 Prepare influence of sdxl and controlnet
        influence_sdxl = torch.randn((1, 1, 128, 128), device=device)
        influence_ctrl = torch.randn((1, 1, 128, 128), device=device)
        scale_range = np.linspace(scale_range[0], scale_range[1], len(timesteps))

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred_sdxl, noise_pred_ctrl = self.get_noise(
                    latents,
                    do_classifier_free_guidance,
                    t,
                    guess_mode,
                    prompt_embeds,
                    controlnet_keep,
                    controlnet_conditioning_scale,
                    i,
                    add_text_embeds,
                    add_time_ids,
                    image,
                    cross_attention_kwargs,
                    guidance_scale,
                )
                with torch.enable_grad():
                    influence_sdxl = influence_sdxl.clone().detach().requires_grad_(True)
                    influence_ctrl = influence_ctrl.clone().detach().requires_grad_(True)
                    noise_pred = self.compose_noise(noise_pred_sdxl, noise_pred_ctrl, influence_sdxl, influence_ctrl)
                    latents_mid = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    attn_sdxl, attn_ctrl = self.get_attn(
                        latents_mid,
                        do_classifier_free_guidance,
                        t,
                        guess_mode,
                        prompt_embeds,
                        controlnet_keep,
                        controlnet_conditioning_scale,
                        i,
                        add_text_embeds,
                        add_time_ids,
                        image,
                        cross_attention_kwargs,
                        controller,
                    )
                    loss = self.get_loss(attn_sdxl, attn_ctrl, bboxs, token_location, device)
                    # print(torch.autograd.grad(loss.requires_grad_(True), [attn_ctrl.requires_grad_(True)], retain_graph=True)[0])
                    grad_cond_influence_sdxl = torch.autograd.grad(loss.requires_grad_(True), [influence_sdxl], retain_graph=True)[0]
                    grad_cond_influence_ctrl = torch.autograd.grad(loss.requires_grad_(True), [influence_ctrl], retain_graph=True)[0]
                    step_size = scale_factor * np.sqrt(scale_range[i])
                    influence_sdxl = influence_sdxl - step_size * grad_cond_influence_sdxl
                    influence_ctrl = influence_ctrl - step_size * grad_cond_influence_ctrl

                noise_pred_sdxl, noise_pred_ctrl = self.get_noise(
                    latents,
                    do_classifier_free_guidance,
                    t,
                    guess_mode,
                    prompt_embeds,
                    controlnet_keep,
                    controlnet_conditioning_scale,
                    i,
                    add_text_embeds,
                    add_time_ids,
                    image,
                    cross_attention_kwargs,
                    guidance_scale,
                )
                noise_pred = self.compose_noise(noise_pred_sdxl, noise_pred_ctrl, influence_sdxl, influence_ctrl)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)