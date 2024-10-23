import os
import math
from PIL import Image
import numpy as np
import torch
import safetensors.torch as sf

from enum import Enum
from torch.hub import download_url_to_file
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

class IcLightFC:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.setup_models()
        self.setup_schedulers()
        self.setup_pipelines()

    def setup_models(self):
        self.sd15_name = 'stablediffusionapi/realistic-vision-v51'
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sd15_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.sd15_name, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.sd15_name, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.sd15_name, subfolder="unet")
        self.rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

        # Modify UNet
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(8, self.unet.conv_in.out_channels, 
                                        self.unet.conv_in.kernel_size, 
                                        self.unet.conv_in.stride, 
                                        self.unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            new_conv_in.bias = self.unet.conv_in.bias
            self.unet.conv_in = new_conv_in

        self.unet_original_forward = self.unet.forward
        self.unet.forward = self.hooked_unet_forward

        # Load model weights
        model_path = './models/iclight_sd15_fc.safetensors'
        if not os.path.exists(model_path):
            download_url_to_file('https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', model_path)

        sd_offset = sf.load_file(model_path)
        sd_origin = self.unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)

        # Move to device and set precision
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=torch.float16)
        self.vae = self.vae.to(device=self.device, dtype=torch.bfloat16)
        self.unet = self.unet.to(device=self.device, dtype=torch.float16)
        self.rmbg = self.rmbg.to(device=self.device, dtype=torch.float32)

        # Set attention processors
        self.unet.set_attn_processor(AttnProcessor2_0())
        self.vae.set_attn_processor(AttnProcessor2_0())

    def setup_schedulers(self):
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1
        )

        self.euler_a_scheduler = EulerAncestralDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1
        )

    def setup_pipelines(self):
        self.t2i_pipe = StableDiffusionPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

        self.i2i_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.dpmpp_2m_sde_karras_scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None
        )

    def hooked_unet_forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1)
        kwargs['cross_attention_kwargs'] = {}
        return self.unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

    # Add all the helper methods from gradio_demo.py
    @torch.inference_mode()
    def encode_prompt_inner(self, txt: str):
        max_length = self.tokenizer.model_max_length
        chunk_length = self.tokenizer.model_max_length - 2
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = self.tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = self.text_encoder(token_ids).last_hidden_state

        return conds

    @torch.inference_mode()
    def encode_prompt_pair(self, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(positive_prompt)
        uc = self.encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc

    @torch.inference_mode()
    def pytorch2numpy(self, imgs, quant=True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)

            if quant:
                y = y * 127.5 + 127.5
                y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

            results.append(y)
        return results

    @staticmethod
    def numpy2pytorch(imgs):
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
        h = h.movedim(-1, 1)
        return h

    @staticmethod
    def resize_and_center_crop(image, target_width, target_height):
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(target_width / original_width, target_height / original_height)
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    @staticmethod
    def resize_without_crop(image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)


    @torch.inference_mode()
    def run_rmbg(self, img, sigma=0.0):
        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = self.resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
        feed = self.numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)
        alpha = self.rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha
    
    @torch.inference_mode()
    def process(self, input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
        bg_source = BGSource(bg_source)
        input_bg = None

        if bg_source == BGSource.NONE:
            pass
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise 'Wrong initial latent!'

        rng = torch.Generator(device=self.device).manual_seed(int(seed))

        fg = self.resize_and_center_crop(input_fg, image_width, image_height)

        concat_conds = self.numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor

        conds, unconds = self.encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

        if input_bg is None:
            latents = self.t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type='latent',
                guidance_scale=cfg,
                cross_attention_kwargs={'concat_conds': concat_conds},
            ).images.to(self.vae.dtype) / self.vae.config.scaling_factor
        else:
            bg = self.resize_and_center_crop(input_bg, image_width, image_height)
            bg_latent = self.numpy2pytorch([bg]).to(device=self.vae.device, dtype=self.vae.dtype)
            bg_latent = self.vae.encode(bg_latent).latent_dist.mode() * self.vae.config.scaling_factor
            latents = self.i2i_pipe(
                image=bg_latent,
                strength=lowres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / lowres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type='latent',
                guidance_scale=cfg,
                cross_attention_kwargs={'concat_conds': concat_conds},
            ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        pixels = self.vae.decode(latents).sample
        pixels = self.pytorch2numpy(pixels)
        pixels = [self.resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64))
        for p in pixels]

        pixels = self.numpy2pytorch(pixels).to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixels).latent_dist.mode() * self.vae.config.scaling_factor
        latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

        image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        fg = self.resize_and_center_crop(input_fg, image_width, image_height)
        concat_conds = self.numpy2pytorch([fg]).to(device=self.vae.device, dtype=self.vae.dtype)
        concat_conds = self.vae.encode(concat_conds).latent_dist.mode() * self.vae.config.scaling_factor

        latents = self.i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(self.vae.dtype) / self.vae.config.scaling_factor

        pixels = self.vae.decode(latents).sample

        return self.pytorch2numpy(pixels)

    @torch.inference_mode()
    def process_relight(self, input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
        input_fg, matting = self.run_rmbg(input_fg)
        results = self.process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
        return input_fg, results
