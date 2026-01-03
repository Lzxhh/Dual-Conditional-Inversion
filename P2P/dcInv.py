import time

# %%
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
from P2P import ptp_utils
from P2P import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import os
from P2P.scheduler_dev import DDIMSchedulerDev
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from datetime import datetime
from random import randrange

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class DualConditionalInversion:
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    def get_noise_pred_single_dci(self, latents, t, context,img_ref,dps_rate):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        difference = img_ref-noise_pred
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=noise_pred)[0]
        noise_pred -= norm_grad * dps_rate
        return noise_pred
    

    
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    @torch.no_grad()
    def init_img(self, input_img):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        image = Image.open(input_img).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        image_emb = clip_model.get_image_features(**inputs)
        return image_emb
        
    @torch.no_grad()
    def init_prompt(self, prompt_src: str,prompt_tar:str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt_src],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        text_input2 = self.model.tokenizer(
            [prompt_tar],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        target_cond_embeddings = self.model.text_encoder(text_input2.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings,target_cond_embeddings])
        self.prompt = prompt_src
    def auto_corr_loss(self, x, random_shift=True):
        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = randrange(1, noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    def kl_divergence(self, x):
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)
    @torch.no_grad()
    def DCI_loop(self, latent,img_path):
        orig_latent=latent
        uncond_embeddings, cond_embeddings,target_cond_embeddings = self.context.chunk(3)

        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            #xt-> xt-1
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent_ztm1 = latent.clone().detach()
            #xt-1-> xt
            latent = self.next_step(noise_pred, t, latent_ztm1)
            ################ DCI optimization steps #################
            optimal_latent = latent.clone().detach()
            optimal_latent.requires_grad = True
            optimizer = torch.optim.AdamW([optimal_latent], lr=self.lr)
            for rid in range(self.opt_round):
                with torch.enable_grad():
                    optimizer.zero_grad()
                    noise_pred = self.get_noise_pred_single_dci(optimal_latent, t, cond_embeddings,img_ref=orig_latent,dps_rate=self.dps_rate)
                    pred_latent = self.next_step(noise_pred, t, latent_ztm1)
                    loss = F.mse_loss(optimal_latent, pred_latent)
                    loss.backward()
                    optimizer.step()
                    if self.enable_threshold and loss < self.threshold:
                        break

            ############## End DCI optimization ###################

            latent = optimal_latent.clone().detach()  
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def DCI_inversion(self, image,image_path):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)

        DCInv_latents = self.DCI_loop(latent,image_path)

        return image_rec, DCInv_latents, latent

    def invert(self, image_path: str, prompt_src: str,prompt_tar:str, offsets=(0, 0, 0, 0), npi_interp=0.0, verbose=False):
        self.init_prompt(prompt_src,prompt_tar)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)

        image_rec, ddim_latents, image_rec_latent = self.DCI_inversion(image_gt,image_path)
        uncond_embeddings, cond_embeddings,target_cond_embeddings = self.context.chunk(3)
        if npi_interp > 0.0:
            cond_embeddings = ptp_utils.slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return (image_gt, image_rec, image_rec_latent), ddim_latents, uncond_embeddings

    def __init__(self, model, K_round=25, num_ddim_steps=50, learning_rate=0.001, delta_threshold=5e-6,
                 enable_threshold=True,dps_rate=1.0):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        self.opt_round = K_round
        self.num_ddim_steps = num_ddim_steps
        self.lr = learning_rate
        self.threshold = delta_threshold
        self.enable_threshold = enable_threshold
        self.dps_rate = dps_rate
