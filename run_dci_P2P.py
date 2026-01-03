from typing import Optional, Union, List
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline
import numpy as np

from P2P import ptp_utils
from PIL import Image
import os
from P2P.scheduler_dev import DDIMSchedulerDev
import argparse

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from utils.control_utils import load_512, make_controller
from P2P.dcInv import DualConditionalInversion
import json
import traceback
import gc
from transformers import CLIPProcessor, CLIPModel
@torch.no_grad()
def editing_p2p(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        return_type='image',
        inference_stage=True,
        x_stars=None,
        **kwargs,
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    start_time = num_inference_steps
    model.scheduler.set_timesteps(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], total=num_inference_steps)):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


@torch.no_grad()
def P2P_inversion_and_edit(
        image_path,
        prompt_src,
        prompt_tar,
        output_dir='output',
        guidance_scale=7.5,
        npi_interp=0,
        cross_replace_steps=0.8,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        offsets=(0, 0, 0, 0),
        is_replace_controller=False,
        use_inversion_guidance=True,
        K_round=25,
        num_of_ddim_steps=50,
        learning_rate=0.001,
        delta_threshold=5e-6,
        enable_threshold=True,
        dps_rate=1.0,
        model_path="runwayml/stable-diffusion-v1-4",
        **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    scheduler = DDIMSchedulerDev(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                 set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(
        device)

    DCI_inversion = DualConditionalInversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                        learning_rate=learning_rate, delta_threshold=delta_threshold,
                                                        enable_threshold=enable_threshold,dps_rate=dps_rate)
    _, x_stars, uncond_embeddings = DCI_inversion.invert(
        image_path, prompt_src,prompt_tar, offsets=offsets, npi_interp=npi_interp, verbose=True)

    z_inverted_noise_code = x_stars[-1]
    ########## edit ##########
    prompts = [prompt_src, prompt_tar]
    cross_replace_steps = {'default_': cross_replace_steps, }
    if isinstance(blend_word, str):
        s1, s2 = blend_word.split(",")
        blend_word = (((s1,), (
            s2,)))  
    if isinstance(eq_params, str):
        s1, s2 = eq_params.split(",")
        eq_params = {"words": (s1,), "values": (float(s2),)}  

    controller = make_controller(ldm_stable, prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                blend_word, eq_params, num_ddim_steps=num_of_ddim_steps)
    images, _ = editing_p2p(ldm_stable, prompts, controller, latent=z_inverted_noise_code,
                            num_inference_steps=num_of_ddim_steps,
                            guidance_scale=guidance_scale,
                            uncond_embeddings=uncond_embeddings,
                            inversion_guidance=use_inversion_guidance, x_stars=x_stars, )

    filename = image_path.split('/')[-1].replace(".jpg",".png")
    Image.fromarray(np.concatenate(images, axis=1)).save(f"{output_dir}/P2P_with_blend_word{filename}")
    print('image saved:', f"{output_dir}/P2P_with_blend_word{filename}")



def parse_args():
    parser = argparse.ArgumentParser(description="Input your image and editing prompt.")
    parser.add_argument(
        "--input",
        type=str,
        default='',
        help="Image path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="a round cake with orange frosting on a wooden plate",
        help="Source prompt",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="a square cake with orange frosting on a wooden plate",
        help="Target prompt",
    )
    parser.add_argument(
        "--blended_word",
        type=str,
        default="dog cat",
        # default="round square", 
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--K_round",
        type=int,
        default=10,
        help="Optimization Round",
    )
    parser.add_argument(
        "--num_of_ddim_steps",
        type=int,
        default=50,
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--dps_rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=5e-6,
        help="Delta threshold",
    )
    parser.add_argument(
        "--enable_threshold",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--eq_params",
        type=float,
        default=2.,
        help="Eq parameter weight for P2P",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dci_test",
        help="Save editing results",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="./data/mapping_file.json",
        help="JSON file path for batch processing",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-4",
        help="Path to Stable Diffusion model (local path or HuggingFace model ID)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = {}
    params['guidance_scale'] = args.guidance_scale
    params['K_round'] = args.K_round
    params['dps_rate'] = args.dps_rate
    params['num_of_ddim_steps'] = args.num_of_ddim_steps
    params['learning_rate'] = args.learning_rate
    params['enable_threshold'] = args.enable_threshold
    params['delta_threshold'] = args.delta_threshold
    params['output_dir'] = args.output
    params['model_path'] = args.model_path
    print(args.K_round, args.dps_rate, args.learning_rate)
    print(args.output)
    
    # Single image processing mode
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Image not found: {args.input}")
            exit(1)
        if not args.source or not args.target:
            print("Error: --source and --target prompts are required for single image processing")
            exit(1)
        
        print(f"\nProcessing single image: {args.input}")
        print(f"Source prompt: [{args.source}]")
        print(f"Target prompt: [{args.target}]")
        
        # Process blend_word
        blend_word = None
        if args.blended_word:
            blend_words = [w for w in args.blended_word.split() if w]
            if len(blend_words) >= 2:
                blend_word = ((blend_words[0],), (blend_words[1],))
            print(f"Blend words: {blend_word or 'None'}")
        
        params['prompt_src'] = args.source
        params['prompt_tar'] = args.target
        params['image_path'] = args.input
        params['blend_word'] = blend_word
        
        try:
            P2P_inversion_and_edit(**params)
            print("Processing completed successfully!")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            traceback.print_exc()
    
    # Batch processing mode (from JSON file)
    else:
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found: {args.json}")
            exit(1)
        
        with open(args.json, "r") as f:
            editing_instruction = json.load(f)
        
        print(f"\nBatch processing mode: Processing {len(editing_instruction)} images from {args.json}")
        
        for idx, (key, item) in enumerate(editing_instruction.items()):
            try:
                z_inverted_noise_code = None
                x_stars = None
                uncond_embeddings = None
                
                print(f"\nProcessing {idx+1}/{len(editing_instruction)} - Key: {key}")
                prompt_src = item["original_prompt"].replace("[", "").replace("]", "").strip()
                prompt_tar = item["editing_prompt"].replace("[", "").replace("]", "").strip()
                print(f"Original: [{prompt_src}], Target: [{prompt_tar}]")
                
                image_path = os.path.normpath(os.path.join(
                    "./data/annotation_images",
                    item["image_path"]
                ))
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
                print(f"Loading: {image_path}")
                blend_word = [w for w in item["blended_word"].split() if w]  
                blend_word = ((blend_word[0],), (blend_word[1],)) if len(blend_word) == 2 else None
                print(f"Blend words: {blend_word or 'None'}")
                params['prompt_src'] = prompt_src
                params['prompt_tar'] = prompt_tar
                params['image_path'] = image_path
                params['blend_word'] = blend_word
                P2P_inversion_and_edit(**params)

            except Exception as e:
                print(f"ERROR in {key}: {str(e)}")
                traceback.print_exc()
                continue
        
        print(f"\nBatch processing completed! Processed {len(editing_instruction)} images.")

