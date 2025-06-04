import os
import torch
import random
import diffusers
import torch.utils
import unet.utils as utils
from unet.unet_controller import UNetController
import argparse
from datetime import datetime


import numpy as np
from controlnet.utils import pre_process_images
diffusers.utils.logging.set_verbosity_error()

def load_unet_controller(pipe, device):
    unet_controller = UNetController()
    unet_controller.device = device
    unet_controller.tokenizer = pipe.tokenizer

    return unet_controller

def generate_images(unet_controller: UNetController, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed, verbose=True, control_image_list = None):
    generate = torch.Generator().manual_seed(seed)
    if unet_controller.Use_ipca is True:
        unet_controller.Store_qkv = True
        original_prompt_embeds_mode = unet_controller.Prompt_embeds_mode
        unet_controller.Prompt_embeds_mode = "original"
        #_ = pipe(id_prompt, generator=generate, unet_controller=unet_controller).images
        img_0 = pipe(id_prompt, image=control_image_list[0], generator=generate, unet_controller=unet_controller, num_inference_steps=50, controlnet_conditioning_scale=args.cn_scale).images[0]
        # img_0 = pipe(id_prompt, negative_prompt="blurry, lowres, cartoon, 3d render, bad anatomy, extra limbs",
        #              image=control_image_list[0], generator=generate, unet_controller=unet_controller,
        #              num_inference_steps=80).images[0]
        img_0.save(os.path.join(save_dir, 'id_prompt.jpg'))
        unet_controller.Prompt_embeds_mode = original_prompt_embeds_mode


    unet_controller.Store_qkv = False
    images, story_image = utils.movement_gen_story_slide_windows(
        id_prompt, frame_prompt_list, pipe, window_length, seed, unet_controller, save_dir, verbose=verbose, control_images=control_image_list[1:], cn_scale=args.cn_scale
    )

    return images, story_image


def main(device, model_path, save_dir, id_prompt, frame_prompt_list, precision, seed, window_length, control_model_path=None, control_image_list=None):
    pipe, _ = utils.load_pipe_from_path(model_path, device, torch.float16 if precision == "fp16" else torch.float32, precision, control_model_path)

    if control_model_path is not None:
        control_image_list = pre_process_images(control_model_path, control_image_list)

    unet_controller = load_unet_controller(pipe, device)          
    images, story_image = generate_images(unet_controller, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed, control_image_list=control_image_list)

    return images, story_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a specific device.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation (e.g., cuda:0, cpu)')
    parser.add_argument('--model_path', type=str, default='playgroundai/playground-v2.5-1024px-aesthetic', help='Path to the model')
    parser.add_argument('--control_model_path', type=str, default=None, help='Path to the controlnet model')
    parser.add_argument('--project_base_path', type=str, default='.', help='Path to save the generated images')
    #parser.add_argument('--id_prompt', type=str, default="A photorealistic image of an old teacher", help='Initial prompt for image generation')

    # parser.add_argument('--id_prompt', type=str, default="a portrait  photo of professional middle-aged Asian female teacher, single subject, standing confidently in front of a plain background, soft natural lighting, photorealistic, 4k, high detail, neutral expression, clean background, DSLR style",
    #                     help='Initial prompt for image generation')
    parser.add_argument('--id_prompt', type=str, default="A photo of a elderly gentleman",
                        help='Initial prompt for image generation')
    parser.add_argument('--frame_prompt_list', type=str, nargs='+', default=[
        "cross-legged on a mountain, surrounded by ancient scrolls",
        "sitting on a bench under cherry blossoms",
        "sitting in a sunny garden, playing with a puppy",
    ], help='List of frame prompts')
    parser.add_argument('--control_image_list', type=str, nargs='+', default=[
        "resource/controlnet/white.png",
        "resource/controlnet/cat.png",
        "resource/controlnet/cat.png",
        "resource/controlnet/cat.png",
    ], help='List of control images')
    parser.add_argument('--precision', type=str, choices=["fp16", "fp32"], default="fp16", help='Model precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--window_length', type=int, default=10, help='Window length for story generation')
    parser.add_argument('--save_padding', type=str, default='test', help='Padding for save directory')
    parser.add_argument('--random_seed', action='store_true', help='Use random seed')
    parser.add_argument('--json_path', type=str,)
    parser.add_argument('--cn_scale', type=float, default=1.0, help='control_net scale for generation')
    args = parser.parse_args()
    if args.random_seed:
        args.seed = random.randint(0, 1000000)

    current_time = datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(args.project_base_path, f'result/{current_time}/{current_time_}_{args.save_padding}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    if args.json_path is None:
        main(args.device, args.model_path, save_dir, args.id_prompt, args.frame_prompt_list, args.precision, args.seed, args.window_length, args.control_model_path, args.control_image_list)
    else:
        import json
        with open(args.json_path, "r") as file:
            data = json.load(file)

        combinations = data["combinations"]

        for combo in combinations:
            main(args.device, args.model_path, save_dir, combo['id_prompt'], combo['frame_prompt_list'], args.precision, args.seed, args.window_length)
