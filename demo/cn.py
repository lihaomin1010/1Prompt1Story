from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, EulerDiscreteScheduler
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import torch
import numpy as np
import cv2

from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffusers.models.unets import UNet2DConditionModel

controlnet_conditioning_scale = 1.0
prompt = "a portrait  photo of professional middle-aged Asian female teacher, single subject, standing confidently in front of a plain background, soft natural lighting, photorealistic, 4k, high detail, neutral expression, clean background, DSLR style"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'


# controlnet = ControlNetModel.from_pretrained(
#     "xinsir/controlnet-openpose-sdxl-1.0",
#     torch_dtype=torch.float16
# )

# when test with other base model, you need to change the vae also.
#vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

model_path = "stabilityai/stable-diffusion-xl-base-1.0"
variant = "fp16"
torch_dtype = torch.float16

vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype, variant=variant, )
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=torch_dtype, variant=variant, )
tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", torch_dtype=torch_dtype,
                                            variant=variant, )
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype,
                                             variant=variant, )
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2",
                                                             torch_dtype=torch_dtype, variant=variant, )

unet_model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch_dtype, variant=variant,)
scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch_dtype, variant=variant,)

controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16,
    #variant=variant,
)

pipe =  StableDiffusionXLControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet_model,
        scheduler=scheduler,
        controlnet=controlnet,
    )

processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


controlnet_img = cv2.imread("resource/controlnet/openpose0.png")
controlnet_img = processor(controlnet_img, hand_and_face=False, output_type='cv2')


# need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
height, width, _  = controlnet_img.shape
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = cv2.resize(controlnet_img, (new_width, new_height))
controlnet_img = Image.fromarray(controlnet_img)

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=controlnet_img,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=new_width,
    height=new_height,
    num_inference_steps=30,
    ).images

images[0].save(f"result.jpg")
