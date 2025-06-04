from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers.utils import load_image
from PIL import Image
import torch
from controlnet_aux import OpenposeDetector
import cv2
import numpy as np

def get_depth_map(image):
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def get_canny_image(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def pre_process_images(control_net, images_f):
    images = [load_image(file).resize((1024, 1024)) for file in images_f]
    p_images = []
    if "openpose" in control_net:
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        p_images = [openpose(image) for image in images]
    elif "canny" in control_net:
        p_images = [get_canny_image(image) for image in images]
    elif "depth" in control_net:
        p_images = [get_depth_map(image) for image in images]
    for i in range(len(p_images)):
        p_images[i].save(f"demo/{i}.png")
    return p_images


if __name__ == "__main__":
    init_image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/kandinsky/cat.png"
    ).resize((1024, 1024))

    control_image = get_depth_map(init_image)