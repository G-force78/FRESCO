#@title Use IP adapter to iterate over depth maps, can change model if desired eg: SG161222/Realistic_Vision_V4.0_noVAE

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import transformers
from transformers import automodel
import torch
from diffusers.utils import load_image
import os

device = ("cuda")

controlnet_model_path = "/content/checkpoints" #change to local path
#controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
#controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)


pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V4.0_noVAE", controlnet=controlnet, torch_dtype=torch.float16
)
pipeline.to("cuda")

cat_image = load_image("/content/eva.png")  # Load the cat image once outside the loop

# Directory containing controlnet maps
depth_map_dir = "/content/test"

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

generator = torch.Generator(device="cuda").manual_seed(15)

# Directory containing depth maps
#depth_map_dir = "/content/test"

# Iterate over each depth map in the directory
for filename in os.listdir(depth_map_dir):
    if filename.endswith((".jpg", ".png")):  # image type in depth map dir
        depth_map_path = os.path.join(depth_map_dir, filename)
        depth_map = load_image(depth_map_path)

        images = pipeline(
            prompt='best quality, high quality',
            image=depth_map,
            ip_adapter_image=cat_image,  # Use the loaded cat image for each iteration
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, 2D, anime, cartoon",
            num_inference_steps=20,
            generator=generator,
        ).images

        # Save each output image with a unique name
        output_filename = f"/content/evabox/{os.path.splitext(filename)[0]}_out.jpg"
        images[0].save(output_filename)


