#@title IPadapter FAceID plus WORKING DO NOT MODIFY THIS CELL! (first move ip_adapter into /content/)


import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
import os
from datetime import datetime
import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, StableDiffusionControlNetPipeline, ControlNetModel
#from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
#import ip_adapter
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from PIL import Image
from diffusers.utils import load_image
# Get the current date and time
now = datetime.now()
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

# Create the output directory if it doesn't exist
output_dir = "/content/output"
os.makedirs(output_dir, exist_ok=True)
# Directory containing controlnet maps
depth_map_dir = "/content/content/test1" # or whichever you have the depthmap images in

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.imread("/content/0000.png")
faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face

v2 = False
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "/content/ip-adapter-faceid-plus_sd15.bin" if not v2 else "ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"

# Control net test
#controlnet_model_path = "/content/checkpoints" #change to local path
#controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"
controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

noise_scheduler = DDIMScheduler(
num_train_timesteps=1000,
beta_start=0.00085,
beta_end=0.012,
beta_schedule="scaled_linear",
clip_sample=False,
set_alpha_to_one=False,
steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
base_model_path,
torch_dtype=torch.float16, controlnet=controlnet,
scheduler=noise_scheduler,
vae=vae,
feature_extractor=None,
safety_checker=None
)

# load ip-adapter
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)
# generate image
prompt = ""
negative_prompt = ""

depth_map_files = [f for f in os.listdir(depth_map_dir) if f.endswith((".jpg", ".png"))]

for idx, filename in enumerate(depth_map_files):
    depth_map_path = os.path.join(depth_map_dir, filename)
    depth_map = load_image(depth_map_path)
    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=depth_map,
        face_image=face_image,
        faceid_embeds=faceid_embeds,
        shortcut=v2,
        s_scale=0.6,
        num_samples=1,  # Generate one image per depth map
        width=512,
        height=512,
        num_inference_steps=10,
        seed=2023,
    )

    # Save the image with the prompt name, date/time, and depth map index
    for i, image in enumerate(images):
        image_name = f"{prompt.replace(' ', '_')}_{date_time}_{idx}_{i}.png"
        image_path = os.path.join(output_dir, image_name)
        image.save(image_path)
#torch.cuda.empty_cache()
#gc.collect()
