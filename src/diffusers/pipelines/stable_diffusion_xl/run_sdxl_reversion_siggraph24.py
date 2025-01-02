from diffusers import AutoPipelineForText2Image
from diffusers import DDIMScheduler
import torch
import os


#=============
import math
from typing import List

from PIL import Image


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image

#=============

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
# pipeline_text2image = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# pipeline_text2image.scheduler = DDIMScheduler.from_config(pipeline_text2image.scheduler.config)


prompt = "dog is on top of plate"
# prompt = "Two kangaroos holding hands and spinning, both leaning backwards"

save_dir = f"results/_results_reversion_siggraph24/sdxl/{prompt}"
# save_dir = f"results/_results_reversion_siggraph24/sd15/{prompt}"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/samples", exist_ok=True)

images = pipeline_text2image(prompt=prompt, num_images_per_prompt=10).images
for idx, image in enumerate(images):
    image.save(os.path.join(f"{save_dir}/samples", f"{idx}.png"))


joined_image = get_image_grid(images)
joined_image.save(os.path.join(save_dir, f"{prompt}.png"))