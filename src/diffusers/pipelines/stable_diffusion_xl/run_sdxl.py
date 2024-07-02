from diffusers import AutoPipelineForText2Image
from diffusers import DDIMScheduler
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

# print(pipeline_text2image.scheduler.compatibles)

pipeline_text2image.scheduler = DDIMScheduler.from_config(pipeline_text2image.scheduler.config)


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt).images[0]
image.save("test.png")