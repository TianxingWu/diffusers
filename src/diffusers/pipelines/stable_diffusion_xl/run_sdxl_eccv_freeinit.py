from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLFreeInitPipeline
from diffusers import DDIMScheduler
import torch
from omegaconf import OmegaConf
import os
from diffusers.training_utils import set_seed

# pipeline_sdxl_freeinit = StableDiffusionXLFreeInitPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

pipeline_sdxl_freeinit = StableDiffusionXLFreeInitPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

pipeline_sdxl_freeinit.scheduler = DDIMScheduler.from_config(pipeline_sdxl_freeinit.scheduler.config)


W = 1024
H = 1024
fi_config = OmegaConf.load('/home/tianxing001_e_ntu_edu_sg/project/diffusers/src/diffusers/pipelines/stable_diffusion_xl/freeinit_config.yaml')
filter_params = fi_config.filter_params

pipeline_sdxl_freeinit.init_filter(
                width               = W,
                height              = H,
                video_length        = 1,
                filter_params       = filter_params,
            )


# prompt = "Astronaut in a jungle, cold color palette, muted colors, highly detailed, 8k"
# prompt = "A dark room with dim light on the table"
prompt = "A dim dark room with strong light above the table"

# prompt = "The light is so bright that the canvas is almost white"
# prompt = "A white image"
# prompt = "A white canvas with nothing on it"
# prompt = "Small line-art logo at the center of pure white background"
# prompt = "A black dot at the center of pure white background"
# prompt = "Clean, blue sky with no cloud"
# prompt = "Superhero's shadow in dark alley at night"
# prompt = "A very dark alleyway with graffiti after rainstorm at dark night"



seed = 42

save_dir = f"results/{prompt.replace(' ', '_').replace(',', '_')}_{fi_config.filter_params.method}_seed{seed}_nofilter"
os.makedirs(save_dir, exist_ok=True)
with open(f'{save_dir}/prompt.txt', 'w') as file:
                    file.write(f'{prompt}')

OmegaConf.save(fi_config, os.path.join(save_dir, "freeinit_args.yaml"))

set_seed(seed)
pipeline_sdxl_freeinit.eccv_sample(
    prompt=prompt,
    height=H,
    width=W,
    # freeinit args
    num_iters=fi_config.num_iters,
    use_fast_sampling=False,
    # save_intermediate: bool = False,
    # return_orig: bool = False,
    save_dir= save_dir,
    # save_name: str = None,
)


# image = pipeline_text2image(prompt=prompt).images[0]
# image