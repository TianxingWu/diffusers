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
fi_config = OmegaConf.load('/mnt/sfs-common/txwu/project/diffusers/src/diffusers/pipelines/stable_diffusion_xl/freeinit_config.yaml')
filter_params = fi_config.filter_params

pipeline_sdxl_freeinit.init_filter(
                width               = W,
                height              = H,
                video_length        = 1,
                filter_params       = filter_params,
            )


# prompt = "Astronaut in a jungle, cold color palette, muted colors, highly detailed, 8k"
# prompt = "A dark room with dim light on the table"
# prompt = "A dim dark room with strong light above the table"

# prompt = "The light is so bright that the canvas is almost white"
# prompt = "A white image"
# prompt = "A white canvas with nothing on it"
# prompt = "Small line-art logo at the center of pure white background"
# prompt = "A black dot at the center of pure white background"
# prompt = "Clean, blue sky with no cloud"
# prompt = "Superhero's shadow in dark alley at night"
# prompt = "A very dark alleyway with graffiti after rainstorm at dark night"

# prompt = "a cute fluffy rabbit pilot walking on a military aircraft carrier, unreal engine render, 8k, cinematic"
# prompt = "dark moon night, a cute fluffy rabbit pilot walking on a military aircraft carrier, unreal engine render, 8k, cinematic"
# prompt = "dark night, a cute rabbit pilot walking on a military aircraft carrier, unreal engine render, 8k, cinematic"
# prompt = "a cute fluffy rabbit pilot walking on a military aircraft carrier in dark night, unreal engine render, 8k, cinematic"
# prompt = "a cute fluffy rabbit pilot with headlight walking in dark tunnel, unreal engine render, 8k, cinematic"
# prompt = "an ant walking in dark tunnel with dim light, 8k, cinematic"
# prompt = "ant navigating the inside of an ant nest, 8k, cinematic"
# prompt = "an ant navigating the inside of an very dark ant nest, 8k, cinematic"
# prompt = "an ant walking the inside of an very dark ant nest, dim light, 8k, cinematic"


# prompt = "a painting of a city at night with a moon in the sky and a bridge, night scene"

# prompt = "a painting of a city at night with a moon in the sky and a bridge, night scene, carl gustav carus, night landscape, nighttime scene, by August Friedrich Schenck, night time scene, Jakub Schikaneder, at night with dramatic moonlight, by Johann Kretzschmer, german romanticism, Andreas Achenbach, by Friedrich Traffelet, Oswald Achenbach, by Heinrich Brocksieper, by Johann Bodin, achenbach, moonlight night, by Andreas Achenbach, by Friedrich Ritter von Friedländer-Malheim, by Wilhelm Schnarrenberger"

# prompt = "a city at very dark night with a moon in the sky and a bridge, night scene"

# prompt = "Shadowed silhouette of a brunette boy and blonde girl, both 18, in sci-fi suits, sitting in a dark futuristic room. Background features a window view of Earth from space, hyperrealistic, 8K."

# prompt = "In dark room, a large stack of vintage televisions all showing different programs — 1950s sci-fi movies, horror movies, news, static, a 1970s sitcom, etc, set inside a large New York museum gallery."

# prompt = "in very dark night, a wolf howling at the moon, forest in the background"

# prompt = "dark face at night, dim green light"
# prompt = "dark face at night, dim green light, realistic"
# prompt = "dark apple at night, dim green light"
# prompt = "blue apple at dark night, dim green light"
# prompt = "a crystal at night, dim pink light"
# prompt = "a piece of pink crystal at very dark night, dim light"
# prompt = "a small shinning ring at very dark night, dim light"

# prompt = "a bird logo at the center of pure white background"
# prompt = "a drop of water in the dark, dim light"

# prompt = "a skull with red eyes in the dark, dim light, cinematic"

# prompt = "a wet red rose on pure white background"

# prompt = "a bird logo is being painted on the pure white canvas"
# prompt = "a blue bird logo painted on pure white canvas"

# prompt = "a euphonium in the dark with strong light above"
# prompt = "an euphonium in the dark with strong light above"
# prompt = "a trumpet instrument in the dark with warm light above"
# prompt = "an euphonium in the corner of a dark room, instrument, dim warm light above, cinematic"
# prompt = "an euphonium lying at the corner of a very dark room, instrument, dim warm light above, cinematic"
# prompt = "a small euphonium in dark, instrument, dim warm light above, cinematic"
# prompt = "a trumpet with red rose in dark, dim warm light, cinematic"

# prompt = "2d euphonium logo on pure white canvas"
# prompt = "a trumpet logo on pure white canvas"
# prompt = "trumpet with music notes, round black logo, minimalism design, white backgound"
# prompt = "euphonium, round black logo, minimalism design, white backgound"

# prompt = "small blue bird on a concert flute in the dark, dim side light, cinematic"
# prompt = "small blue bird on a concert flute in the dark, dim top light, cinematic, rich details"

# prompt = "small blue bird on a horizontal concert flute at night, dim warm light"
# prompt = "A small blue bird with delicate feathers perched on a polished, horizontal concert flute. The scene is set at night, illuminated by a dim, warm light, creating a serene and magical atmosphere."
# prompt = "A small blue bird perched on a horizontal concert flute under dim, warm night light."
# prompt = "very dark scene, a small blue bird perched on a concert flute under dim, warm light."
# prompt = "in the dark, a small blue bird on a polished concert flute under dim light."
# prompt = "small blue bird on a silver concert flute in the dark, dim light."
# prompt = "small blue bird on a silver concert flute at night, dark dim light"
# prompt = "small blue bird on a silver concert flute, very dark environment, dim light, rich details, cinematic"


# prompt = "a blue electric guitar, round logo, minimalism design, pure white background"
# prompt = "blue electric guitar icon, minimalism design, pure white background"
# prompt = "small blue electric guitar pendant at the center of pure white background"
# prompt = "blue electric guitar in the dark, dim pink light, cinematic"
# prompt = "blue electric guitar in the dark, close look, dim light, cinematic"
# prompt = "Gibson Heritage Cherry Sunburst guitar in the dark, close-up, dim light, cinematic"
# prompt = "a red guitar in the dark, dim light, cinematic"
# prompt = "a red guitar in the dark, almost no light, cinematic"
# prompt = "a red guitar in very dark doom, with spotlight, 8k cinematic"
# prompt = "a small red car toy in very dark doom, top spotlight"

# prompt = "a yellow car toy on pure white background"
# prompt = "a yellow dot on white canvas"
# prompt = "one yellow drop on pure white background"
# prompt = "A small stroke of yellow oil paint on pure white background"
prompt = "Yellow oil paint stroke on white background"


seed = 42

if len(prompt) > 200:
    save_dir = f"results/{(prompt[:200] + '_etal').replace(' ', '_').replace(',', '_')}_{fi_config.filter_params.method}_seed{seed}_nofilter"
    save_dir = f"results/{(prompt[:200] + '_etal').replace(' ', '_').replace(',', '_')}_{fi_config.filter_params.method}_seed{seed}"
else:
    # save_dir = f"results/{prompt.replace(' ', '_').replace(',', '_')}_{fi_config.filter_params.method}_seed{seed}_nofilter"
    save_dir = f"results/{prompt.replace(' ', '_').replace(',', '_')}_{fi_config.filter_params.method}_seed{seed}"
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