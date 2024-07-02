# from diffusers import StableVideoDiffusionPipelineAblation

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineAblation
from diffusers.utils import load_image, export_to_video
import torch
import os
from torchvision.io import write_video
import numpy as np

abl_type = "zero_input" # "full" "zero_embed" "zero_input" "no_input_conv"

generator = torch.Generator()
# seed = generator.seed()
seed = 42
generator.manual_seed(seed)
# pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
# pipe = StableVideoDiffusionPipelineAblation.from_pretrained("stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16")
pipe = StableVideoDiffusionPipelineAblation.from_pretrained("stabilityai/stable-video-diffusion-img2vid")
import pdb
pdb.set_trace()

pipe.to("cuda")

input_img_path = "/home/tianxing001_e_ntu_edu_sg/project/diffusers/assets/cond_imgs/anya_dance.png"

image = load_image(input_img_path)
image = image.resize((1024, 576))

frames = pipe(image, num_frames=14, decode_chunk_size=8, abl_type=abl_type, generator=generator).frames[0]
# export_to_video(frames, "/home/tianxing001_e_ntu_edu_sg/project/diffusers/outputs/generated.mp4", fps=7)

img_name = os.path.splitext(os.path.basename(input_img_path))[0]
# video_path = os.path.join(output_folder, f"{img_name}_seed{seed}.mp4")
video_path = os.path.join("/home/tianxing001_e_ntu_edu_sg/project/diffusers/outputs/", f"{img_name}_{abl_type}_seed{seed}.mp4")
# frames = [np.array(frame) for frame in frames]
frames = np.array([np.array(frame) for frame in frames])
write_video(video_path, frames, fps=7)