from diffusers import ControlNetModel
from keypoint_model.model import StableDiffusionXLControlNet
from keypoint_model.detector import Detector
from keypoint_model.attention import AttentionStore, register_attention_control_sdxl
import torch
import argparse
import os
import ast
from diffusers.utils import load_image

device = "cuda:0" 

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=115, help="random seed")
parser.add_argument("--scale_factor", type=int, default=1000000, help="")
parser.add_argument("--scale_range", type=tuple, default=(1.0, 0.5), help="")
parser.add_argument('--user_prompt', type=str,help='input user prompt')
parser.add_argument("--token_location", type=str, default=None, help="the set of locations where each object appears in the prompt")
args = parser.parse_args()

# Compute openpose conditioning image.
openpose = Detector.from_pretrained("lllyasviel/ControlNet")
image = load_image("figs/keypoint_origin.png")
openpose_image, bboxs = openpose(image)

# Initialize ControlNet pipeline.
controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNet.from_pretrained(
    "stablediffusionapi/albedobase-xl-20", controlnet=controlnet, torch_dtype=torch.float16
).to(device)

controller = AttentionStore()
register_attention_control_sdxl(pipe, controller)

# Infer.
prompt = args.user_prompt
negative_prompt = "low quality, bad quality"
token_location = ast.literal_eval(args.token_location)
save_folder = "generation/" + prompt.replace(" ", "_")
os.makedirs(save_folder, exist_ok=True)


images = pipe(
    prompt=prompt, 
    height = 1024,
    width = 1024,
    negative_prompt=negative_prompt,
    controller = controller,
    num_inference_steps=50,
    num_images_per_prompt=1,
    image=openpose_image.resize((1024, 1024)),
    generator=torch.manual_seed(args.seed),
    device=device,
    bboxs=bboxs,
    scale_factor=args.scale_factor,
    scale_range=args.scale_range,
    token_location=token_location
).images[0]
openpose_image.save(save_folder + "/openpose.png")
images.save(save_folder + "/seed_" + str(args.seed) + "_3.png")
