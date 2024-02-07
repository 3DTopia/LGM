import os
import tyro
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
import gradio as gr

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
GRADIO_VIDEO_PATH = 'gradio_output.mp4'
GRADIO_PLY_PATH = 'gradio_output.ply'

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load dreams
pipe_text = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe_text = pipe_text.to(device)

pipe_image = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe_image = pipe_image.to(device)

# load rembg
bg_remover = rembg.new_session()

# process function
def process(input_image, prompt, prompt_neg='', input_elevation=0, input_num_steps=30, input_seed=42):

    # seed
    kiui.seed_everything(input_seed)

    os.makedirs(opt.workspace, exist_ok=True)
    output_video_path = os.path.join(opt.workspace, GRADIO_VIDEO_PATH)
    output_ply_path = os.path.join(opt.workspace, GRADIO_PLY_PATH)

    # text-conditioned
    if input_image is None:
        mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=7.5, elevation=input_elevation)
        mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
        # bg removal
        mv_image = []
        for i in range(4):
            image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
            # to white bg
            image = image.astype(np.float32) / 255
            image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
            image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
            mv_image.append(image)
    # image-conditioned (may also input text, but no text usually works too)
    else:
        input_image = np.array(input_image) # uint8
        # bg removal
        carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
        mask = carved_image[..., -1] > 0
        image = recenter(carved_image, mask, border_ratio=0.2)
        image = image.astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        mv_image = pipe_image(prompt, image, negative_prompt=prompt_neg, num_inference_steps=input_num_steps, guidance_scale=5.0,  elevation=input_elevation)
        
    mv_image_grid = np.concatenate([
        np.concatenate([mv_image[1], mv_image[2]], axis=1),
        np.concatenate([mv_image[3], mv_image[0]], axis=1),
    ], axis=0)

    # generate gaussians
    input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, output_ply_path)
        
        # render 360 video 
        images = []
        elevation = 0
        if opt.fancy_video:
            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(output_video_path, images, fps=30)

    return mv_image_grid, output_video_path, output_ply_path

# gradio UI

_TITLE = '''LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation'''

_DESCRIPTION = '''
<div>
<a style="display:inline-block" href="https://me.kiui.moe/lgm/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/3DTopia/LGM"><img src='https://img.shields.io/github/stars/3DTopia/LGM?style=social'/></a>
</div>

* Input can be only text, only image, or both image and text. 
* If you find the output unsatisfying, try using different seeds!
'''

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            # input image
            input_image = gr.Image(label="image", type='pil')
            # input prompt
            input_text = gr.Textbox(label="prompt")
            # negative prompt
            input_neg_text = gr.Textbox(label="negative prompt", value='ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate')
            # elevation
            input_elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)
            # inference steps
            input_num_steps = gr.Slider(label="inference steps", minimum=1, maximum=100, step=1, value=30)
            # random seed
            input_seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=0)
            # gen button
            button_gen = gr.Button("Generate")

        
        with gr.Column(scale=1):
            with gr.Tab("Video"):
                # final video results
                output_video = gr.Video(label="video")
                # ply file
                output_file = gr.File(label="ply")
            with gr.Tab("Multi-view Image"):
                # multi-view results
                output_image = gr.Image(interactive=False, show_label=False)

        button_gen.click(process, inputs=[input_image, input_text, input_neg_text, input_elevation, input_num_steps, input_seed], outputs=[output_image, output_video, output_file])
    
    gr.Examples(
        examples=[
            "data_test/anya_rgba.png",
            "data_test/bird_rgba.png",
            "data_test/catstatue_rgba.png",
        ],
        inputs=[input_image],
        outputs=[output_image, output_video, output_file],
        fn=lambda x: process(input_image=x, prompt=''),
        cache_examples=False,
        label='Image-to-3D Examples'
    )

    gr.Examples(
        examples=[
            "a motorbike",
            "a hamburger",
            "a furry red fox head",
        ],
        inputs=[input_text],
        outputs=[output_image, output_video, output_file],
        fn=lambda x: process(input_image=None, prompt=x),
        cache_examples=False,
        label='Text-to-3D Examples'
    )
    
block.launch(server_name="0.0.0.0", share=False)