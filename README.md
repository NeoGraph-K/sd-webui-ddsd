# sd-webui-ddsd
A script that searches for specific keywords, inpaints them, and then upscales them

## What is
### Controlnet Random
[Controlnet](https://github.com/Mikubill/sd-webui-controlnet) random sample image. usage glob
### Upscale
Upscaling an image by a specific factor. Utilizes a tiled approach to scale with less memory
### Detect Detailer
Inpainting with additional prompts after mask search with specific keywords. Add counts separated by semicolons

## Installation
1. Download [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
    1. You need current CUDA and cuDNN version
    2. This is [CUDA 117](https://drive.google.com/file/d/1HRTOLTB44-pRcrwIw9lQak2OC2ohNle3/view?usp=share_link) and [cuDNN](https://drive.google.com/file/d/1QcgaxUra0WnCWrCLjsWp_QKw1PKcvqpj/view?usp=share_link)
2. After installing CUDA, overwrite cuDNN in the folder where you installed CUDA
3. Install from the extensions tab with url `https://github.com/NeoGraph-K/sd-webui-ddsd/`
4. Start Sd web UI
5. It takes some time to install sam model and dino model

## Credits

dustysys/[ddetailer](https://github.com/dustysys/ddetailer)

AUTOMATIC1111/[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

facebookresearch/[Segment Anything](https://github.com/facebookresearch/segment-anything)

IDEA-Research/[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

IDEA-Research/[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

continue-revolution/[sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything)
