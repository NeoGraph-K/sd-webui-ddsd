# sd-webui-ddsd
A script that searches for specific keywords, inpaints them, and then upscales them

## What is
### Controlnet Random
[Controlnet](https://github.com/Mikubill/sd-webui-controlnet) random sample image. usage glob
### Upscale
Upscaling an image by a specific factor. Utilizes a tiled approach to scale with less memory
### Detect Detailer
Inpainting with additional prompts after mask search with specific keywords. Add counts separated by semicolons
#### Detect Detailer How to use
0. Enable Inpaint Inner(or Outer) Mask Area(Use I2I Only)
    1. When using the inpaint inner option, the mask is created only inside the inpaint mask.
    2. When using the inpaint outer option, the mask is created only outside the inpaint mask.
1. Input dino prompt
    1. Inpaint the dino prompt multiple times, separated by tabs.
    2. Additional options can be controlled.
    3. Each dino prompt can be calculated with AND, OR, XOR, NOR, and NAND gates.
        1. face OR (body NAND outfit) -> Create a body mask that does not overlap with the outfit. And composited with a face mask.
        2. Use parentheses sparingly. Parentheses operations consume more VRAM because they generate masks in advance.
    4. Option values ​​of each dino prompt can be entered by separating them with colons.
        1. face:0:0.4:4 OR outfit:2:0.5:8
        2. Each option, in order, is prompt, detection level (0-2:default 0), box threshold (0-1:default 0.3), and dilation value (0-128:default 8).
        3. You can omit it if you wish. Replace with default value if omitted.
2. Input positive prompt
    1. Inpaint the positive prompt multiple times, separated by semicolons.
3. Input negative prompt
    1. Inpaint the negative prompt multiple times, separated by semicolons.
4. Check the option to separate and inpaint the unconnected mask.
    1. When separating and inpainting, the number of inpaintings increases. But quality rises.
5. Select a small area of ​​pixels to remove from the inpainting area when inpainting by isolation.
6. Generate!
## Installation
1. Download [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
    1. You need current CUDA and cuDNN version
    2. This is [CUDA 117](https://drive.google.com/file/d/1HRTOLTB44-pRcrwIw9lQak2OC2ohNle3/view?usp=share_link) and [cuDNN](https://drive.google.com/file/d/1QcgaxUra0WnCWrCLjsWp_QKw1PKcvqpj/view?usp=share_link)
2. After installing CUDA, overwrite cuDNN in the folder where you installed CUDA
3. Install from the extensions tab with url `https://github.com/NeoGraph-K/sd-webui-ddsd`
4. Start Sd web UI
5. It takes some time to install sam model and dino model

## Credits

dustysys/[ddetailer](https://github.com/dustysys/ddetailer)

AUTOMATIC1111/[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

facebookresearch/[Segment Anything](https://github.com/facebookresearch/segment-anything)

IDEA-Research/[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

IDEA-Research/[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

continue-revolution/[sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything)
