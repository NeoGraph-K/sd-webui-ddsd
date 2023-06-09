from json import load as json_read, dumps as json_write
import os
import math
import re

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from scripts.ddsd_sam import sam_model_list
from scripts.ddsd_dino import dino_model_list
from scripts.ddsd_postprocess import lut_model_list, ddsd_postprocess
from scripts.ddsd_utils import dino_detect_from_prompt, mask_spliter_and_remover, I2I_Generator_Create, get_fonts_list, image_apply_watermark, matched_noise

import modules
from modules import processing, shared, images, devices, modelloader, sd_models, sd_vae
from modules.processing import create_infotext, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
from modules.sd_models import model_hash
from modules.paths import models_path
from modules.scripts import AlwaysVisible

from basicsr.utils.download_util import load_file_from_url

grounding_models_path = os.path.join(models_path, "grounding")
sam_models_path = os.path.join(models_path, "sam")
lut_models_path = os.path.join(models_path, 'lut')
yolo_models_path = os.path.join(models_path, 'yolo')
ddsd_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'config')

ckpt_model_name_pattern = re.compile('([\\w\\.\\[\\]\\\\\\+\\(\\)/]+)\\s*\\[.*\\]')

def list_models(model_path, filter):
        model_list = modelloader.load_models(model_path=model_path, ext_filter=[filter])
        
        def modeltitle(path, shorthash):
            abspath = os.path.abspath(path)

            if abspath.startswith(model_path):
                name = abspath.replace(model_path, '')
            else:
                name = os.path.basename(path)

            if name.startswith("\\") or name.startswith("/"):
                name = name[1:]

            shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

            return f'{name} [{shorthash}]', shortname
        
        models = []
        for filename in model_list:
            h = model_hash(filename)
            title, short_model_name = modeltitle(filename, h)
            models.append(title)
        
        return models
        
def startup():
    if (len(list_models(yolo_models_path, '.pth')) == 0) and (len(list_models(yolo_models_path, '.pt')) == 0):
        print("No detection yolo models found, downloading...")
        load_file_from_url('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt',yolo_models_path)
        load_file_from_url('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt',yolo_models_path)
        load_file_from_url('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt',yolo_models_path)
        load_file_from_url('https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt',yolo_models_path)
        load_file_from_url('https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt',yolo_models_path)
        load_file_from_url('https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt',yolo_models_path)
        
    if (len(list_models(grounding_models_path, '.pth')) == 0):
        print("No detection groundingdino models found, downloading...")
        load_file_from_url('https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth',grounding_models_path)
        load_file_from_url('https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py',grounding_models_path, file_name='groundingdino_swint_ogc.py')
        #load_file_from_url('https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth',grounding_models_path)
        #load_file_from_url('https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB.cfg.py',grounding_models_path, file_name='groundingdino_swinb_cogcoor.py')
        
        
    if (len(list_models(sam_models_path, '.pth')) == 0):
        print("No detection sam models found, downloading...")
        #load_file_from_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',sam_models_path)
        #load_file_from_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',sam_models_path)
        load_file_from_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',sam_models_path)
        
    if (len(list_models(lut_models_path, '.cube')) == 0): # Free use lut files.
        print('No detection lut models found, downloading...')
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Arabica%2012.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Ava%20614.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Azrael%2093.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Bourbon%2064.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Byers%2011.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Chemical%20168.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Clayton%2033.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Clouseau%2054.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Cobi%203.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Contrail%2035.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Cubicle%2099.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Django%2025.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Domingo%20145.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineBasic.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineBright.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineCold.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineDrama.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineTealOrange1.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineTealOrange2.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineVibrant.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/FGCineWarm.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Faded%2047.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Folger%2050.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Fusion%2088.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Hyla%2068.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Korben%20214.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/LBK-K-Tone_33.cube', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Lenox%20340.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Lucky%2064.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/McKinnon%2075.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Milo%205.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Neon%20770.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Paladin%201875.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Pasadena%2021.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Pitaya%2015.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Reeve%2038.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Remy%2024.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Sprocket%20231.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Teigen%2028.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Trent%2018.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Tweed%2071.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Vireo%2037.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Zed%2032.CUBE', lut_models_path)
        load_file_from_url('https://huggingface.co/datasets/NeoGraph/Luts_Cube/resolve/main/Zeke%2039.CUBE', lut_models_path)

startup()

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

def gr_list_refresh(choices, value):
    return {'choices':choices,'value':value,'__type__':'update'}
def gr_value_refresh(value):
    return {'value':value,'__type__':'update'}
class Script(modules.scripts.Script):
    def __init__(self):
        self.original_scripts = None
        self.original_scripts_always = None
        _ ,self.font_path = get_fonts_list()
        self.ckptname = None
        self.vae = None
        self.clip_skip = 1
        
    def title(self):
        return "ddetailer + sdupscale"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        pp_types = [
            'none', 
            'saturation','sharpening','gaussian blur','brightness','color','contrast',
            #'color extraction',
            'hue', 'inversion', 'bilateral','color tint(type)','color tint(lut)']
        ckpt_list = list(sd_models.checkpoints_list.keys())
        ckpt_list.insert(0, 'Original')
        vae_list = list(sd_vae.vae_dict.keys())
        vae_list.insert(0, 'Original')
        sample_list = [x.name for x in shared.list_samplers()]
        sample_list = [x for x in sample_list if x not in ['PLMS','UniPC','DDIM']]
        sample_list.insert(0,"Original")
        fonts_list, _ = get_fonts_list()
        ddsd_config_list = [x[:-6] for x in os.listdir(ddsd_config_path) if x.endswith('.ddcfg')]
        ret = []
        dino_detection_ckpt_list = []
        dino_detection_vae_list = []
        dino_detection_prompt_list = []
        dino_detection_positive_list = []
        dino_detection_negative_list = []
        dino_detection_denoise_list = []
        dino_detection_cfg_list = []
        dino_detection_steps_list = []
        dino_detection_spliter_disable_list = []
        dino_detection_spliter_remove_area_list = []
        dino_detection_clip_skip_list = []
        pp_type_list = []
        pp_saturation_strength_list = []
        pp_sharpening_radius_list = []
        pp_sharpening_percent_list = []
        pp_sharpening_threshold_list = []
        pp_gaussian_radius_list = []
        pp_brightness_strength_list = []
        pp_color_strength_list = []
        pp_contrast_strength_list = []
        pp_hue_strength_list = []
        pp_bilateral_sigmaC_list = []
        pp_bilateral_sigmaS_list = []
        pp_color_tint_type_name_list = []
        pp_color_tint_lut_name_list = []
        watermark_type_list = []
        watermark_position_list = []
        watermark_image_list = []
        watermark_image_size_width_list = []
        watermark_image_size_height_list = []
        watermark_text_list = []
        watermark_text_color_list = []
        watermark_text_font_list = []
        watermark_text_size_list = []
        watermark_padding_list = []
        watermark_alpha_list = []
        outpaint_positive_list = []
        outpaint_negative_list = []
        outpaint_denoise_list = []
        outpaint_cfg_list = []
        outpaint_steps_list = []
        outpaint_pixels_list = []
        outpaint_direction_list = []
        dino_tabs = None
        watermark_tabs = None
        postprocess_tabs = None
        outpaint_tabs = None
        
        with gr.Accordion('DDSD', open=False, elem_id='ddsd_all_option_acc'):
            
            with gr.Row():
                ddsd_save_path = gr.Textbox(label='Save File Name', visible=True, interactive=True, value='ddsd')
                ddsd_save = gr.Button('Save', elem_id='save_button', visible=True, interactive=True)
            with gr.Row():
                ddsd_load_path = gr.Dropdown(label='Load File Name', visible=True, interactive=True, choices=ddsd_config_list)
                ddsd_load = gr.Button('Load', elem_id='load_button',visible=True, interactive=True)

            with gr.Accordion("Script Option", open = False, elem_id="ddsd_enable_script_acc"):
                with gr.Column():
                    all_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I All process target script</p>')
                    enable_script_names = gr.Textbox(label="Enable Script(Extension)", elem_id="enable_script_names", value='dynamic_thresholding;dynamic_prompting',show_label=True, lines=1, placeholder="Extension python file name(ex - dynamic_thresholding;dynamic_prompting)")
            
            with gr.Accordion("Outpainting", open=False, elem_id='ddsd_outpaint_acc'):
                with gr.Column():
                    outpaint_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Outpainting</p>')
                    disable_outpaint = gr.Checkbox(label='Disable Outpaint', elem_id='disable_outpaint', value=True, visible=True)
                    outpaint_sample = gr.Dropdown(label='Outpaint Sampling', elem_id='outpaint_sample', choices=sample_list, value=sample_list[0], visible=False, type="value")
                    with gr.Tabs(elem_id = 'outpaint_arguments', visible=False) as outpaint_tabs_acc:
                        for outpaint_index in range(shared.opts.data.get('outpaint_count', 1)):
                            with gr.Tab(f'Outpaint {outpaint_index + 1} Argument', elem_id=f'outpaint_{outpaint_index+1}_argument_tab'):
                                outpaint_pixels = gr.Slider(label=f'Outpaint {outpaint_index+1} Pixels', minimum=0, maximum=256, value=64, step=16)
                                outpaint_direction = gr.Radio(choices=['None', 'Left','Right','Up','Down'], value='None', label=f'Outpaint {outpaint_index+1} Direction')
                                with gr.Row():
                                    outpaint_positive = gr.Textbox(label=f'Positive {outpaint_index+1} Prompt', show_label=True, lines=2, placeholder='Outpaint Positive Prompt(Empty is Original)')
                                    outpaint_negative = gr.Textbox(label=f'Negative {outpaint_index+1} Prompt', show_label=True, lines=2, placeholder='Outpaint Negative Prompt(Empty is Original)')
                                outpaint_denoise = gr.Slider(label=f'Outpaint {outpaint_index+1} Denoise', minimum=0, maximum=1.0, step=0.01, value=0.8)
                                outpaint_cfg = gr.Slider(label=f'Outpaint {outpaint_index+1} CFG(0 To Original)', minimum=0, maximum=500, step=0.5, value=0)
                                outpaint_steps = gr.Slider(label=f'Outpaint {outpaint_index+1} Steps(0 To Original)', minimum=0, maximum=150, step=1, value=0)
                            outpaint_positive_list.append(outpaint_positive)
                            outpaint_negative_list.append(outpaint_negative)
                            outpaint_denoise_list.append(outpaint_denoise)
                            outpaint_cfg_list.append(outpaint_cfg)
                            outpaint_steps_list.append(outpaint_steps)
                            outpaint_pixels_list.append(outpaint_pixels)
                            outpaint_direction_list.append(outpaint_direction)
                        outpaint_tabs = outpaint_tabs_acc
                    outpaint_mask_blur = gr.Slider(label='Outpaint Blur', elem_id='outpaint_mask_blur', minimum=0, maximum=128, step=4, value=8, visible=False)
        
            with gr.Accordion("Upscaler", open=False, elem_id="ddsd_upscaler_acc"):
                with gr.Column():
                    sd_upscale_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Upscaler Option</p>')
                    disable_upscaler = gr.Checkbox(label='Disable Upscaler', elem_id='disable_upscaler', value=True, visible=True)
                    ddetailer_before_upscaler = gr.Checkbox(label='Upscaler before running detailer', elem_id='upscaler_before_running_detailer', value=False, visible=False)
                    with gr.Row():
                        upscaler_sample = gr.Dropdown(label='Upscaler Sampling', elem_id='upscaler_sample', choices=sample_list, value=sample_list[0], visible=False, type="value")
                        upscaler_index = gr.Dropdown(label='Upscaler', elem_id='upscaler_index', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[-1].name, type="index", visible=False)
                    with gr.Row():
                        upscaler_ckpt = gr.Dropdown(label='Upscaler CKPT Model', elem_id=f'upscaler_detect_ckpt', choices=ckpt_list, value=ckpt_list[0], visible=False)
                        upscaler_vae = gr.Dropdown(label='Upscaler VAE Model', elem_id=f'upscaler_detect_vae', choices=vae_list, value=vae_list[0], visible=False)
                    scalevalue = gr.Slider(minimum=1, maximum=16, step=0.5, elem_id='upscaler_scalevalue', label='Resize', value=2, visible=False)
                    overlap = gr.Slider(minimum=0, maximum=256, step=32, elem_id='upscaler_overlap', label='Tile overlap', value=32, visible=False)
                    with gr.Row():
                        rewidth = gr.Slider(minimum=0, maximum=1024, step=64, elem_id='upscaler_rewidth', label='Width(0 to No Inpainting)', value=512, visible=False)
                        reheight = gr.Slider(minimum=0, maximum=1024, step=64, elem_id='upscaler_reheight', label='Height(0 to No Inpainting)', value=512, visible=False)
                    denoising_strength = gr.Slider(minimum=0, maximum=1.0, step=0.01, elem_id='upscaler_denoising', label='Denoising strength', value=0.1, visible=False)
        
            with gr.Accordion("DINO Detect", open=False, elem_id="ddsd_dino_detect_acc"):
                with gr.Column():
                    ddetailer_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Detection Detailer Option</p>')
                    disable_detailer = gr.Checkbox(label='Disable Detection Detailer', elem_id='disable_detailer',value=True, visible=True)
                    disable_mask_paint_mode = gr.Checkbox(label='Disable I2I Mask Paint Mode', value=True, visible=False)
                    inpaint_mask_mode = gr.Radio(choices=['Inner', 'Outer'], value='Inner', label='Inpaint Mask Paint Mode', visible=False, show_label=True)
                    detailer_sample = gr.Dropdown(label='Detailer Sampling', elem_id='detailer_sample', choices=sample_list, value=sample_list[0], visible=False, type="value")
                    with gr.Row():
                        detailer_sam_model = gr.Dropdown(label='Detailer SAM Model', elem_id='detailer_sam_model', choices=sam_model_list(), value=sam_model_list()[0], visible=False)
                        detailer_dino_model = gr.Dropdown(label='Detailer DINO Model', elem_id='detailer_dino_model', choices=dino_model_list(), value=dino_model_list()[0], visible=False)
                    with gr.Tabs(elem_id = 'dino_detct_arguments', visible=False) as dino_tabs_acc:
                        for index in range(shared.opts.data.get('dino_detect_count', 2)):
                            with gr.Tab(f'DINO {index + 1} Argument', elem_id=f'dino_{index + 1}_argument_tab'):
                                with gr.Row():
                                    dino_detection_ckpt = gr.Dropdown(label='Detailer CKPT Model', elem_id=f'detailer_detect_ckpt_{index+1}', choices=ckpt_list, value=ckpt_list[0], visible=True)
                                    dino_detection_vae = gr.Dropdown(label='Detailer VAE Model', elem_id=f'detailer_detect_vae_{index+1}', choices=vae_list, value=vae_list[0], visible=True)
                                dino_detection_prompt = gr.Textbox(label=f"Detect {index + 1} Prompt", elem_id=f"detailer_detect_prompt_{index + 1}", show_label=True, lines=2, placeholder="Detect Token Prompt(ex - face:level(0-2):threshold(0-1):dilation(0-128))", visible=True)
                                with gr.Row():
                                    dino_detection_positive = gr.Textbox(label=f"Positive {index + 1} Prompt", elem_id=f"detailer_detect_positive_{index + 1}", show_label=True, lines=2, placeholder="Detect Mask Inpaint Positive(ex - perfect anatomy)", visible=True)
                                    dino_detection_negative = gr.Textbox(label=f"Negative {index + 1} Prompt", elem_id=f"detailer_detect_negative_{index + 1}", show_label=True, lines=2, placeholder="Detect Mask Inpaint Negative(ex - nsfw)", visible=True)
                                dino_detection_denoise = gr.Slider(minimum=0, maximum=1.0, step=0.01, elem_id=f'dino_detect_{index+1}_denoising', label=f'DINO {index + 1} Denoising strength', value=0.4, visible=True)
                                dino_detection_cfg = gr.Slider(minimum=0, maximum=500, step=0.5, elem_id=f'dino_detect_{index+1}_cfg_scale', label=f'DINO  {index + 1} CFG Scale(0 to Origin)', value=0, visible=True)
                                dino_detection_steps = gr.Slider(minimum=0, maximum=150, step=1, elem_id=f'dino_detect_{index+1}_steps', label=f'DINO {index + 1} Steps(0 to Origin)', value=0, visible=True)
                                dino_detection_spliter_disable = gr.Checkbox(label=f'Disable DINO {index + 1} Detect Split Mask', value=True, visible=True)
                                dino_detection_spliter_remove_area = gr.Slider(minimum=0, maximum=800, step=8, elem_id=f'dino_detect_{index+1}_remove_area', label=f'Remove {index + 1} Area', value=16, visible=True)
                                dino_detection_clip_skip = gr.Slider(minimum=0, maximum=10, step=1, elem_id=f'dino_detect_{index+1}_clip_skip', label=f'Clip skip {index + 1} Inpaint(0 to Origin)', value=0, visible=True)
                                dino_detection_ckpt_list.append(dino_detection_ckpt)
                                dino_detection_vae_list.append(dino_detection_vae)
                                dino_detection_prompt_list.append(dino_detection_prompt)
                                dino_detection_positive_list.append(dino_detection_positive)
                                dino_detection_negative_list.append(dino_detection_negative)
                                dino_detection_denoise_list.append(dino_detection_denoise)
                                dino_detection_cfg_list.append(dino_detection_cfg)
                                dino_detection_steps_list.append(dino_detection_steps)
                                dino_detection_spliter_disable_list.append(dino_detection_spliter_disable)
                                dino_detection_spliter_remove_area_list.append(dino_detection_spliter_remove_area)
                                dino_detection_clip_skip_list.append(dino_detection_clip_skip)
                        dino_tabs = dino_tabs_acc
                    dino_full_res_inpaint = gr.Checkbox(label='Inpaint at full resolution ', elem_id='detailer_full_res', value=True, visible = False)
                    with gr.Row():
                        dino_inpaint_padding = gr.Slider(label='Inpaint at full resolution padding, pixels ', elem_id='detailer_padding', minimum=0, maximum=256, step=4, value=0, visible=False)
                        detailer_mask_blur = gr.Slider(label='Detailer Blur', elem_id='detailer_mask_blur', minimum=0, maximum=64, step=1, value=4, visible=False)
                
            with gr.Accordion("Postprocessing", open=False, elem_id='ddsd_post_processing'):
                with gr.Column():
                    postprocess_info = gr.HTML('<br><p style="margin-bottom:0.75em">Postprocessing to the final image</p>')
                    disable_postprocess = gr.Checkbox(label='Disable PostProcess', elem_id='disable_postprocess',value=True, visible=True)
                    with gr.Tabs(elem_id = 'ddsd_postprocess_arguments', visible=False) as postprocess_tabs_acc:
                        for index in range(shared.opts.data.get('postprocessing_count', 1)):
                            with gr.Tab(f'Postprocessing {index + 1} Argument', elem_id=f'postprocessing_{index + 1}_argument_tab'):
                                pp_type = gr.Dropdown(label=f'Postprocessing type {index+1}', elem_id=f'postprocessing_{index+1}', choices=pp_types, value=pp_types[0], visible=True)
                                pp_saturation_strength = gr.Slider(label=f'Saturation strength {index+1}', minimum=0, maximum=3, step=0.01, value=1.1, visible=False)
                                pp_sharpening_radius = gr.Slider(label=f'Sharpening radius {index+1}', minimum=0, maximum=50, step=1, value=2, visible=False)
                                pp_sharpening_percent = gr.Slider(label=f'Sharpening percent {index+1}', minimum=0, maximum=300, step=1, value=150, visible=False)
                                pp_sharpening_threshold = gr.Slider(label=f'Sharpening threshold {index+1}', minimum=0, maximum=10, step=0.01, value=3, visible=False)
                                pp_gaussian_radius = gr.Slider(label=f'Gaussian Blur radius {index+1}', minimum=0, maximum=50, step=1, value=2, visible=False)
                                pp_brightness_strength = gr.Slider(label=f'Brightness strength {index+1}', minimum=0, maximum=5, step=0.01, value=1.1, visible=False)
                                pp_color_strength = gr.Slider(label=f'Color strength {index+1}', minimum=0, maximum=5, step=0.01, value=1.1, visible=False)
                                pp_contrast_strength = gr.Slider(label=f'Contrast strength {index+1}', minimum=0, maximum=5, step=0.01, value=1.1, visible=False)
                                pp_hue_strength = gr.Slider(label=f'Hue strength {index+1}', minimum=-1, maximum=1, step=0.01, value=0, visible=False)
                                pp_bilateral_sigmaC = gr.Slider(label=f'Bilateral sigmaC {index+1}', minimum=0, maximum=100, step=1, value=10, visible=False)
                                pp_bilateral_sigmaS = gr.Slider(label=f'Bilateral sigmaS {index+1}', minimum=0, maximum=30, step=1, value=10, visible=False)
                                pp_color_tint_type_name = gr.Radio(label=f'Color tint type name {index+1}',choices=['warm', 'cool'], value='warm', visible=False)
                                pp_color_tint_lut_name = gr.Dropdown(label=f'Color tint lut name {index+1}',choices=lut_model_list(), value=lut_model_list()[0], visible=False)
                            pp_type_list.append(pp_type)
                            pp_saturation_strength_list.append(pp_saturation_strength)
                            pp_sharpening_radius_list.append(pp_sharpening_radius)
                            pp_sharpening_percent_list.append(pp_sharpening_percent)
                            pp_sharpening_threshold_list.append(pp_sharpening_threshold)
                            pp_gaussian_radius_list.append(pp_gaussian_radius)
                            pp_brightness_strength_list.append(pp_brightness_strength)
                            pp_color_strength_list.append(pp_color_strength)
                            pp_contrast_strength_list.append(pp_contrast_strength)
                            pp_hue_strength_list.append(pp_hue_strength)
                            pp_bilateral_sigmaC_list.append(pp_bilateral_sigmaC)
                            pp_bilateral_sigmaS_list.append(pp_bilateral_sigmaS)
                            pp_color_tint_type_name_list.append(pp_color_tint_type_name)
                            pp_color_tint_lut_name_list.append(pp_color_tint_lut_name)
                            def pp_type_change_func(pp_saturation_strength,pp_sharpening_radius,pp_sharpening_percent,pp_sharpening_threshold,pp_gaussian_radius,pp_brightness_strength,pp_color_strength,pp_contrast_strength,pp_hue_strength,pp_bilateral_sigmaC,pp_bilateral_sigmaS,pp_color_tint_type_name,pp_color_tint_lut_name):
                                saturation_strength, sharpening_radius, sharpening_percent, sharpening_threshold, gaussian_radius, brightness_strength, color_strength, contrast_strength, hue_strength, bilateral_sigmaC, bilateral_sigmaS, color_tint_type_name, color_tint_lut_name = pp_saturation_strength,pp_sharpening_radius,pp_sharpening_percent,pp_sharpening_threshold,pp_gaussian_radius,pp_brightness_strength,pp_color_strength,pp_contrast_strength,pp_hue_strength,pp_bilateral_sigmaC,pp_bilateral_sigmaS,pp_color_tint_type_name,pp_color_tint_lut_name
                                return lambda data:{
                                    saturation_strength:gr_show(data == 'saturation'),
                                    sharpening_radius:gr_show(data == 'sharpening'),
                                    sharpening_percent:gr_show(data == 'sharpening'),
                                    sharpening_threshold:gr_show(data == 'sharpening'),
                                    gaussian_radius:gr_show(data == 'gaussian blur'),
                                    brightness_strength:gr_show(data == 'brightness'),
                                    color_strength:gr_show(data == 'color'),
                                    contrast_strength:gr_show(data == 'contrast'),
                                    hue_strength:gr_show(data == 'hue'),
                                    bilateral_sigmaC:gr_show(data == 'bilateral'),
                                    bilateral_sigmaS:gr_show(data == 'bilateral'),
                                    color_tint_type_name:gr_show(data == 'color tint(type)'),
                                    color_tint_lut_name:gr_show(data == 'color tint(lut)')
                                }
                            def pp_type_change_func2(pp_saturation_strength,pp_sharpening_radius,pp_sharpening_percent,pp_sharpening_threshold,pp_gaussian_radius,pp_brightness_strength,pp_color_strength,pp_contrast_strength,pp_hue_strength,pp_bilateral_sigmaC,pp_bilateral_sigmaS,pp_color_tint_type_name,pp_color_tint_lut_name):
                                saturation_strength, sharpening_radius, sharpening_percent, sharpening_threshold, gaussian_radius, brightness_strength, color_strength, contrast_strength, hue_strength, bilateral_sigmaC, bilateral_sigmaS, color_tint_type_name, color_tint_lut_name = pp_saturation_strength,pp_sharpening_radius,pp_sharpening_percent,pp_sharpening_threshold,pp_gaussian_radius,pp_brightness_strength,pp_color_strength,pp_contrast_strength,pp_hue_strength,pp_bilateral_sigmaC,pp_bilateral_sigmaS,pp_color_tint_type_name,pp_color_tint_lut_name
                                return [saturation_strength, sharpening_radius, sharpening_percent, sharpening_threshold, gaussian_radius, brightness_strength, color_strength, contrast_strength, hue_strength, bilateral_sigmaC, bilateral_sigmaS, color_tint_type_name, color_tint_lut_name]
                            pp_type.change(
                                pp_type_change_func(pp_saturation_strength,pp_sharpening_radius,pp_sharpening_percent,pp_sharpening_threshold,pp_gaussian_radius,pp_brightness_strength,pp_color_strength,pp_contrast_strength,pp_hue_strength,pp_bilateral_sigmaC,pp_bilateral_sigmaS,pp_color_tint_type_name,pp_color_tint_lut_name),
                                inputs=[pp_type],
                                outputs=pp_type_change_func2(pp_saturation_strength,pp_sharpening_radius,pp_sharpening_percent,pp_sharpening_threshold,pp_gaussian_radius,pp_brightness_strength,pp_color_strength,pp_contrast_strength,pp_hue_strength,pp_bilateral_sigmaC,pp_bilateral_sigmaS,pp_color_tint_type_name,pp_color_tint_lut_name)
                            )
                        postprocess_tabs = postprocess_tabs_acc
                            
            with gr.Accordion("Watermark", open=False, elem_id='ddsd_watermark_option'):
                with gr.Column():
                    watermark_info = gr.HTML('<br><p style="margin-bottom:0.75em">Add a watermark to the final saved image</p>')
                    disable_watermark = gr.Checkbox(label='Disable Watermark', elem_id='disable_watermark',value=True, visible=True)
                    with gr.Tabs(elem_id='watermark_tabs', visible=False) as watermark_tabs_acc:
                        for index in range(shared.opts.data.get('watermark_count', 1)):
                            with gr.Tab(f'Watermark {index + 1} Argument', elem_id=f'watermark_{index+1}_argument_tab'):
                                watermark_type = gr.Radio(choices=['Text','Image'], value='Text', label=f'Watermark {index+1} text')
                                watermark_position = gr.Dropdown(choices=['Left','Left-Top','Top','Right-Top','Right','Right-Bottom','Bottom','Left-Bottom','Center'], value='Center', label=f'Watermark {index+1} Position', elem_id=f'watermark_{index+1}_position')
                                with gr.Column():
                                    watermark_image = gr.Image(label=f"Watermark {index+1} Upload image", visible=False)
                                    with gr.Row():
                                        watermark_image_size_width = gr.Slider(label=f'Watermark {index+1} Width', visible=False, minimum=50, maximum=500, step=10, value=100)
                                        watermark_image_size_height = gr.Slider(label=f'Watermark {index+1} Height', visible=False, minimum=50, maximum=500, step=10, value=100)    
                                with gr.Column():
                                    watermark_text = gr.Textbox(placeholder='watermark text - ex) Copyright © NeoGraph. All Rights Reserved.', visible=True, value='')
                                    with gr.Row():
                                        watermark_text_color = gr.ColorPicker(label=f'Watermark {index+1} Color')
                                        watermark_text_font = gr.Dropdown(label=f'Watermark {index+1} Fonts', choices=fonts_list, value=fonts_list[0])
                                        watermark_text_size = gr.Slider(label=f'Watermark {index+1} Size', visible=True, minimum=10, maximum=500, step=1, value=50)
                                watermark_padding = gr.Slider(label=f'Watermark {index+1} Padding', visible=True, minimum=0, maximum=200, step=1, value=10)
                                watermark_alpha = gr.Slider(label=f'Watermark {index+1} Alpha', visible=True, minimum=0, maximum=1, step=0.01, value=0.4)
                            watermark_type_list.append(watermark_type)
                            watermark_position_list.append(watermark_position)
                            watermark_image_list.append(watermark_image)
                            watermark_image_size_width_list.append(watermark_image_size_width)
                            watermark_image_size_height_list.append(watermark_image_size_height)
                            watermark_text_list.append(watermark_text)
                            watermark_text_color_list.append(watermark_text_color)
                            watermark_text_font_list.append(watermark_text_font)
                            watermark_text_size_list.append(watermark_text_size)
                            watermark_padding_list.append(watermark_padding)
                            watermark_alpha_list.append(watermark_alpha)
                            def watermark_type_change_func(watermark_image, watermark_image_size_width, watermark_image_size_height, watermark_text, watermark_text_color, watermark_text_font, watermark_text_size):
                                image, image_size_width, iamge_size_height, text, text_color, text_font, text_size = watermark_image, watermark_image_size_width, watermark_image_size_height, watermark_text, watermark_text_color, watermark_text_font, watermark_text_size
                                return lambda data:{
                                    image:gr_show(data == 'Image'),
                                    image_size_width:gr_show(data == 'Image'), 
                                    iamge_size_height:gr_show(data == 'Image'), 
                                    text:gr_show(data == 'Text'), 
                                    text_color:gr_show(data == 'Text'), 
                                    text_font:gr_show(data == 'Text'), 
                                    text_size:gr_show(data == 'Text')
                                }
                            def watermark_type_change_func2(watermark_image, watermark_image_size_width, watermark_image_size_height, watermark_text, watermark_text_color, watermark_text_font, watermark_text_size):
                                image, image_size_width, iamge_size_height, text, text_color, text_font, text_size = watermark_image, watermark_image_size_width, watermark_image_size_height, watermark_text, watermark_text_color, watermark_text_font, watermark_text_size
                                return [image, image_size_width, iamge_size_height, text, text_color, text_font, text_size]
                            watermark_type.change(
                                watermark_type_change_func(watermark_image,watermark_image_size_width,watermark_image_size_height,watermark_text,watermark_text_color,watermark_text_font,watermark_text_size),
                                inputs=[watermark_type],
                                outputs=watermark_type_change_func2(watermark_image, watermark_image_size_width, watermark_image_size_height, watermark_text, watermark_text_color, watermark_text_font, watermark_text_size)
                            )
                        watermark_tabs = watermark_tabs_acc
        disable_outpaint.change(
            lambda disable:{
                outpaint_sample:gr_show(not disable),
                outpaint_tabs:gr_show(not disable),
                outpaint_mask_blur:gr_show(not disable)
            },
            inputs=[disable_outpaint],
            outputs=[outpaint_sample, outpaint_tabs, outpaint_mask_blur]
        )
        disable_watermark.change(
            lambda disable:{
                watermark_tabs:gr_show(not disable)
            },
            inputs=[disable_watermark],
            outputs=watermark_tabs
        )
        disable_postprocess.change(
            lambda disable:{
                postprocess_tabs:gr_show(not disable)
            },
            inputs=[disable_postprocess],
            outputs=postprocess_tabs
        )
        disable_upscaler.change(
            lambda disable: {
                ddetailer_before_upscaler:gr_show(not disable),
                upscaler_sample:gr_show(not disable),
                upscaler_index:gr_show(not disable),
                upscaler_ckpt:gr_show(not disable),
                upscaler_vae:gr_show(not disable),
                scalevalue:gr_show(not disable),
                overlap:gr_show(not disable),
                rewidth:gr_show(not disable),
                reheight:gr_show(not disable),
                denoising_strength:gr_show(not disable),
            },
            inputs= [disable_upscaler],
            outputs =[ddetailer_before_upscaler, upscaler_sample, upscaler_index, upscaler_ckpt, upscaler_vae, scalevalue, overlap, rewidth, reheight, denoising_strength]
        )
        
        disable_mask_paint_mode.change(
            lambda disable:{
                inpaint_mask_mode:gr_show(is_img2img and not disable)
                },
            inputs=[disable_mask_paint_mode],
            outputs=inpaint_mask_mode
        )
        
        disable_detailer.change(
            lambda disable, in_disable:{
                disable_mask_paint_mode:gr_show(not disable and is_img2img),
                inpaint_mask_mode:gr_show(not disable and is_img2img and not in_disable),
                detailer_sample:gr_show(not disable),
                detailer_sam_model:gr_show(not disable),
                detailer_dino_model:gr_show(not disable),
                dino_full_res_inpaint:gr_show(not disable),
                dino_inpaint_padding:gr_show(not disable),
                detailer_mask_blur:gr_show(not disable),
                dino_tabs:gr_show(not disable)
            },
            inputs=[disable_detailer, disable_mask_paint_mode],
            outputs=[
                disable_mask_paint_mode,
                inpaint_mask_mode,
                detailer_sample,
                detailer_sam_model,
                detailer_dino_model,
                dino_full_res_inpaint,
                dino_inpaint_padding,
                detailer_mask_blur,
                dino_tabs
            ]
        )
        
        ret += [enable_script_names]
        ret += [disable_watermark, disable_postprocess]
        ret += [disable_upscaler, ddetailer_before_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength, upscaler_ckpt, upscaler_vae]
        ret += [disable_detailer, disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model, dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur]
        ret += [disable_outpaint, outpaint_sample, outpaint_mask_blur]
        ret += dino_detection_ckpt_list + \
                dino_detection_vae_list + \
                dino_detection_prompt_list + \
                dino_detection_positive_list + \
                dino_detection_negative_list + \
                dino_detection_denoise_list + \
                dino_detection_cfg_list + \
                dino_detection_steps_list + \
                dino_detection_spliter_disable_list + \
                dino_detection_spliter_remove_area_list + \
                dino_detection_clip_skip_list + \
                watermark_type_list + \
                watermark_position_list + \
                watermark_image_list + \
                watermark_image_size_width_list + \
                watermark_image_size_height_list + \
                watermark_text_list + \
                watermark_text_color_list + \
                watermark_text_font_list + \
                watermark_text_size_list + \
                watermark_padding_list + \
                watermark_alpha_list + \
                pp_type_list + \
                pp_saturation_strength_list + \
                pp_sharpening_radius_list + \
                pp_sharpening_percent_list + \
                pp_sharpening_threshold_list + \
                pp_gaussian_radius_list + \
                pp_brightness_strength_list + \
                pp_color_strength_list + \
                pp_contrast_strength_list + \
                pp_hue_strength_list + \
                pp_bilateral_sigmaC_list + \
                pp_bilateral_sigmaS_list + \
                pp_color_tint_type_name_list + \
                pp_color_tint_lut_name_list + \
                outpaint_positive_list + \
                outpaint_negative_list + \
                outpaint_denoise_list + \
                outpaint_cfg_list + \
                outpaint_steps_list + \
                outpaint_pixels_list + \
                outpaint_direction_list

        def ds(*args):
            args = list(args)
            ddsd_save_path = args[0]
            args = args[1:]
            enable_script_names,disable_watermark,disable_postprocess,disable_upscaler,ddetailer_before_upscaler,scalevalue,upscaler_sample,overlap,upscaler_index,rewidth,reheight,denoising_strength,upscaler_ckpt,upscaler_vae,disable_detailer,disable_mask_paint_mode,inpaint_mask_mode,detailer_sample,detailer_sam_model,detailer_dino_model,dino_full_res_inpaint,dino_inpaint_padding,detailer_mask_blur,disable_outpaint, outpaint_sample, outpaint_mask_blur = args[:26]
            result = {}
            result['enable_script_names'] = enable_script_names
            result['disable_watermark'] = disable_watermark
            result['disable_postprocess'] = disable_postprocess
            result['disable_upscaler'] = disable_upscaler
            result['ddetailer_before_upscaler'] = ddetailer_before_upscaler
            result['scalevalue'] = scalevalue
            result['upscaler_sample'] = upscaler_sample
            result['overlap'] = overlap
            result['upscaler_index'] = shared.sd_upscalers[upscaler_index].name
            result['rewidth'] = rewidth
            result['reheight'] = reheight
            result['denoising_strength'] = denoising_strength
            result['upscaler_ckpt'] = upscaler_ckpt
            result['upscaler_vae'] = upscaler_vae
            result['disable_detailer'] = disable_detailer
            result['disable_mask_paint_mode'] = disable_mask_paint_mode
            result['inpaint_mask_mode'] = inpaint_mask_mode
            result['detailer_sample'] = detailer_sample
            result['detailer_sam_model'] = detailer_sam_model
            result['detailer_dino_model'] = detailer_dino_model
            result['dino_full_res_inpaint'] = dino_full_res_inpaint
            result['dino_inpaint_padding'] = dino_inpaint_padding
            result['detailer_mask_blur'] = detailer_mask_blur
            result['disable_outpaint'] = disable_outpaint
            result['outpaint_sample'] = outpaint_sample
            result['outpaint_mask_blur'] = outpaint_mask_blur
            args = args[26:]
            result['dino_detect_count'] = shared.opts.data.get('dino_detect_count', 2)
            for index in range(result['dino_detect_count']):
                result[f'dino_detection_ckpt_{index+1}'] = args[index + result['dino_detect_count'] * 0]
                result[f'dino_detection_vae_{index+1}'] = args[index + result['dino_detect_count'] * 1]
                result[f'dino_detection_prompt_{index+1}'] = args[index + result['dino_detect_count'] * 2]
                result[f'dino_detection_positive_{index+1}'] = args[index + result['dino_detect_count'] * 3]
                result[f'dino_detection_negative_{index+1}'] = args[index + result['dino_detect_count'] * 4]
                result[f'dino_detection_denoise_{index+1}'] = args[index + result['dino_detect_count'] * 5]
                result[f'dino_detection_cfg_{index+1}'] = args[index + result['dino_detect_count'] * 6]
                result[f'dino_detection_steps_{index+1}'] = args[index + result['dino_detect_count'] * 7]
                result[f'dino_detection_spliter_disable_{index+1}'] = args[index + result['dino_detect_count'] * 8]
                result[f'dino_detection_spliter_remove_area_{index+1}'] = args[index + result['dino_detect_count'] * 9]
                result[f'dino_detection_clip_skip_{index+1}'] = args[index + result['dino_detect_count'] * 10]
            args = args[result['dino_detect_count'] * 11:]
            result['watermark_count'] = shared.opts.data.get('watermark_count', 1)
            for index in range(result['watermark_count']):
                result[f'watermark_type_{index+1}'] = args[index + result['watermark_count'] * 0]
                result[f'watermark_position_{index+1}'] = args[index + result['watermark_count'] * 1]
                result[f'watermark_image_{index+1}'] = None
                result[f'watermark_image_size_width_{index+1}'] = args[index + result['watermark_count'] * 3]
                result[f'watermark_image_size_height_{index+1}'] = args[index + result['watermark_count'] * 4]
                result[f'watermark_text_{index+1}'] = args[index + result['watermark_count'] * 5]
                result[f'watermark_text_color_{index+1}'] = args[index + result['watermark_count'] * 6]
                result[f'watermark_text_font_{index+1}'] = args[index + result['watermark_count'] * 7]
                result[f'watermark_text_size_{index+1}'] = args[index + result['watermark_count'] * 8]
                result[f'watermark_padding_{index+1}'] = args[index + result['watermark_count'] * 9]
                result[f'watermark_alpha_{index+1}'] = args[index + result['watermark_count'] * 10]
            args = args[result['watermark_count'] * 11:]
            result['postprocessing_count'] = shared.opts.data.get('postprocessing_count', 1)
            for index in range(result['postprocessing_count']):
                result[f'pp_type_{index+1}'] = args[index + result['postprocessing_count'] * 0]
                result[f'pp_saturation_strength_{index+1}'] = args[index + result['postprocessing_count'] * 1]
                result[f'pp_sharpening_radius_{index+1}'] = args[index + result['postprocessing_count'] * 2]
                result[f'pp_sharpening_percent_{index+1}'] = args[index + result['postprocessing_count'] * 3]
                result[f'pp_sharpening_threshold_{index+1}'] = args[index + result['postprocessing_count'] * 4]
                result[f'pp_gaussian_radius_{index+1}'] = args[index + result['postprocessing_count'] * 5]
                result[f'pp_brightness_strength_{index+1}'] = args[index + result['postprocessing_count'] * 6]
                result[f'pp_color_strength_{index+1}'] = args[index + result['postprocessing_count'] * 7]
                result[f'pp_contrast_strength_{index+1}'] = args[index + result['postprocessing_count'] * 8]
                result[f'pp_hue_strength_{index+1}'] = args[index + result['postprocessing_count'] * 9]
                result[f'pp_bilateral_sigmaC_{index+1}'] = args[index + result['postprocessing_count'] * 10]
                result[f'pp_bilateral_sigmaS_{index+1}'] = args[index + result['postprocessing_count'] * 11]
                result[f'pp_color_tint_type_name_{index+1}'] = args[index + result['postprocessing_count'] * 12]
                result[f'pp_color_tint_lut_name_{index+1}'] = args[index + result['postprocessing_count'] * 13]
            args = args[result['postprocessing_count'] * 13:]
            result['outpaint_count'] = shared.opts.data.get('outpaint_count', 1)
            for index in range(result['outpaint_count']):
                result[f'outpaint_positive_{index+1}'] = args[index + result['outpaint_count'] * 0]
                result[f'outpaint_negative_{index+1}'] = args[index + result['outpaint_count'] * 1]
                result[f'outpaint_denoise_{index+1}'] = args[index + result['outpaint_count'] * 2]
                result[f'outpaint_cfg_{index+1}'] = args[index + result['outpaint_count'] * 3]
                result[f'outpaint_steps_{index+1}'] = args[index + result['outpaint_count'] * 4]
                result[f'outpaint_pixels_{index+1}'] = args[index + result['outpaint_count'] * 5]
                result[f'outpaint_direction_{index+1}'] = args[index + result['outpaint_count'] * 6]
            args = args[result['outpaint_count'] * 6:]
            if not os.path.exists(ddsd_config_path):
                os.mkdir(ddsd_config_path)
            with open(os.path.join(ddsd_config_path, f'{ddsd_save_path}.ddcfg'), 'w', encoding='utf-8') as f:
                f.write(json_write(result))
            choices = [x[:-6] for x in os.listdir(ddsd_config_path) if x.endswith('.ddcfg')]
            return {
                ddsd_load_path:gr_list_refresh(choices, choices[0])
            }
        def dl(ddsd_load_path):
            with open(os.path.join(ddsd_config_path, f'{ddsd_load_path}.ddcfg'), 'r', encoding='utf-8') as f:
                result = json_read(f)
            results = [result['enable_script_names'],result['disable_watermark'],result['disable_postprocess'],result['disable_upscaler'],result['ddetailer_before_upscaler'],result['scalevalue'],result['upscaler_sample'],result['overlap'],result['upscaler_index'],result['rewidth'],result['reheight'],result['denoising_strength'],result['upscaler_ckpt'],result['upscaler_vae'],result['disable_detailer'],result['disable_mask_paint_mode'],result['inpaint_mask_mode'],result['detailer_sample'],result['detailer_sam_model'],result['detailer_dino_model'],result['dino_full_res_inpaint'],result['dino_inpaint_padding'],result['detailer_mask_blur'],result['disable_outpaint'],result['outpaint_sample'],result['outpaint_mask_blur']]
            def result_create(token,file_count,count, default):
                data = file_count if file_count < count else count
                temp = []
                for index in range(data):
                    temp.append(result.get(f'{token}_{index+1}',default))
                while len(temp) < count:
                    temp.append(default)
                return temp
            results += result_create('dino_detection_ckpt',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 'Original')
            results += result_create('dino_detection_vae',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 'Original')
            results += result_create('dino_detection_prompt',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), '')
            results += result_create('dino_detection_positive',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), '')
            results += result_create('dino_detection_negative',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), '')
            results += result_create('dino_detection_denoise',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 0.4)
            results += result_create('dino_detection_cfg',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 0)
            results += result_create('dino_detection_steps',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 0)
            results += result_create('dino_detection_spliter_disable',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), True)
            results += result_create('dino_detection_spliter_remove_area',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 8)
            results += result_create('dino_detection_clip_skip',result['dino_detect_count'], shared.opts.data.get('dino_detect_count', 2), 0)
            
            results += result_create('watermark_type',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 'Text')
            results += result_create('watermark_position',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 'Center')
            results += result_create('watermark_image',result['watermark_count'], shared.opts.data.get('watermark_count', 1), None)
            results += result_create('watermark_image_size_width',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 100)
            results += result_create('watermark_image_size_height',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 100)
            results += result_create('watermark_text',result['watermark_count'], shared.opts.data.get('watermark_count', 1), '')
            results += result_create('watermark_text_color',result['watermark_count'], shared.opts.data.get('watermark_count', 1), None)
            results += result_create('watermark_text_font',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 'Arial')
            results += result_create('watermark_text_size',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 50)
            results += result_create('watermark_padding',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 10)
            results += result_create('watermark_alpha',result['watermark_count'], shared.opts.data.get('watermark_count', 1), 0.4)
            
            results += result_create('pp_type',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 'none')
            results += result_create('pp_saturation_strength',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 1.1)
            results += result_create('pp_sharpening_radius',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 2)
            results += result_create('pp_sharpening_percent',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 100)
            results += result_create('pp_sharpening_threshold',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 1)
            results += result_create('pp_gaussian_radius',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 2)
            results += result_create('pp_brightness_strength',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 1.1)
            results += result_create('pp_color_strength',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 1.1)
            results += result_create('pp_contrast_strength',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 1.1)
            results += result_create('pp_hue_strength',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 0)
            results += result_create('pp_bilateral_sigmaC',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 10)
            results += result_create('pp_bilateral_sigmaS',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 10)
            results += result_create('pp_color_tint_type_name',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 'warm')
            results += result_create('pp_color_tint_lut_name',result['postprocessing_count'], shared.opts.data.get('postprocessing_count', 1), 'FGCineBasic.cube')
            
            results += result_create('outpaint_positive', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), '')
            results += result_create('outpaint_negative', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), '')
            results += result_create('outpaint_denoise', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), 0.8)
            results += result_create('outpaint_cfg', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), 0)
            results += result_create('outpaint_steps', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), 0)
            results += result_create('outpaint_pixels', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), 64)
            results += result_create('outpaint_direction', result['outpaint_count'], shared.opts.data.get('outpaint_count', 1), 'None')
            return dict(zip(ret, [gr_value_refresh(x) for x in results]))
        ddsd_save.click(ds, inputs=[ddsd_save_path]+ret, outputs=[ddsd_load_path])
        ddsd_load.click(dl, inputs=[ddsd_load_path], outputs=ret)
        
        return ret
    
    def outpainting(self, p, init_image, 
                    outpaint_sample, outpaint_mask_blur, outpaint_count,
                    outpaint_denoise_list, outpaint_cfg_list, outpaint_steps_list, outpaint_positive_list, outpaint_negative_list,
                    outpaint_pixels_list, outpaint_direction_list):
        
        for outpaint_index in range(outpaint_count):
            if outpaint_direction_list[outpaint_index] == 'None': continue
            is_horiz = outpaint_direction_list[outpaint_index] in ['Left','Right']
            is_vert = outpaint_direction_list[outpaint_index] in ['Up','Down']
            
            target_w = init_image.width + (outpaint_pixels_list[outpaint_index] if is_horiz else 0)
            target_h = init_image.height + (outpaint_pixels_list[outpaint_index] if is_vert else 0)
            
            image = Image.new('RGB', (target_w,target_h))
            image.paste(init_image, 
                        (outpaint_pixels_list[outpaint_index] if outpaint_direction_list[outpaint_index] == 'Left' else 0, 
                         outpaint_pixels_list[outpaint_index] if outpaint_direction_list[outpaint_index] == 'Up' else 0))
            mask = Image.new('L', (target_w, target_h), 'white')
            mask_draw = ImageDraw.Draw(mask)
            
            mask_draw.rectangle((
                (outpaint_pixels_list[outpaint_index] + outpaint_mask_blur * 2) if outpaint_direction_list[outpaint_index] == 'Left' else 0,
                (outpaint_pixels_list[outpaint_index] + outpaint_mask_blur * 2) if outpaint_direction_list[outpaint_index] == 'Up' else 0,
                (mask.width - outpaint_pixels_list[outpaint_index] - outpaint_mask_blur * 2) if outpaint_direction_list[outpaint_index] == 'Right' else target_w,
                (mask.height - outpaint_pixels_list[outpaint_index] - outpaint_mask_blur * 2) if outpaint_direction_list[outpaint_index] == 'Down' else target_h
                ), fill='black')
            
            latent_mask = Image.new('L', (target_w, target_h), 'white')
            latent_mask_draw = ImageDraw.Draw(latent_mask)
            latent_mask_draw.rectangle((
                (outpaint_pixels_list[outpaint_index] + outpaint_mask_blur // 2) if outpaint_direction_list[outpaint_index] == 'Left' else 0,
                (outpaint_pixels_list[outpaint_index] + outpaint_mask_blur // 2) if outpaint_direction_list[outpaint_index] == 'Up' else 0,
                (mask.width - outpaint_pixels_list[outpaint_index] - outpaint_mask_blur // 2) if outpaint_direction_list[outpaint_index] == 'Right' else target_w,
                (mask.height - outpaint_pixels_list[outpaint_index] - outpaint_mask_blur // 2) if outpaint_direction_list[outpaint_index] == 'Down' else target_h
            ), fill='black')
            
            devices.torch_gc()
            
            pi = I2I_Generator_Create(
                p, ('Euler' if p.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p.sampler_name) if outpaint_sample == 'Original' else outpaint_sample,
                outpaint_mask_blur * 2, False, 0, image,
                outpaint_denoise_list[outpaint_index],
                outpaint_cfg_list[outpaint_index] if outpaint_cfg_list[outpaint_index] > 0 else p.cfg_scale,
                outpaint_steps_list[outpaint_index] if outpaint_steps_list[outpaint_index] > 0 else p.steps,
                target_w, 
                target_h, 
                p.tiling, p.scripts, self.i2i_scripts, self.i2i_scripts_always, p.script_args,
                outpaint_positive_list[outpaint_index] if outpaint_positive_list[outpaint_index] else self.target_prompts,
                outpaint_negative_list[outpaint_index] if outpaint_negative_list[outpaint_index] else self.target_negative_prompts,
                0
            )
            
            
            pi.image_mask = mask
            pi.latent_mask = latent_mask
            pi.seed = self.target_seeds + outpaint_index
            
            state.job_count += 1
            proc = processing.process_images(pi)
            
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} Direction'] = outpaint_direction_list[outpaint_index]
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} Pixels'] = outpaint_pixels_list[outpaint_index]
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} Positive'] = proc.all_prompts[0] if outpaint_positive_list[outpaint_index] else "Original"
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} Negative'] = proc.all_negative_prompts[0] if outpaint_negative_list[outpaint_index] else "Original"
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} Denoising'] = pi.denoising_strength
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} CFG Scale'] = pi.cfg_scale
            p.extra_generation_params[f'Outpaint {outpaint_index + 1} Steps'] = pi.steps
            
            init_image = proc.images[0]
            
        return init_image
        
        
    
    def dino_detect_detailer(self, p, init_image,
                             disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model,
                             dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur,
                             dino_detect_count,
                             dino_detection_ckpt_list,
                             dino_detection_vae_list,
                             dino_detection_prompt_list,
                             dino_detection_positive_list,
                             dino_detection_negative_list,
                             dino_detection_denoise_list,
                             dino_detection_cfg_list,
                             dino_detection_steps_list,
                             dino_detection_spliter_disable_list,
                             dino_detection_spliter_remove_area_list,
                             dino_detection_clip_skip_list):
        self.image_results.append([])
        def mask_image_suffle(mask, image):
            if shared.opts.data.get('mask_type', False): return mask
            mask_image = Image.new("RGBA", mask.size, (255,255,255,0))
            mask_image.paste(mask, mask=mask)
            mask_image = Image.composite(mask, image, mask_image)
            return Image.blend(image, mask_image, 0.5)
        for detect_index in range(dino_detect_count):
            if len(dino_detection_prompt_list[detect_index]) < 1: continue
            pi = I2I_Generator_Create(
                p, ('Euler' if p.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p.sampler_name) if detailer_sample == 'Original' else detailer_sample,
                detailer_mask_blur, dino_full_res_inpaint, dino_inpaint_padding, init_image,
                dino_detection_denoise_list[detect_index],
                dino_detection_cfg_list[detect_index] if dino_detection_cfg_list[detect_index] > 0 else p.cfg_scale,
                dino_detection_steps_list[detect_index] if dino_detection_steps_list[detect_index] > 0 else p.steps,
                p.width, p.height, p.tiling, p.scripts, self.i2i_scripts, self.i2i_scripts_always, p.script_args,
                dino_detection_positive_list[detect_index] if dino_detection_positive_list[detect_index] else self.target_prompts,
                dino_detection_negative_list[detect_index] if dino_detection_negative_list[detect_index] else self.target_negative_prompts
            )
            mask = dino_detect_from_prompt(dino_detection_prompt_list[detect_index], detailer_sam_model, detailer_dino_model, init_image, disable_mask_paint_mode or isinstance(p, StableDiffusionProcessingTxt2Img), inpaint_mask_mode, getattr(p,'image_mask',None))
            if mask is not None:
                self.change_ckpt_model(dino_detection_ckpt_list[detect_index] if dino_detection_ckpt_list[detect_index] != 'Original' else self.ckptname)
                self.change_vae_model(dino_detection_vae_list[detect_index] if dino_detection_vae_list[detect_index] != 'Original' else self.vae)
                opts.CLIP_stop_at_last_layers = dino_detection_clip_skip_list[detect_index] if dino_detection_clip_skip_list[detect_index] else self.clip_skip
                if not dino_detection_spliter_disable_list[detect_index]:
                    mask = mask_spliter_and_remover(mask, dino_detection_spliter_remove_area_list[detect_index])
                    for mask_index, mask_split in enumerate(mask):
                        pi.seed = self.target_seeds + mask_index + detect_index
                        pi.init_images = [init_image]
                        pi.image_mask = Image.fromarray(mask_split)
                        if shared.opts.data.get('save_ddsd_working_on_dino_mask_images', False):
                            images.save_image(mask_image_suffle(pi.image_mask, pi.init_images[0]), p.outpath_samples, 
                                          shared.opts.data.get('save_ddsd_working_on_dino_mask_images_prefix', ''), 
                                          pi.seed, self.target_prompts, opts.samples_format, 
                                          suffix='' if shared.opts.data.get('save_ddsd_working_on_dino_mask_images_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_working_on_dino_mask_images_suffix', '')}",
                                          info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                        state.job_count += 1
                        if shared.opts.data.get('preview_masks_images', False):
                            shared.state.current_image = mask_image_suffle(pi.image_mask, pi.init_images[0])
                        if shared.opts.data.get('result_masks', False):
                            self.image_results[-1].append(mask_image_suffle(pi.image_mask, pi.init_images[0]))
                        processed = processing.process_images(pi)
                        init_image = processed.images[0]
                        if shared.opts.data.get('save_ddsd_working_on_images', False):
                            images.save_image(init_image, p.outpath_samples, 
                                            shared.opts.data.get('save_ddsd_working_on_images_prefix', ''), 
                                            pi.seed, self.target_prompts, opts.samples_format, 
                                            suffix='' if shared.opts.data.get('save_ddsd_working_on_images_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_working_on_images_suffix', '')}",
                                            info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                else:
                    pi.seed = self.target_seeds + detect_index
                    pi.init_images = [init_image]
                    pi.image_mask = Image.fromarray(mask)
                    if shared.opts.data.get('save_ddsd_working_on_dino_mask_images', False):
                        images.save_image(mask_image_suffle(pi.image_mask, pi.init_images[0]), p.outpath_samples, 
                                          shared.opts.data.get('save_ddsd_working_on_dino_mask_images_prefix', ''), 
                                          pi.seed, self.target_prompts, opts.samples_format, 
                                          suffix='' if shared.opts.data.get('save_ddsd_working_on_dino_mask_images_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_working_on_dino_mask_images_suffix', '')}",
                                          info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                    state.job_count += 1
                    if shared.opts.data.get('preview_masks_images', False):
                        shared.state.current_image = mask_image_suffle(pi.image_mask, pi.init_images[0])
                    if shared.opts.data.get('result_masks', False):
                        self.image_results[-1].append(mask_image_suffle(pi.image_mask, pi.init_images[0]))
                    processed = processing.process_images(pi)
                    init_image = processed.images[0]
                    if shared.opts.data.get('save_ddsd_working_on_images', False):
                        images.save_image(init_image, p.outpath_samples, 
                                          shared.opts.data.get('save_ddsd_working_on_images_prefix', ''), 
                                          pi.seed, self.target_prompts, opts.samples_format, 
                                          suffix='' if shared.opts.data.get('save_ddsd_working_on_images_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_working_on_images_suffix', '')}",
                                          info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                p.extra_generation_params[f'DINO {detect_index + 1}'] = dino_detection_prompt_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} Positive'] = processed.all_prompts[0] if dino_detection_positive_list[detect_index] else "Original"
                p.extra_generation_params[f'DINO {detect_index + 1} Negative'] = processed.all_negative_prompts[0] if dino_detection_negative_list[detect_index] else "Original"
                p.extra_generation_params[f'DINO {detect_index + 1} Denoising'] = pi.denoising_strength
                p.extra_generation_params[f'DINO {detect_index + 1} CFG Scale'] = pi.cfg_scale
                p.extra_generation_params[f'DINO {detect_index + 1} Steps'] = pi.steps
                p.extra_generation_params[f'DINO {detect_index + 1} Spliter'] = not dino_detection_spliter_disable_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} SplitRemove Area'] = dino_detection_spliter_remove_area_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} Ckpt Model'] = dino_detection_ckpt_list[detect_index] if dino_detection_ckpt_list[detect_index] != 'Original' else self.ckptname
                p.extra_generation_params[f'DINO {detect_index + 1} Vae Model'] = dino_detection_vae_list[detect_index] if dino_detection_vae_list[detect_index] != 'Original' else self.vae
                p.extra_generation_params[f'DINO {detect_index + 1} Clip Skip'] = dino_detection_clip_skip_list[detect_index] if dino_detection_clip_skip_list[detect_index] else 'Original'
            else:
                p.extra_generation_params[f'DINO {detect_index + 1}'] = 'Error'
        opts.CLIP_stop_at_last_layers = self.clip_skip
        return init_image
    
    def upscale(self, p, init_image, 
                scalevalue, upscaler_sample, overlap, rewidth, reheight, denoising_strength, upscaler_ckpt, upscaler_vae,
                detailer_mask_blur, dino_full_res_inpaint, dino_inpaint_padding):
        self.change_ckpt_model(upscaler_ckpt if upscaler_ckpt != 'Original' else self.ckptname)
        self.change_vae_model(upscaler_vae if upscaler_vae != 'Original' else self.vae)
        pi = I2I_Generator_Create(
                p, ('Euler' if p.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p.sampler_name) if upscaler_sample == 'Original' else upscaler_sample,
                detailer_mask_blur, dino_full_res_inpaint, dino_inpaint_padding, init_image,
                denoising_strength, p.cfg_scale, p.steps,
                rewidth, reheight, p.tiling, p.scripts, self.i2i_scripts, self.i2i_scripts_always, p.script_args,
                self.target_prompts, self.target_negative_prompts
            )
        p.extra_generation_params[f'Tile upscale value'] = scalevalue
        p.extra_generation_params[f'Tile upscale width'] = rewidth
        p.extra_generation_params[f'Tile upscale height'] = reheight
        p.extra_generation_params[f'Tile upscale overlap'] = overlap
        p.extra_generation_params[f'Tile upscale upscaler'] = self.upscaler.name
        p.extra_generation_params[f'Tile upscale Ckpt Model'] = upscaler_ckpt if upscaler_ckpt != 'Original' else self.ckptname
        p.extra_generation_params[f'Tile upscale Vae Model'] = upscaler_vae if upscaler_vae != 'Original' else self.vae
        if(self.upscaler.name != "None"): 
            img = self.upscaler.scaler.upscale(init_image, scalevalue, self.upscaler.data_path)
        else:
            img = init_image
        if rewidth and reheight:
            devices.torch_gc()
            grid = images.split_grid(img, tile_w=rewidth, tile_h=reheight, overlap=overlap)
            work = []
            for y, h, row in grid.tiles:
                for tiledata in row:
                    work.append(tiledata[2])

            batch_count = math.ceil(len(work))
            state.job = 'Upscaler Batching'
            state.job_count += batch_count

            print(f"Tile upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches (I2I).")
            
            pi.seed = self.target_seeds
            work_results = []
            for i in range(batch_count):
                pi.init_images = work[i:(i+1)]
                processed = processing.process_images(pi)

                p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for y, h, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (rewidth, reheight))
                    image_index += 1
            init_image = images.combine_grid(grid)
        else:
            init_image = img
        if shared.opts.data.get('save_ddsd_working_on_images', False):
            images.save_image(init_image, p.outpath_samples, 
                              shared.opts.data.get('save_ddsd_working_on_images_prefix', ''), 
                              pi.seed, self.target_prompts, opts.samples_format, 
                              suffix = '' if shared.opts.data.get('save_ddsd_working_on_images_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_working_on_images_suffix', '')}",
                              info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
        return init_image
    
    def watermark(self, p, init_image):
        if shared.opts.data.get('save_ddsd_watermark_with_and_without', False):
            images.save_image(init_image, p.outpath_samples, 
                              shared.opts.data.get('save_ddsd_watermark_with_and_without_prefix', ''), 
                              self.target_seeds, self.target_prompts, opts.samples_format, 
                              suffix= '' if shared.opts.data.get('save_ddsd_watermark_with_and_without_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_watermark_with_and_without_suffix', '')}",
                              info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
        for water_index in range(self.watermark_count):
            init_image = image_apply_watermark(init_image, 
                                                self.watermark_type_list[water_index],
                                                self.watermark_position_list[water_index],
                                                self.watermark_image_list[water_index],
                                                self.watermark_image_size_width_list[water_index],
                                                self.watermark_image_size_height_list[water_index],
                                                self.watermark_text_list[water_index],
                                                self.watermark_text_color_list[water_index],
                                                self.font_path[self.watermark_text_font_list[water_index]],
                                                self.watermark_text_size_list[water_index],
                                                self.watermark_padding_list[water_index],
                                                self.watermark_alpha_list[water_index])
        return init_image
    
    def postprocess_target(self, p, init_image, 
                           pp_type_list,
                           pp_saturation_strength_list,
                           pp_sharpening_radius_list, pp_sharpening_percent_list, pp_sharpening_threshold_list,
                           pp_gaussian_radius_list,
                           pp_brightness_strength_list,
                           pp_color_strength_list,
                           pp_contrast_strength_list,
                           pp_hue_strength_list,
                           pp_bilateral_sigmaC_list, pp_bilateral_sigmaS_list,
                           pp_color_tint_type_name_list,
                           pp_color_tint_lut_name_list):
        for pp_index in range(shared.opts.data.get('postprocessing_count', 1)):
            if pp_type_list[pp_index] == 'none': continue
            if shared.opts.data.get('save_ddsd_postprocessing_with_and_without', False):
                images.save_image(init_image, p.outpath_samples, 
                                shared.opts.data.get('save_ddsd_postprocessing_with_and_without_prefix', ''), 
                                self.target_seeds, self.target_prompts, opts.samples_format, 
                                suffix= '' if shared.opts.data.get('save_ddsd_postprocessing_with_and_without_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_postprocessing_with_and_without_suffix', '')}",
                                info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
            init_image = ddsd_postprocess(init_image, pp_type_list[pp_index], pp_saturation_strength_list[pp_index], pp_sharpening_radius_list[pp_index], pp_sharpening_percent_list[pp_index], pp_sharpening_threshold_list[pp_index], pp_gaussian_radius_list[pp_index], pp_brightness_strength_list[pp_index], pp_color_strength_list[pp_index], pp_contrast_strength_list[pp_index], pp_hue_strength_list[pp_index], pp_bilateral_sigmaC_list[pp_index], pp_bilateral_sigmaS_list[pp_index], pp_color_tint_lut_name_list[pp_index], pp_color_tint_type_name_list[pp_index])
            p.extra_generation_params[f'Postprocess {pp_index+1} type'] = pp_type_list[pp_index]
            if pp_type_list[pp_index] == 'saturation': 
                p.extra_generation_params[f'Postprocess {pp_index+1} strength'] = pp_saturation_strength_list[pp_index]
            elif pp_type_list[pp_index] == 'sharpening': 
                p.extra_generation_params[f'Postprocess {pp_index+1} radius'] = pp_sharpening_radius_list[pp_index]
                p.extra_generation_params[f'Postprocess {pp_index+1} percent'] = pp_sharpening_percent_list[pp_index]
                p.extra_generation_params[f'Postprocess {pp_index+1} threshold'] = pp_sharpening_threshold_list[pp_index]
            elif pp_type_list[pp_index] == 'gaussian blur': 
                p.extra_generation_params[f'Postprocess {pp_index+1} radius'] = pp_gaussian_radius_list[pp_index]
            elif pp_type_list[pp_index] == 'brightness': 
                p.extra_generation_params[f'Postprocess {pp_index+1} strength'] = pp_brightness_strength_list[pp_index]
            elif pp_type_list[pp_index] == 'color': 
                p.extra_generation_params[f'Postprocess {pp_index+1} strength'] = pp_color_strength_list[pp_index]
            elif pp_type_list[pp_index] == 'contrast': 
                p.extra_generation_params[f'Postprocess {pp_index+1} strength'] = pp_contrast_strength_list[pp_index]
            elif pp_type_list[pp_index] == 'hue': 
                p.extra_generation_params[f'Postprocess {pp_index+1} strength'] = pp_hue_strength_list[pp_index]
            elif pp_type_list[pp_index] == 'bilateral': 
                p.extra_generation_params[f'Postprocess {pp_index+1} sigma c'] = pp_bilateral_sigmaC_list[pp_index]
                p.extra_generation_params[f'Postprocess {pp_index+1} sigma s'] = pp_bilateral_sigmaS_list[pp_index]
            elif pp_type_list[pp_index] == 'color tint(type)': 
                p.extra_generation_params[f'Postprocess {pp_index+1} type'] = pp_color_tint_type_name_list[pp_index]
            elif pp_type_list[pp_index] == 'color tint(lut)': 
                p.extra_generation_params[f'Postprocess {pp_index+1} lut'] = pp_color_tint_lut_name_list[pp_index]
        return init_image

    def change_vae_model(self, name:str):
        if name is None: return
        if name.lower() in ['auto', 'automatic']: modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=modules.sd_vae.unspecified)
        elif name.lower() == 'none': modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=None)
        else: modules.sd_vae.reload_vae_weights(shared.sd_model, vae_file=modules.sd_vae.vae_dict[name])
    
    def change_ckpt_model(self, name:str):
        if name is None: return
        info = modules.sd_models.get_closet_checkpoint_match(name)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {name}")
        modules.sd_models.reload_model_weights(shared.sd_model, info)
    
    def postprocess(self, p, res, *args, **kargs):
        if getattr(p, 'sub_processing', False): return
        self.change_ckpt_model(self.ckptname)
        self.change_vae_model(self.vae)
        opts.CLIP_stop_at_last_layers = self.clip_skip
        if len(self.image_results) < 1: return
        final_count = len(res.images)
        if (p.n_iter > 1 or p.batch_size > 1) and final_count != p.n_iter * p.batch_size:
            grid = res.images[0]
            res.images = res.images[1:]
            grid_texts = res.infotexts[0]
            res.infotexts = res.infotexts[1:]
        images = [[*masks, image] for masks, image in zip(self.image_results,res.images)]
        res.images = [image for sub in images for image in sub]
        infos = [[info] * (len(masks) + 1) for masks, info in zip(self.image_results, res.infotexts)]
        res.infotexts = [info for sub in infos for info in sub]
        if (p.n_iter > 1 or p.batch_size > 1) and final_count != p.n_iter * p.batch_size:
            res.images = [grid] + res.images
            res.infotexts = [grid_texts] + res.infotexts
    
    def process(self, p,
            enable_script_names,
            disable_watermark, disable_postprocess,
            disable_upscaler, ddetailer_before_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength, upscaler_ckpt, upscaler_vae,
            disable_detailer, disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model,
            dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur,
            disable_outpaint, outpaint_sample, outpaint_mask_blur,
            *args):
        if getattr(p, 'sub_processing', False): return
        self.image_results = []
        self.ckptname = shared.opts.data['sd_model_checkpoint']
        self.vae = shared.opts.data['sd_vae']
        self.clip_skip = opts.CLIP_stop_at_last_layers
        self.restore_script(p)
        self.enable_script_names = enable_script_names
        self.disable_watermark = disable_watermark
        self.disable_postprocess = disable_postprocess
        self.disable_upscaler = disable_upscaler
        self.ddetailer_before_upscaler = ddetailer_before_upscaler
        self.scalevalue = scalevalue
        self.upscaler_sample = upscaler_sample
        self.overlap = overlap
        self.upscaler_index = upscaler_index
        self.rewidth = rewidth
        self.reheight = reheight
        self.denoising_strength = denoising_strength
        self.upscaler_ckpt = upscaler_ckpt
        self.upscaler_vae = upscaler_vae
        self.disable_detailer = disable_detailer
        self.disable_mask_paint_mode = disable_mask_paint_mode
        self.inpaint_mask_mode = inpaint_mask_mode
        self.detailer_sample = detailer_sample
        self.detailer_sam_model = detailer_sam_model
        self.detailer_dino_model = detailer_dino_model
        self.dino_full_res_inpaint = dino_full_res_inpaint
        self.dino_inpaint_padding = dino_inpaint_padding
        self.detailer_mask_blur = detailer_mask_blur
        self.disable_outpaint = disable_outpaint
        self.outpaint_sample = outpaint_sample
        self.outpaint_mask_blur = outpaint_mask_blur
        args_list = [*args]
        self.dino_detect_count = shared.opts.data.get('dino_detect_count', 2)
        self.dino_detection_ckpt_list = args_list[self.dino_detect_count * 0:self.dino_detect_count * 1]
        self.dino_detection_vae_list = args_list[self.dino_detect_count * 1:self.dino_detect_count * 2]
        self.dino_detection_prompt_list = args_list[self.dino_detect_count * 2:self.dino_detect_count * 3]
        self.dino_detection_positive_list = args_list[self.dino_detect_count * 3:self.dino_detect_count * 4] 
        self.dino_detection_negative_list = args_list[self.dino_detect_count * 4:self.dino_detect_count * 5]
        self.dino_detection_denoise_list = args_list[self.dino_detect_count * 5:self.dino_detect_count * 6]
        self.dino_detection_cfg_list = args_list[self.dino_detect_count * 6:self.dino_detect_count * 7]
        self.dino_detection_steps_list = args_list[self.dino_detect_count * 7:self.dino_detect_count * 8]
        self.dino_detection_spliter_disable_list = args_list[self.dino_detect_count * 8:self.dino_detect_count * 9]
        self.dino_detection_spliter_remove_area_list = args_list[self.dino_detect_count * 9:self.dino_detect_count * 10]
        self.dino_detection_clip_skip_list = args_list[self.dino_detect_count * 10 : self.dino_detect_count * 11]
        self.watermark_count = shared.opts.data.get('watermark_count', 1)
        self.watermark_type_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 0:self.dino_detect_count * 11 + self.watermark_count * 1]
        self.watermark_position_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 1:self.dino_detect_count * 11 + self.watermark_count * 2]
        self.watermark_image_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 2:self.dino_detect_count * 11 + self.watermark_count * 3]
        self.watermark_image_size_width_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 3:self.dino_detect_count * 11 + self.watermark_count * 4]
        self.watermark_image_size_height_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 4:self.dino_detect_count * 11 + self.watermark_count * 5]
        self.watermark_text_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 5:self.dino_detect_count * 11 + self.watermark_count * 6]
        self.watermark_text_color_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 6:self.dino_detect_count * 11 + self.watermark_count * 7]
        self.watermark_text_font_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 7:self.dino_detect_count * 11 + self.watermark_count * 8]
        self.watermark_text_size_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 8:self.dino_detect_count * 11 + self.watermark_count * 9]
        self.watermark_padding_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 9:self.dino_detect_count * 11 + self.watermark_count * 10]
        self.watermark_alpha_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 10:self.dino_detect_count * 11 + self.watermark_count * 11]
        self.pp_count = shared.opts.data.get('postprocessing_count', 1)
        self.pp_type_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 0:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 1]
        self.pp_saturation_strength_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 1:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 2]
        self.pp_sharpening_radius_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 2:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 3]
        self.pp_sharpening_percent_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 3:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 4]
        self.pp_sharpening_threshold_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 4:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 5]
        self.pp_gaussian_radius_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 5:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 6]
        self.pp_brightness_strength_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 6:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 7]
        self.pp_color_strength_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 7:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 8]
        self.pp_contrast_strength_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 8:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 9]
        self.pp_hue_strength_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 9:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 10]
        self.pp_bilateral_sigmaC_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 10:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 11]
        self.pp_bilateral_sigmaS_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 11:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 12]
        self.pp_color_tint_type_name_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 12:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 13]
        self.pp_color_tint_lut_name_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 13:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14]
        self.outpaint_count = shared.opts.data.get('outpaint_count', 1)
        self.outpaint_positive_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 0:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 1]
        self.outpaint_negative_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 1:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 2]
        self.outpaint_denoise_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 2:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 3]
        self.outpaint_cfg_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 3:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 4]
        self.outpaint_steps_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 4:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 5]
        self.outpaint_pixels_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 5:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 6]
        self.outpaint_direction_list = args_list[self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 6:self.dino_detect_count * 11 + self.watermark_count * 11 + self.pp_count * 14 + self.outpaint_count * 7]
        self.script_names_list = [x.strip()+'.py' for x in enable_script_names.split(';') if len(x) > 1]
        self.script_names_list += [os.path.basename(__file__)]
        self.i2i_scripts = [x for x in self.original_scripts if os.path.basename(x.filename) in self.script_names_list].copy()
        self.i2i_scripts_always = [x for x in self.original_scripts_always if os.path.basename(x.filename) in self.script_names_list].copy()
        self.upscaler = shared.sd_upscalers[upscaler_index]
    
    def before_process_batch(self, p, *args, **kargs):
        if getattr(p, 'sub_processing', False): return
        self.iter_number = kargs['batch_number']
        self.batch_number = 0
    
    def restore_script(self, p):
        if self.original_scripts is None: self.original_scripts = p.scripts.scripts.copy()
        else: 
            if len(p.scripts.scripts) != len(self.original_scripts): p.scripts.scripts = self.original_scripts.copy()
        if self.original_scripts_always is None: self.original_scripts_always = p.scripts.alwayson_scripts.copy()
        else: 
            if len(p.scripts.alwayson_scripts) != len(self.original_scripts_always): p.scripts.alwayson_scripts = self.original_scripts_always.copy()
        p.scripts.scripts = self.original_scripts.copy()
        p.scripts.alwayson_scripts = self.original_scripts_always.copy()
    
    def postprocess_image(self, p, pp, *args):
        if getattr(p, 'sub_processing', False): return
        devices.torch_gc()
        output_image = pp.image
        self.target_prompts = p.all_prompts[self.iter_number * p.batch_size:(self.iter_number + 1) * p.batch_size][self.batch_number]
        self.target_negative_prompts = p.all_negative_prompts[self.iter_number * p.batch_size:(self.iter_number + 1) * p.batch_size][self.batch_number]
        self.target_seeds = p.all_seeds[self.iter_number * p.batch_size:(self.iter_number + 1) * p.batch_size][self.batch_number]
        if shared.opts.data.get('save_ddsd_working_on_images', False):
            images.save_image(output_image, p.outpath_samples, 
                              shared.opts.data.get('save_ddsd_working_on_images_prefix', ''), 
                              self.target_seeds, self.target_prompts, opts.samples_format, 
                              suffix= '' if shared.opts.data.get('save_ddsd_working_on_images_suffix', '') == '' else f"-{shared.opts.data.get('save_ddsd_working_on_images_suffix', '')}",
                              info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
        
        if not self.disable_outpaint:
            output_image = self.outpainting(p, output_image,
                                            self.outpaint_sample, self.outpaint_mask_blur, self.outpaint_count, 
                                            self.outpaint_denoise_list, self.outpaint_cfg_list, self.outpaint_steps_list, 
                                            self.outpaint_positive_list, self.outpaint_negative_list, self.outpaint_pixels_list,self.outpaint_direction_list)
        devices.torch_gc()
        
        if self.ddetailer_before_upscaler and not self.disable_upscaler:
            output_image = self.upscale(p, output_image,
                                        self.scalevalue, self.upscaler_sample, 
                                        self.overlap, self.rewidth, self.reheight, self.denoising_strength,
                                        self.upscaler_ckpt, self.upscaler_vae,
                                        self.detailer_mask_blur, self.dino_full_res_inpaint, self.dino_inpaint_padding)
        devices.torch_gc()
        
        if not self.disable_detailer:
            output_image = self.dino_detect_detailer(p, output_image, 
                                                     self.disable_mask_paint_mode, self.inpaint_mask_mode, self.detailer_sample, self.detailer_sam_model, self.detailer_dino_model, 
                                                     self.dino_full_res_inpaint, self.dino_inpaint_padding, self.detailer_mask_blur,
                                                     self.dino_detect_count,
                                                     self.dino_detection_ckpt_list,
                                                     self.dino_detection_vae_list,
                                                     self.dino_detection_prompt_list,
                                                     self.dino_detection_positive_list,
                                                     self.dino_detection_negative_list,
                                                     self.dino_detection_denoise_list,
                                                     self.dino_detection_cfg_list,
                                                     self.dino_detection_steps_list,
                                                     self.dino_detection_spliter_disable_list,
                                                     self.dino_detection_spliter_remove_area_list,
                                                     self.dino_detection_clip_skip_list)
        devices.torch_gc()
        
        if not self.ddetailer_before_upscaler and not self.disable_upscaler:
            output_image = self.upscale(p, output_image,
                                        self.scalevalue, self.upscaler_sample, 
                                        self.overlap, self.rewidth, self.reheight, self.denoising_strength,
                                        self.upscaler_ckpt, self.upscaler_vae,
                                        self.detailer_mask_blur, self.dino_full_res_inpaint, self.dino_inpaint_padding)
        devices.torch_gc()
        
        if not self.disable_postprocess:
            output_image = self.postprocess_target(p, output_image,
                                                   self.pp_type_list,
                                                   self.pp_saturation_strength_list,
                                                   self.pp_sharpening_radius_list,
                                                   self.pp_sharpening_percent_list,
                                                   self.pp_sharpening_threshold_list,
                                                   self.pp_gaussian_radius_list,
                                                   self.pp_brightness_strength_list,
                                                   self.pp_color_strength_list,
                                                   self.pp_contrast_strength_list,
                                                   self.pp_hue_strength_list,
                                                   self.pp_bilateral_sigmaC_list,
                                                   self.pp_bilateral_sigmaS_list,
                                                   self.pp_color_tint_type_name_list,
                                                   self.pp_color_tint_lut_name_list)
        
        devices.torch_gc()
        
        if not self.disable_watermark:
            output_image = self.watermark(p, output_image)
        
        devices.torch_gc()
        self.batch_number += 1
        self.restore_script(p)
        pp.image = output_image

def on_ui_settings():
    section = ('ddsd_script', "DDSD")
    shared.opts.add_option("save_ddsd_working_on_images", shared.OptionInfo(
        False, "Save all images you are working on", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_working_on_images_prefix", shared.OptionInfo(
        '', "Save all images you are working on prefix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_working_on_images_suffix", shared.OptionInfo(
        'Working_On', "Save all images you are working on suffix", gr.Textbox, {"interactive": True}, section=section))
    
    shared.opts.add_option("outpaint_count", shared.OptionInfo(
        1, "Outpainting Max Count", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))
        
    shared.opts.add_option("save_ddsd_working_on_dino_mask_images", shared.OptionInfo(
        False, "Save dino mask images you are working on", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_working_on_dino_mask_images_prefix", shared.OptionInfo(
        '', "Save dino mask images you are working on prefix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_working_on_dino_mask_images_suffix", shared.OptionInfo(
        'Mask', "Save dino mask images you are working on suffix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("dino_detect_count", shared.OptionInfo(
        2, "Dino Detect Max Count", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))
    
    shared.opts.add_option("save_ddsd_postprocessing_with_and_without", shared.OptionInfo(
        False, "Save with and without postprocessing ", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_postprocessing_with_and_without_prefix", shared.OptionInfo(
        '', "Save with and without postprocesing prefix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_postprocessing_with_and_without_suffix", shared.OptionInfo(
        'Without_Postprocessing', "Save with and without postprocessing suffix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("postprocessing_count", shared.OptionInfo(
        1, "Postprocessing Count", gr.Slider, {"minimum": 1, "maximum": 5, "step": 1}, section=section))
    
    shared.opts.add_option("save_ddsd_watermark_with_and_without", shared.OptionInfo(
        False, "Save with and without watermark ", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_watermark_with_and_without_prefix", shared.OptionInfo(
        '', "Save with and without watermark prefix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_watermark_with_and_without_suffix", shared.OptionInfo(
        'Without_Watermark', "Save with and without watermark suffix", gr.Textbox, {"interactive": True}, section=section))
    shared.opts.add_option("watermark_count", shared.OptionInfo(
        1, "Watermark Count", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))
    
    shared.opts.add_option("preview_masks_images", shared.OptionInfo(
        False, "Show the working mask in preview.", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("result_masks", shared.OptionInfo(
        False, "The mask result is output on the final output.", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("mask_type", shared.OptionInfo(
        False, "The type of mask is a black and white image.", gr.Checkbox, {"interactive": True}, section=section))

modules.script_callbacks.on_ui_settings(on_ui_settings)