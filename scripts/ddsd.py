import os
import math
import copy
from random import choice
from glob import glob

import gradio as gr
import numpy as np
from PIL import Image

from scripts.sam import sam_model_list
from scripts.dino import dino_model_list
from scripts.ddsd_utils import dino_detect_from_prompt, mask_spliter_and_remover, I2I_Generator_Create, get_fonts_list, image_apply_watermark

import modules
from modules import processing, shared, images, devices, modelloader
from modules.processing import StableDiffusionProcessingImg2Img, create_infotext
from modules.shared import opts, state
from modules.sd_models import model_hash
from modules.paths import models_path
from modules.scripts import AlwaysVisible

from basicsr.utils.download_util import load_file_from_url

dd_models_path = os.path.join(models_path, "mmdet")
grounding_models_path = os.path.join(models_path, "grounding")
sam_models_path = os.path.join(models_path, "sam")


def list_models(model_path):
        model_list = modelloader.load_models(model_path=model_path, ext_filter=[".pth"])
        
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
    if (len(list_models(grounding_models_path)) == 0):
        print("No detection groundingdino models found, downloading...")
        load_file_from_url('https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth',grounding_models_path)
        load_file_from_url('https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py',grounding_models_path, file_name='groundingdino_swint_ogc.py')
        #load_file_from_url('https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth',grounding_models_path)
        #load_file_from_url('https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB.cfg.py',grounding_models_path, file_name='groundingdino_swinb_cogcoor.py')
        
        
    if (len(list_models(sam_models_path)) == 0):
        print("No detection sam models found, downloading...")
        #load_file_from_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',sam_models_path)
        #load_file_from_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',sam_models_path)
        load_file_from_url('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',sam_models_path)

startup()

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

class Script(modules.scripts.Script):
    def __init__(self):
        self.original_scripts = None
        self.original_scripts_always = None
        _ ,self.font_path = get_fonts_list()
        
    def title(self):
        return "ddetailer + sdupscale"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        sample_list = [x.name for x in shared.list_samplers()]
        sample_list = [x for x in sample_list if x not in ['PLMS','UniPC','DDIM']]
        sample_list.insert(0,"Original")
        fonts_list, _ = get_fonts_list()
        ret = []
        dino_detection_prompt_list = []
        dino_detection_positive_list = []
        dino_detection_negative_list = []
        dino_detection_denoise_list = []
        dino_detection_cfg_list = []
        dino_detection_steps_list = []
        dino_detection_spliter_disable_list = []
        dino_detection_spliter_remove_area_list = []
        watermark_type_list = []
        watermark_position_list = []
        watermark_image_gr_list = []
        watermark_text_gr_list = []
        watermark_image_list = []
        watermark_image_size_width_list = []
        watermark_image_size_height_list = []
        watermark_text_list = []
        watermark_text_color_list = []
        watermark_text_font_list = []
        watermark_text_size_list = []
        watermark_padding_list = []
        watermark_alpha_list = []
        dino_tabs = None
        watermark_tabs = None
        
        with gr.Accordion('DDSD', open=False, elem_id='ddsd_all_option_acc'):
        
            with gr.Accordion("Script Option", open = False, elem_id="ddsd_enable_script_acc"):
                with gr.Column():
                    all_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I All process target script</p>')
                    enable_script_names = gr.Textbox(label="Enable Script(Extension)", elem_id="enable_script_names", value='dynamic_thresholding;dynamic_prompting',show_label=True, lines=1, placeholder="Extension python file name(ex - dynamic_thresholding;dynamic_prompting)")
        
            with gr.Accordion("Upscaler", open=False, elem_id="ddsd_upsacler_acc"):
                with gr.Column():
                    sd_upscale_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Upscaler Option</p>')
                    disable_upscaler = gr.Checkbox(label='Disable Upscaler', elem_id='disable_upscaler', value=True, visible=True)
                    ddetailer_before_upscaler = gr.Checkbox(label='Upscaler before running detailer', elem_id='upscaler_before_running_detailer', value=False, visible=False)
                    with gr.Row():
                        upscaler_sample = gr.Dropdown(label='Upscaler Sampling', elem_id='upscaler_sample', choices=sample_list, value=sample_list[0], visible=False, type="value")
                        upscaler_index = gr.Dropdown(label='Upscaler', elem_id='upscaler_index', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[-1].name, type="index", visible=False)
                    scalevalue = gr.Slider(minimum=1, maximum=16, step=0.5, elem_id='upscaler_scalevalue', label='Resize', value=2, visible=False)
                    overlap = gr.Slider(minimum=0, maximum=256, step=32, elem_id='upscaler_overlap', label='Tile overlap', value=32, visible=False)
                    with gr.Row():
                        rewidth = gr.Slider(minimum=0, maximum=1024, step=64, elem_id='upscaler_rewidth', label='Width', value=512, visible=False)
                        reheight = gr.Slider(minimum=0, maximum=1024, step=64, elem_id='upscaler_reheight', label='Height', value=512, visible=False)
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
                        detailer_dino_model = gr.Dropdown(label='Deteiler DINO Model', elem_id='detailer_dino_model', choices=dino_model_list(), value=dino_model_list()[0], visible=False)
                    with gr.Tabs(elem_id = 'dino_detct_arguments', visible=False) as dino_tabs_acc:
                        for index in range(shared.opts.data.get('dino_detect_count', 2)):
                            with gr.Tab(f'DINO {index + 1} Argument', elem_id=f'dino_{index + 1}_argument_tab'):
                                dino_detection_prompt = gr.Textbox(label=f"Detect {index + 1} Prompt", elem_id=f"detailer_detect_prompt_{index + 1}", show_label=True, lines=2, placeholder="Detect Token Prompt(ex - face:level(0-2):threshold(0-1):dilation(0-128))", visible=True)
                                with gr.Row():
                                    dino_detection_positive = gr.Textbox(label=f"Positive {index + 1} Prompt", elem_id=f"detailer_detect_positive_{index + 1}", show_label=True, lines=2, placeholder="Detect Mask Inpaint Positive(ex - perfect anatomy)", visible=True)
                                    dino_detection_negative = gr.Textbox(label=f"Negative {index + 1} Prompt", elem_id=f"detailer_detect_negative_{index + 1}", show_label=True, lines=2, placeholder="Detect Mask Inpaint Negative(ex - nsfw)", visible=True)
                                dino_detection_denoise = gr.Slider(minimum=0, maximum=1.0, step=0.01, elem_id=f'dino_detect_{index+1}_denoising', label=f'DINO {index + 1} Denoising strength', value=0.4, visible=True)
                                dino_detection_cfg = gr.Slider(minimum=0, maximum=500, step=0.5, elem_id=f'dino_detect_{index+1}_cfg_scale', label=f'DINO  {index + 1} CFG Scale(0 to Origin)', value=0, visible=True)
                                dino_detection_steps = gr.Slider(minimum=0, maximum=150, step=1, elem_id=f'dino_detect_{index+1}_steps', label=f'DINO {index + 1} Steps(0 to Origin)', value=0, visible=True)
                                dino_detection_spliter_disable = gr.Checkbox(label=f'Disable DINO {index + 1} Detect Split Mask', value=True, visible=True)
                                dino_detection_spliter_remove_area = gr.Slider(minimum=0, maximum=800, step=8, elem_id=f'dino_detect_{index+1}_remove_area', label=f'Remove {index + 1} Area', value=16, visible=True)
                                dino_detection_prompt_list.append(dino_detection_prompt)
                                dino_detection_positive_list.append(dino_detection_positive)
                                dino_detection_negative_list.append(dino_detection_negative)
                                dino_detection_denoise_list.append(dino_detection_denoise)
                                dino_detection_cfg_list.append(dino_detection_cfg)
                                dino_detection_steps_list.append(dino_detection_steps)
                                dino_detection_spliter_disable_list.append(dino_detection_spliter_disable)
                                dino_detection_spliter_remove_area_list.append(dino_detection_spliter_remove_area)
                        dino_tabs = dino_tabs_acc
                    dino_full_res_inpaint = gr.Checkbox(label='Inpaint at full resolution ', elem_id='detailer_full_res', value=True, visible = False)
                    with gr.Row():
                        dino_inpaint_padding = gr.Slider(label='Inpaint at full resolution padding, pixels ', elem_id='detailer_padding', minimum=0, maximum=256, step=4, value=32, visible=False)
                        detailer_mask_blur = gr.Slider(label='Detailer Blur', elem_id='detailer_mask_blur', minimum=0, maximum=64, step=1, value=4, visible=False)
                        
            with gr.Accordion("Watermark", open=False, elem_id='ddsd_watermark_option'):
                with gr.Column():
                    watermark_info = gr.HTML('<br><p style="margin-bottom:0.75em">Add a watermark to the final saved image</p>')
                    disable_watermark = gr.Checkbox(label='Disable Watermark', elem_id='disable_watermark',value=True, visible=True)
                    with gr.Tabs(elem_id='watermark_tabs', visible=False) as watermark_tabs_acc:
                        for index in range(shared.opts.data.get('watermark_count', 1)):
                            with gr.Tab(f'Watermark {index + 1} Argument', elem_id=f'watermark_{index+1}_argument_tab'):
                                watermark_type = gr.Radio(choices=['Text','Image'], value='Text', label=f'Watermark {index+1} text')
                                watermark_position = gr.Dropdown(choices=['Left','Left-Top','Top','Right-Top','Right','Right-Bottom','Bottom','Left-Bottom','Center'], value='Center', label=f'Watermark {index+1} Position', elem_id=f'watermark_{index+1}_position')
                                with gr.Column(visible=False) as watermark_image_gr:
                                    watermark_image = gr.Image(label=f"Watermark {index+1} Upload image", visible=True)
                                    with gr.Row():
                                        watermark_image_size_width = gr.Slider(label=f'Watermark {index+1} Width', visible=True, minimum=50, maximum=500, step=10, value=100)
                                        watermark_image_size_height = gr.Slider(label=f'Watermark {index+1} Height', visible=True, minimum=50, maximum=500, step=10, value=100)    
                                    watermark_image_gr_list.append(watermark_image_gr)
                                with gr.Column(visible=True) as watermark_text_gr:
                                    watermark_text = gr.Textbox(placeholder='watermark text - ex) Copyright © NeoGraph. All Rights Reserved.', visible=True, value='')
                                    with gr.Row():
                                        watermark_text_color = gr.ColorPicker(label=f'Watermark {index+1} Color')
                                        watermark_text_font = gr.Dropdown(label=f'Watermark {index+1} Fonts', choices=fonts_list, value=fonts_list[0])
                                        watermark_text_size = gr.Slider(label=f'Watermark {index+1} Size', visible=True, minimum=10, maximum=500, step=1, value=50)
                                    watermark_text_gr_list.append(watermark_text_gr)
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
                            
                        watermark_tabs = watermark_tabs_acc
        for index, watermark_type_data in enumerate(watermark_type_list):
            watermark_type_data.change(
                lambda type_data:dict(zip(watermark_image_gr_list+watermark_text_gr_list, [gr_show(type_data=='Image')]*len(watermark_image_gr_list)+[gr_show(type_data=='Text')]*len(watermark_text_gr_list))),
                inputs=[watermark_type_data],
                outputs=watermark_image_gr_list+watermark_text_gr_list
            )
        disable_watermark.change(
            lambda disable:{
                watermark_tabs:gr_show(not disable)
            },
            inputs=[disable_watermark],
            outputs=watermark_tabs
        )
        disable_upscaler.change(
            lambda disable: {
                ddetailer_before_upscaler:gr_show(not disable),
                upscaler_sample:gr_show(not disable),
                upscaler_index:gr_show(not disable),
                scalevalue:gr_show(not disable),
                overlap:gr_show(not disable),
                rewidth:gr_show(not disable),
                reheight:gr_show(not disable),
                denoising_strength:gr_show(not disable),
            },
            inputs= [disable_upscaler],
            outputs =[ddetailer_before_upscaler, upscaler_sample, upscaler_index, scalevalue, overlap, rewidth, reheight, denoising_strength]
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
        ret += [disable_watermark]
        ret += [disable_upscaler, ddetailer_before_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength]
        ret += [disable_detailer, disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model, dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur]
        ret += dino_detection_prompt_list + \
                dino_detection_positive_list + \
                dino_detection_negative_list + \
                dino_detection_denoise_list + \
                dino_detection_cfg_list + \
                dino_detection_steps_list + \
                dino_detection_spliter_disable_list + \
                dino_detection_spliter_remove_area_list + \
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
                watermark_alpha_list

        return ret
    
    def dino_detect_detailer(self, p, init_image,
                             disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model,
                             dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur,
                             dino_detect_count,
                             dino_detection_prompt_list,
                             dino_detection_positive_list,
                             dino_detection_negative_list,
                             dino_detection_denoise_list,
                             dino_detection_cfg_list,
                             dino_detection_steps_list,
                             dino_detection_spliter_disable_list,
                             dino_detection_spliter_remove_area_list):
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
            mask = dino_detect_from_prompt(dino_detection_prompt_list[detect_index], detailer_sam_model, detailer_dino_model, init_image, not disable_mask_paint_mode and isinstance(p, StableDiffusionProcessingImg2Img), inpaint_mask_mode, getattr(p,'image_mask',None))
            if mask is not None:
                if not dino_detection_spliter_disable_list[detect_index]:
                    mask = mask_spliter_and_remover(mask, dino_detection_spliter_remove_area_list[detect_index])
                    for mask_index, mask_split in enumerate(mask):
                        pi.seed = self.target_seeds + mask_index + detect_index
                        pi.init_images = [init_image]
                        pi.image_mask = Image.fromarray(mask_split)
                        if shared.opts.data.get('save_ddsd_working_on_dino_mask_images', False):
                            images.save_image(pi.image_mask, p.outpath_samples, "Mask", pi.seed, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                        state.job_count += 1
                        processed = processing.process_images(pi)
                        init_image = processed.images[0]
                        if shared.opts.data.get('save_ddsd_working_on_images', False):
                            images.save_image(init_image, p.outpath_samples, "Mask_Result", pi.seed, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                else:
                    pi.seed = self.target_seeds + detect_index
                    pi.init_images = [init_image]
                    pi.image_mask = Image.fromarray(mask)
                    if shared.opts.data.get('save_ddsd_working_on_dino_mask_images', False):
                        images.save_image(pi.image_mask, p.outpath_samples, "Mask", pi.seed, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                    state.job_count += 1
                    processed = processing.process_images(pi)
                    init_image = processed.images[0]
                    if shared.opts.data.get('save_ddsd_working_on_images', False):
                        images.save_image(init_image, p.outpath_samples, "Mask_Result", pi.seed, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
                p.extra_generation_params[f'DINO {detect_index + 1}'] = dino_detection_prompt_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} Positive'] = processed.all_prompts[0] if dino_detection_positive_list[detect_index] else "original"
                p.extra_generation_params[f'DINO {detect_index + 1} Negative'] = processed.all_negative_prompts[0] if dino_detection_negative_list[detect_index] else "original"
                p.extra_generation_params[f'DINO {detect_index + 1} Denoising'] = pi.denoising_strength
                p.extra_generation_params[f'DINO {detect_index + 1} CFG Scale'] = pi.cfg_scale
                p.extra_generation_params[f'DINO {detect_index + 1} Steps'] = pi.steps
                p.extra_generation_params[f'DINO {detect_index + 1} Spliter'] = not dino_detection_spliter_disable_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} SplitRemove Area'] = dino_detection_spliter_remove_area_list[detect_index]
            else:
                p.extra_generation_params[f'DINO {detect_index + 1}'] = dino_detection_prompt_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} Positive'] = "Error"
                p.extra_generation_params[f'DINO {detect_index + 1} Negative'] = "Error"
                p.extra_generation_params[f'DINO {detect_index + 1} Denoising'] = pi.denoising_strength
                p.extra_generation_params[f'DINO {detect_index + 1} CFG Scale'] = pi.cfg_scale
                p.extra_generation_params[f'DINO {detect_index + 1} Steps'] = pi.steps
                p.extra_generation_params[f'DINO {detect_index + 1} Spliter'] = not dino_detection_spliter_disable_list[detect_index]
                p.extra_generation_params[f'DINO {detect_index + 1} SplitRemove Area'] = dino_detection_spliter_remove_area_list[detect_index]
        return init_image
    
    def upscale(self, p, init_image, 
                scalevalue, upscaler_sample, overlap, rewidth, reheight, denoising_strength, 
                detailer_mask_blur, dino_full_res_inpaint, dino_inpaint_padding):
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
        if(self.upscaler.name != "None"): 
            img = self.upscaler.scaler.upscale(init_image, scalevalue, self.upscaler.data_path)
        else:
            img = init_image

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
        if shared.opts.data.get('save_ddsd_working_on_images', False):
            images.save_image(init_image, p.outpath_samples, "Upscale Working", pi.seed, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
        return init_image
    
    def watermark(self, p, init_image):
        if shared.opts.data.get('save_ddsd_watermark_with_and_without', False):
            images.save_image(init_image, p.outpath_samples, "Without_Watermark", self.target_seeds, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
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
    
    def process(self, p,
            enable_script_names,
            disable_watermark,
            disable_upscaler, ddetailer_before_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength,
            disable_detailer, disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model,
            dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur,
            *args):
        self.restore_script(p)
        self.enable_script_names = enable_script_names
        self.disable_watermark = disable_watermark
        self.disable_upscaler = disable_upscaler
        self.ddetailer_before_upscaler = ddetailer_before_upscaler
        self.scalevalue = scalevalue
        self.upscaler_sample = upscaler_sample
        self.overlap = overlap
        self.upscaler_index = upscaler_index
        self.rewidth = rewidth
        self.reheight = reheight
        self.denoising_strength = denoising_strength
        self.disable_detailer = disable_detailer
        self.disable_mask_paint_mode = disable_mask_paint_mode
        self.inpaint_mask_mode = inpaint_mask_mode
        self.detailer_sample = detailer_sample
        self.detailer_sam_model = detailer_sam_model
        self.detailer_dino_model = detailer_dino_model
        self.dino_full_res_inpaint = dino_full_res_inpaint
        self.dino_inpaint_padding = dino_inpaint_padding
        self.detailer_mask_blur = detailer_mask_blur
        args_list = [*args]
        self.dino_detect_count = shared.opts.data.get('dino_detect_count', 2)
        self.dino_detection_prompt_list = args_list[self.dino_detect_count * 0:self.dino_detect_count * 1]
        self.dino_detection_positive_list = args_list[self.dino_detect_count * 1:self.dino_detect_count * 2]
        self.dino_detection_negative_list = args_list[self.dino_detect_count * 2:self.dino_detect_count * 3]
        self.dino_detection_denoise_list = args_list[self.dino_detect_count * 3:self.dino_detect_count * 4]
        self.dino_detection_cfg_list = args_list[self.dino_detect_count * 4:self.dino_detect_count * 5]
        self.dino_detection_steps_list = args_list[self.dino_detect_count * 5:self.dino_detect_count * 6]
        self.dino_detection_spliter_disable_list = args_list[self.dino_detect_count * 6:self.dino_detect_count * 7]
        self.dino_detection_spliter_remove_area_list = args_list[self.dino_detect_count * 7:self.dino_detect_count * 8]
        self.watermark_count = shared.opts.data.get('watermark_count', 1)
        self.watermark_type_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 0:self.dino_detect_count * 8 + self.watermark_count * 1]
        self.watermark_position_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 1:self.dino_detect_count * 8 + self.watermark_count * 2]
        self.watermark_image_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 2:self.dino_detect_count * 8 + self.watermark_count * 3]
        self.watermark_image_size_width_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 3:self.dino_detect_count * 8 + self.watermark_count * 4]
        self.watermark_image_size_height_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 4:self.dino_detect_count * 8 + self.watermark_count * 5]
        self.watermark_text_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 5:self.dino_detect_count * 8 + self.watermark_count * 6]
        self.watermark_text_color_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 6:self.dino_detect_count * 8 + self.watermark_count * 7]
        self.watermark_text_font_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 7:self.dino_detect_count * 8 + self.watermark_count * 8]
        self.watermark_text_size_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 8:self.dino_detect_count * 8 + self.watermark_count * 9]
        self.watermark_padding_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 9:self.dino_detect_count * 8 + self.watermark_count * 10]
        self.watermark_alpha_list = args_list[self.dino_detect_count * 8 + self.watermark_count * 10:self.dino_detect_count * 8 + self.watermark_count * 11]
        self.script_names_list = [x.strip()+'.py' for x in enable_script_names.split(';') if len(x) > 1]
        self.i2i_scripts = [x for x in self.original_scripts if os.path.basename(x.filename) in self.script_names_list].copy()
        self.i2i_scripts_always = [x for x in self.original_scripts_always if os.path.basename(x.filename) in self.script_names_list].copy()
        self.upscaler = shared.sd_upscalers[upscaler_index]
    
    def before_process_batch(self, p, *args, **kargs):
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
        devices.torch_gc()
        output_image = pp.image
        self.target_prompts = p.all_prompts[self.iter_number * p.batch_size:(self.iter_number + 1) * p.batch_size][self.batch_number]
        self.target_negative_prompts = p.all_negative_prompts[self.iter_number * p.batch_size:(self.iter_number + 1) * p.batch_size][self.batch_number]
        self.target_seeds = p.all_seeds[self.iter_number * p.batch_size:(self.iter_number + 1) * p.batch_size][self.batch_number]
        if shared.opts.data.get('save_ddsd_working_on_images', False):
            images.save_image(output_image, p.outpath_samples, "Generate Working", self.target_seeds, self.target_prompts, opts.samples_format, info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, None, self.iter_number, self.batch_number), p=p)
        
        if self.ddetailer_before_upscaler and not self.disable_upscaler:
            output_image = self.upscale(p, output_image,
                                        self.scalevalue, self.upscaler_sample, 
                                        self.overlap, self.rewidth, self.reheight, self.denoising_strength,
                                        self.detailer_mask_blur, self.dino_full_res_inpaint, self.dino_inpaint_padding)
        devices.torch_gc()
        
        if not self.disable_detailer:
            output_image = self.dino_detect_detailer(p, output_image, 
                                                     self.disable_mask_paint_mode, self.inpaint_mask_mode, self.detailer_sample, self.detailer_sam_model, self.detailer_dino_model, 
                                                     self.dino_full_res_inpaint, self.dino_inpaint_padding, self.detailer_mask_blur,
                                                     self.dino_detect_count,
                                                     self.dino_detection_prompt_list,
                                                     self.dino_detection_positive_list,
                                                     self.dino_detection_negative_list,
                                                     self.dino_detection_denoise_list,
                                                     self.dino_detection_cfg_list,
                                                     self.dino_detection_steps_list,
                                                     self.dino_detection_spliter_disable_list,
                                                     self.dino_detection_spliter_remove_area_list)
        devices.torch_gc()
        
        if not self.ddetailer_before_upscaler and not self.disable_upscaler:
            output_image = self.upscale(p, output_image,
                                        self.scalevalue, self.upscaler_sample, 
                                        self.overlap, self.rewidth, self.reheight, self.denoising_strength,
                                        self.detailer_mask_blur, self.dino_full_res_inpaint, self.dino_inpaint_padding)
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
    shared.opts.add_option("save_ddsd_working_on_dino_mask_images", shared.OptionInfo(
        False, "Save dino mask images you are working on", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("dino_detect_count", shared.OptionInfo(
        2, "Dino Detect Max Count", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))
    shared.opts.add_option("save_ddsd_watermark_with_and_without", shared.OptionInfo(
        False, "Save with and without watermark ", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("watermark_count", shared.OptionInfo(
        1, "Watermark Count", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))

modules.script_callbacks.on_ui_settings(on_ui_settings)