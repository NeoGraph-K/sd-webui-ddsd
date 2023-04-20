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
from scripts.ddsd_utils import dino_detect_from_prompt, mask_spliter_and_remover, I2I_Generator_Create

import modules
from modules import processing, shared, images, devices, modelloader
from modules.processing import Processed, StableDiffusionProcessingImg2Img
from modules.shared import opts, state
from modules.sd_models import model_hash
from modules.paths import models_path

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
        
    def title(self):
        return "ddetailer + sdupscale"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        sample_list = [x.name for x in shared.list_samplers()]
        sample_list = [x for x in sample_list if x not in ['PLMS','UniPC','DDIM']]
        sample_list.insert(0,"Original")
        ret = []
        image_detectors = []
        dino_detection_prompt_list = []
        dino_detection_positive_list = []
        dino_detection_negative_list = []
        dino_detection_denoise_list = []
        dino_detection_cfg_list = []
        dino_detection_steps_list = []
        dino_detection_spliter_disable_list = []
        dino_detection_spliter_remove_area_list = []
        dino_tabs = None
        
        with gr.Group():
            with gr.Accordion("Random Controlnet", open = False, elem_id="ddsd_random_controlnet_acc"):
                with gr.Column():
                    control_net_info = gr.HTML('<br><p style="margin-bottom:0.75em">T2I Control Net random image process</p>', visible=not is_img2img)
                    disable_random_control_net = gr.Checkbox(label='Disable Random Controlnet', value=True, visible=not is_img2img)
                    cn_models_num = shared.opts.data.get("control_net_max_models_num", 1)
                    for n in range(cn_models_num):
                        cn_image_detect_folder = gr.Textbox(label=f"{n} Control Model Image Random Folder(Using glob)", elem_id=f"{n}_cn_image_detector", value='',show_label=True, lines=1, placeholder="search glob image folder and file extension. ex ) - ./base/**/*.png", visible=False)
                        image_detectors.append(cn_image_detect_folder)
        
            with gr.Accordion("Script Option", open = False, elem_id="ddsd_enable_script_acc"):
                with gr.Column():
                    all_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I All process target script</p>')
                    enable_script_names = gr.Textbox(label="Enable Script(Extension)", elem_id="enable_script_names", value='dynamic_thresholding;dynamic_prompting',show_label=True, lines=1, placeholder="Extension python file name(ex - dynamic_thresholding;dynamic_prompting)")
        
            with gr.Accordion("Upscaler", open=False, elem_id="ddsd_upsacler_acc"):
                with gr.Column():
                    sd_upscale_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Upscaler Option</p>')
                    disable_upscaler = gr.Checkbox(label='Disable Upscaler', elem_id='disable_upscaler', value=True, visible=True)
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
        
        disable_random_control_net.change(
            lambda disable:dict(zip(image_detectors,[gr_show(not disable)]*cn_models_num)),
            inputs=[disable_random_control_net],
            outputs=image_detectors
        )
        disable_upscaler.change(
            lambda disable: {
                upscaler_sample:gr_show(not disable),
                upscaler_index:gr_show(not disable),
                scalevalue:gr_show(not disable),
                overlap:gr_show(not disable),
                rewidth:gr_show(not disable),
                reheight:gr_show(not disable),
                denoising_strength:gr_show(not disable),
            },
            inputs= [disable_upscaler],
            outputs =[upscaler_sample, upscaler_index, scalevalue, overlap, rewidth, reheight, denoising_strength]
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
        ret += [disable_random_control_net]
        ret += [disable_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength]
        ret += [disable_detailer, disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model, dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur]
        ret += dino_detection_prompt_list + dino_detection_positive_list + dino_detection_negative_list + dino_detection_denoise_list + dino_detection_cfg_list + dino_detection_steps_list + dino_detection_spliter_disable_list + dino_detection_spliter_remove_area_list + image_detectors

        return ret

    def run(self, p, 
            enable_script_names,
            disable_random_control_net, 
            disable_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength,
            disable_detailer, disable_mask_paint_mode, inpaint_mask_mode, detailer_sample, detailer_sam_model, detailer_dino_model,
            dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur,
            *args):
        args_list = [*args]
        dino_detect_count = shared.opts.data.get('dino_detect_count', 2)
        dino_detection_prompt_list = args_list[dino_detect_count * 0:dino_detect_count * 1]
        dino_detection_positive_list = args_list[dino_detect_count * 1:dino_detect_count * 2]
        dino_detection_negative_list = args_list[dino_detect_count * 2:dino_detect_count * 3]
        dino_detection_denoise_list = args_list[dino_detect_count * 3:dino_detect_count * 4]
        dino_detection_cfg_list = args_list[dino_detect_count * 4:dino_detect_count * 5]
        dino_detection_steps_list = args_list[dino_detect_count * 5:dino_detect_count * 6]
        dino_detection_spliter_disable_list = args_list[dino_detect_count * 6:dino_detect_count * 7]
        dino_detection_spliter_remove_area_list = args_list[dino_detect_count * 7:dino_detect_count * 8]
        random_controlnet_list = args_list[dino_detect_count * 8:]
        
        processing.fix_seed(p)
        initial_info = []
        initial_prompt = []
        initial_negative = []
        p.batch_size = 1
        ddetail_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p_txt = p
        
        upscaler = shared.sd_upscalers[upscaler_index]
        script_names_list = [x.strip()+'.py' for x in enable_script_names.split(';') if len(x) > 1]
        seed = p_txt.seed
        
        if self.original_scripts is None: self.original_scripts = p_txt.scripts.scripts.copy()
        else: 
            if len(p_txt.scripts.scripts) != len(self.original_scripts): p_txt.scripts.scripts = self.original_scripts.copy()
        if self.original_scripts_always is None: self.original_scripts_always = p_txt.scripts.alwayson_scripts.copy()
        else: 
            if len(p_txt.scripts.alwayson_scripts) != len(self.original_scripts_always): p_txt.scripts.alwayson_scripts = self.original_scripts_always.copy()
        p_txt.scripts.scripts = [x for x in p_txt.scripts.scripts if os.path.basename(x.filename) not in [__file__]]
        if not disable_random_control_net:
            controlnet = [x for x in p_txt.scripts.scripts if os.path.basename(x.filename) in ['controlnet.py']]
            assert len(controlnet) > 0, 'Do not find controlnet, please install controlnet or disable random control net option'
            controlnet = controlnet[0]
            controlnet_args = p_txt.script_args[controlnet.args_from:controlnet.args_to]
            controlnet_search_folders = random_controlnet_list.copy()
            controlnet_image_files = []
            for con_n, conet in enumerate(controlnet_args):
                files = []
                if conet.enabled:
                    if '**' in controlnet_search_folders[con_n]:
                        files = glob(controlnet_search_folders[con_n], recursive=True)
                    else:
                        files = glob(controlnet_search_folders[con_n])
                controlnet_image_files.append(files.copy())
        
        t2i_scripts = p_txt.scripts.scripts.copy()
        i2i_scripts = [x for x in t2i_scripts if os.path.basename(x.filename) in script_names_list].copy()
        t2i_scripts_always = p_txt.scripts.alwayson_scripts.copy()
        i2i_scripts_always = [x for x in t2i_scripts_always if os.path.basename(x.filename) in script_names_list].copy()
        
        print(f"DDetailer {p.width}x{p.height}.")
        
        output_images = []
        result_images = []
        state.job = 'T2I Generate'
        state.job_count = 0
        state.job_count += ddetail_count
        for n in range(ddetail_count):
            devices.torch_gc()
            start_seed = seed + n
            cn_file_paths = []
            print(f"Processing initial image for output generation {n + 1} (Generate).")
            p_txt.seed = start_seed
            p_txt.scripts.scripts = t2i_scripts.copy()
            p_txt.scripts.alwayson_scripts = t2i_scripts_always.copy()
            if not disable_random_control_net:
                for con_n, conet in enumerate(controlnet_args):
                    cn_file_paths.append([])
                    if len(controlnet_image_files[con_n]) > 0:
                        cn_file_paths[con_n].append(choice(controlnet_image_files[con_n]))
                        cn_image = Image.open(cn_file_paths[con_n][0])
                        cn_np = np.array(cn_image)
                        if cn_image.mode == 'RGB':
                            cn_np = np.concatenate([cn_np, 255*np.ones((cn_np.shape[0], cn_np.shape[1], 1), dtype=np.uint8)], axis=-1)
                        cn_np_image = copy.deepcopy(cn_np[:,:,:3])
                        cn_np_mask = copy.deepcopy(cn_np)
                        cn_np_mask[:,:,:3] = 0
                        conet.image = {'image':cn_np_image,'mask':cn_np_mask}
            processed = processing.process_images(p_txt)
            initial_info.append(processed.info)
            initial_info[n] += ', ' + ', '.join([f'ControlNet {n} Random Image : {x}' for n, x in enumerate(cn_file_paths) if len(x) > 0])
            posi, nega = processed.all_prompts[0], processed.all_negative_prompts[0]
            
            initial_prompt.append(posi)
            initial_negative.append(nega)
            output_images.append(processed.images[0])
            
            if shared.opts.data.get('save_ddsd_working_on_images', False):
                images.save_image(output_images[n], p.outpath_samples, "Generate Working", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
                
            if not disable_detailer:
                init_img = output_images[-1]
                state.job = 'DINO Detect Pregress'
                for detect_index in range(dino_detect_count):
                    if len(dino_detection_prompt_list[detect_index]) < 1: continue
                    p = I2I_Generator_Create(
                        p_txt, ('Euler' if p_txt.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p_txt.sampler_name) if detailer_sample == 'Original' else detailer_sample,
                        detailer_mask_blur, dino_full_res_inpaint, dino_inpaint_padding, init_img,
                        dino_detection_denoise_list[detect_index],
                        dino_detection_cfg_list[detect_index] if dino_detection_cfg_list[detect_index] > 0 else p_txt.cfg_scale,
                        dino_detection_steps_list[detect_index] if dino_detection_steps_list[detect_index] > 0 else p_txt.steps,
                        p_txt.width, p_txt.height, p_txt.tiling, p_txt.scripts, i2i_scripts, i2i_scripts_always, p_txt.script_args,
                        dino_detection_positive_list[detect_index] if dino_detection_positive_list[detect_index] else initial_prompt[-1],
                        dino_detection_negative_list[detect_index] if dino_detection_negative_list[detect_index] else initial_negative[-1]
                    )
                    mask = dino_detect_from_prompt(dino_detection_prompt_list[detect_index], detailer_sam_model, detailer_dino_model, init_img, not disable_mask_paint_mode and isinstance(p_txt, StableDiffusionProcessingImg2Img), inpaint_mask_mode, getattr(p_txt,'image_mask',None))
                    if mask is not None:
                        if dino_detection_spliter_disable_list[detect_index]:
                            mask = mask_spliter_and_remover(mask, dino_detection_spliter_remove_area_list[detect_index])
                            for mask_split in mask:
                                p.init_images = [init_img]
                                p.image_mask = Image.fromarray(mask_split)
                                if shared.opts.data.get('save_ddsd_working_on_dino_mask_images', False):
                                    images.save_image(p.image_mask, p.outpath_samples, "Mask", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
                                state.job_count += 1
                                processed = processing.process_images(p)
                                p.seed = processed.seed + 1
                                init_img = processed.images[0]
                        else:
                            p.init_images = [init_img]
                            p.image_mask = Image.fromarray(mask)
                            if shared.opts.data.get('save_ddsd_working_on_dino_mask_images', False):
                                images.save_image(p.image_mask, p.outpath_samples, "Mask", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
                            state.job_count += 1
                            processed = processing.process_images(p)
                            p.seed = processed.seed + 1
                            init_img = processed.images[0]
                    initial_info[n] += ', '.join(['',f'DINO {detect_index+1} : {dino_detection_prompt_list[detect_index]}', 
                                                   f'DINO {detect_index+1} Positive : {processed.all_prompts[0] if dino_detection_positive_list[detect_index] else "original"}', 
                                                   f'DINO {detect_index+1} Negative : {processed.all_negative_prompts[0] if dino_detection_negative_list[detect_index] else "original"}',
                                                   f'DINO {detect_index+1} Denoising : {p.denoising_strength}',
                                                   f'DINO {detect_index+1} CFG Scale : {p.cfg_scale}', 
                                                   f'DINO {detect_index+1} Steps : {p.steps}',
                                                   f'DINO {detect_index+1} Spliter : {"True" if dino_detection_spliter_disable_list[detect_index] else "False"}',
                                                   f'DINO {detect_index+1} Split Remove Area : {dino_detection_spliter_remove_area_list[detect_index]}'])
                    if shared.opts.data.get('save_ddsd_working_on_images', False):
                        images.save_image(init_img, p.outpath_samples, "DINO Working", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
                    output_images[n] = init_img
                    
            if not disable_upscaler:
                
                p = I2I_Generator_Create(
                        p_txt, ('Euler' if p_txt.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p_txt.sampler_name) if upscaler_sample == 'Original' else upscaler_sample,
                        detailer_mask_blur, dino_full_res_inpaint, dino_inpaint_padding, output_images[n],
                        denoising_strength, p_txt.cfg_scale, p_txt.steps,
                        rewidth, reheight, p_txt.tiling, p_txt.scripts, i2i_scripts, i2i_scripts_always, p_txt.script_args,
                        initial_prompt[n], initial_negative[n]
                    )
                
                initial_info[n] += ', '.join(['',
                    f'Tile upscale value : {scalevalue}',
                    f'Tile upscale width : {rewidth}',
                    f'Tile upscale height : {reheight}',
                    f'Tile upscale overlap : {overlap}',
                    f'Tile upscale upscaler : {upscaler.name}'
                ])
                
                init_img = output_images[n]

                if(upscaler.name != "None"): 
                    img = upscaler.scaler.upscale(init_img, scalevalue, upscaler.data_path)
                else:
                    img = init_img

                devices.torch_gc()
                grid = images.split_grid(img, tile_w=rewidth, tile_h=reheight, overlap=overlap)

                batch_size = p.batch_size

                work = []

                for y, h, row in grid.tiles:
                    for tiledata in row:
                        work.append(tiledata[2])

                batch_count = math.ceil(len(work) / batch_size)
                state.job = 'Upscaler Batching'
                state.job_count += batch_count

                print(f"Tile upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches (I2I).")

                p.seed = start_seed
                
                work_results = []
                for i in range(batch_count):
                    p.batch_size = batch_size
                    p.init_images = work[i*batch_size:(i+1)*batch_size]

                    state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                    processed = processing.process_images(p)

                    p.seed = processed.seed + 1
                    work_results += processed.images

                image_index = 0
                for y, h, row in grid.tiles:
                    for tiledata in row:
                        tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (rewidth, reheight))
                        image_index += 1
                output_images[n] = images.combine_grid(grid)
                if shared.opts.data.get('save_ddsd_working_on_images', False):
                    images.save_image(output_images[n], p.outpath_samples, "Upscale Working", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
            result_images.append(output_images[n])
            images.save_image(result_images[-1], p.outpath_samples, "", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
        state.end()
        p_txt.scripts.scripts = self.original_scripts.copy()
        p_txt.scripts.alwayson_scripts = self.original_scripts_always.copy()
        return Processed(p_txt, result_images, start_seed, initial_info[0], all_prompts=initial_prompt, all_negative_prompts=initial_negative, infotexts=initial_info)
    



def on_ui_settings():
    section = ('ddsd_script', "DDSD")
    shared.opts.add_option("save_ddsd_working_on_images", shared.OptionInfo(
        False, "Save all images you are working on", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("save_ddsd_working_on_dino_mask_images", shared.OptionInfo(
        False, "Save dino mask images you are working on", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("dino_detect_count", shared.OptionInfo(
        2, "Dino Detect Max Count", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))

modules.script_callbacks.on_ui_settings(on_ui_settings)