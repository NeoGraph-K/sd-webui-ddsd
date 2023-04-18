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
from scripts.ddsd_utils import dino_detect_from_prompt, try_convert

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
    def title(self):
        return "ddetailer + sdupscale"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        sample_list = [x.name for x in shared.list_samplers()]
        sample_list = [x for x in sample_list if x not in ['PLMS','UniPC','DDIM']]
        sample_list.insert(0,"Original")
        ret = []
        
        with gr.Group():
            control_net_info = gr.HTML('<br><p style="margin-bottom:0.75em">T2I Control Net random image process</p>', visible=not is_img2img)
            disable_random_control_net = gr.Checkbox(label='Disable Random Controlnet', value=True, visible=not is_img2img)
            cn_models_num = shared.opts.data.get("control_net_max_models_num", 1)
            image_detectors = []
            with gr.Column():
                for n in range(cn_models_num):
                    cn_image_detect_folder = gr.Textbox(label=f"{n} Control Model Image Random Folder(Using glob)", elem_id=f"{n}_cn_image_detector", value='',show_label=True, lines=1, placeholder="search glob image folder and file extension. ex ) - ./base/**/*.png", visible=False)
                    image_detectors.append(cn_image_detect_folder)
        
        disable_random_control_net.change(
            lambda disable:dict(zip(image_detectors,[gr_show(not disable)]*cn_models_num)),
            inputs=[disable_random_control_net],
            outputs=image_detectors
        )
        
        with gr.Group():
            all_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I All process target script</p>')
            enable_script_names = gr.Textbox(label="Enable Script(Extension)", elem_id="enable_script_names", value='dynamic_thresholding;dynamic_prompting',show_label=True, lines=1, placeholder="Extension python file name(ex - dynamic_thresholding;dynamic_prompting)")
        
        with gr.Group():
            sd_upscale_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Upscaler Option</p>')
            disable_upscaler = gr.Checkbox(label='Disable Upscaler', elem_id='disable_upscaler', value=False, visible=True)
            with gr.Column():
                with gr.Row():
                    upscaler_sample = gr.Dropdown(label='Upscaler Sampling', elem_id='upscaler_sample', choices=sample_list, value=sample_list[0], visible=True, type="value")
                    upscaler_index = gr.Dropdown(label='Upscaler', elem_id='upscaler_index', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[-1].name, type="index")
                scalevalue = gr.Slider(minimum=1, maximum=16, step=0.5, elem_id='upscaler_scalevalue', label='Resize', value=2)
                overlap = gr.Slider(minimum=0, maximum=256, step=32, elem_id='upscaler_overlap', label='Tile overlap', value=32)
                with gr.Row():
                    rewidth = gr.Slider(minimum=0, maximum=1024, step=64, elem_id='upscaler_rewidth', label='Width', value=512)
                    reheight = gr.Slider(minimum=0, maximum=1024, step=64, elem_id='upscaler_reheight', label='Height', value=512)
                denoising_strength = gr.Slider(minimum=0, maximum=1.0, step=0.01, elem_id='upscaler_denoising', label='Denoising strength', value=0.1)

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
        
        with gr.Group():
            ddetailer_target_info = gr.HTML('<br><p style="margin-bottom:0.75em">I2I Detection Detailer Option</p>')
            disable_detailer = gr.Checkbox(label='Disable Detection Detailer', elem_id='disable_detailer',value=False, visible=True)
            detailer_sample = gr.Dropdown(label='Detailer Sampling', elem_id='detailer_sample', choices=sample_list, value=sample_list[0], visible=True, type="value")
            with gr.Column():
                with gr.Row():
                    detailer_sam_model = gr.Dropdown(label='Detailer SAM Model', elem_id='detailer_sam_model', choices=sam_model_list(), value=sam_model_list()[0], visible=True)
                    detailer_dino_model = gr.Dropdown(label='Deteiler DINO Model', elem_id='detailer_dino_model', choices=dino_model_list(), value=dino_model_list()[0], visible=True)
                with gr.Column():
                    dino_detection_prompt = gr.Textbox(label="Detect Prompt", elem_id="detailer_detect_prompt", show_label=True, lines=2, placeholder="Detect Token Prompt(ex - face:level(0-2):threshold(0-1):dilation(0-128)$denoise(0-1);hand)", visible=True)
                    dino_detection_positive = gr.Textbox(label="Positive Prompt", elem_id="detailer_detect_positive", show_label=True, lines=3, placeholder="Detect Mask Inpaint Positive(ex - pureeros;red hair)", visible=True)
                    dino_detection_negative = gr.Textbox(label="Negative Prompt", elem_id="detailer_detect_negative", show_label=True, lines=3, placeholder="Detect Mask Inpaint Negative(ex - easynagetive;nsfw)", visible=True)
                with gr.Row():
                    dino_full_res_inpaint = gr.Checkbox(label='Inpaint at full resolution ', elem_id='detailer_full_res', value=True, visible = True)
                    dino_inpaint_padding = gr.Slider(label='Inpaint at full resolution padding, pixels ', elem_id='detailer_padding', minimum=0, maximum=256, step=4, value=32, visible=True)
                    detailer_mask_blur = gr.Slider(label='Detailer Blur', elem_id='detailer_mask_blur', minimum=0, maximum=64, step=1, value=4)

        disable_detailer.change(
            lambda disable:{
                detailer_sample:gr_show(not disable),
                detailer_sam_model:gr_show(not disable),
                detailer_dino_model:gr_show(not disable),
                dino_detection_prompt:gr_show(not disable),
                dino_detection_positive:gr_show(not disable),
                dino_detection_negative:gr_show(not disable),
                dino_full_res_inpaint:gr_show(not disable),
                dino_inpaint_padding:gr_show(not disable),
                detailer_mask_blur:gr_show(not disable)
            },
            inputs=[disable_detailer],
            outputs=[
                detailer_sample,
                detailer_sam_model,
                detailer_dino_model,
                dino_detection_prompt,
                dino_detection_positive,
                dino_detection_negative,
                dino_full_res_inpaint,
                dino_inpaint_padding,
                detailer_mask_blur
            ]
        )
        
        ret += [enable_script_names]
        ret += [disable_random_control_net]
        ret += [disable_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength]
        ret += [disable_detailer, detailer_sample, detailer_sam_model, detailer_dino_model, dino_detection_prompt, dino_detection_positive, dino_detection_negative, dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur]
        ret += image_detectors

        return ret

    def run(self, p, 
            enable_script_names,
            disable_random_control_net, 
            disable_upscaler, scalevalue, upscaler_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength,
            disable_detailer, detailer_sample, detailer_sam_model, detailer_dino_model, dino_detection_prompt, dino_detection_positive, dino_detection_negative,
            dino_full_res_inpaint, dino_inpaint_padding, detailer_mask_blur,
            *args):
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
        i2i_sample = ''
        if detailer_sample == 'Original':
            i2i_sample = 'Euler' if p_txt.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p_txt.sampler_name
        else:
            i2i_sample = detailer_sample
        p = StableDiffusionProcessingImg2Img(
                init_images = None,
                resize_mode = 0,
                denoising_strength = 0,
                mask = None,
                mask_blur= detailer_mask_blur,
                inpainting_fill = 1,
                inpaint_full_res = dino_full_res_inpaint,
                inpaint_full_res_padding= dino_inpaint_padding,
                inpainting_mask_invert= 0,
                sd_model=p_txt.sd_model,
                outpath_samples=p_txt.outpath_samples,
                outpath_grids=p_txt.outpath_grids,
                prompt='',
                negative_prompt='',
                styles=p_txt.styles,
                seed=p_txt.seed,
                subseed=p_txt.subseed,
                subseed_strength=p_txt.subseed_strength,
                seed_resize_from_h=p_txt.seed_resize_from_h,
                seed_resize_from_w=p_txt.seed_resize_from_w,
                sampler_name=i2i_sample,
                n_iter=p_txt.n_iter,
                steps=p_txt.steps,
                cfg_scale=p_txt.cfg_scale,
                width=p_txt.width,
                height=p_txt.height,
                tiling=p_txt.tiling,
            )
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.override_settings = {}
        
        if upscaler_sample == 'Original':
            i2i_sample = 'Euler' if p_txt.sampler_name in ['PLMS', 'UniPC', 'DDIM'] else p_txt.sampler_name
        else:
            i2i_sample = upscaler_sample
        p2 = StableDiffusionProcessingImg2Img(
            sd_model=p_txt.sd_model,
            outpath_samples=p_txt.outpath_samples,
            outpath_grids=p_txt.outpath_grids,
            prompt='',
            negative_prompt='',
            styles=p_txt.styles,
            seed=p_txt.seed,
            subseed=p_txt.subseed,
            subseed_strength=p_txt.subseed_strength,
            seed_resize_from_h=p_txt.seed_resize_from_h,
            seed_resize_from_w=p_txt.seed_resize_from_w,
            seed_enable_extras=True,
            sampler_name=i2i_sample,
            batch_size=1,
            n_iter=1,
            steps=p_txt.steps,
            cfg_scale=p_txt.cfg_scale,
            width=rewidth,
            height=reheight,
            restore_faces=p_txt.restore_faces,
            tiling=p_txt.tiling,
            init_images=[],
            mask=None,
            mask_blur=detailer_mask_blur,
            inpainting_fill=1,
            resize_mode=0,
            denoising_strength=denoising_strength,
            inpaint_full_res=dino_full_res_inpaint,
            inpaint_full_res_padding=dino_inpaint_padding,
            inpainting_mask_invert=0,
        )
        p2.do_not_save_grid = True
        p2.do_not_save_samples = True
        p2.override_settings = {}
        
        upscaler = shared.sd_upscalers[upscaler_index]
        script_names_list = [x.strip()+'.py' for x in enable_script_names.split(';') if len(x) > 1]
        processing.fix_seed(p2)
        seed = p_txt.seed
        
        original_scripts = p_txt.scripts.scripts.copy()
        original_scripts_always = p_txt.scripts.alwayson_scripts.copy()
        p_txt.scripts.scripts = [x for x in p_txt.scripts.scripts if os.path.basename(x.filename) not in [__file__]]
        if not disable_random_control_net:
            controlnet = [x for x in p_txt.scripts.scripts if os.path.basename(x.filename) in ['controlnet.py']]
            assert len(controlnet) > 0, 'Do not find controlnet, please install controlnet or disable random control net option'
            controlnet = controlnet[0]
            controlnet_args = p_txt.script_args[controlnet.args_from:controlnet.args_to]
            controlnet_search_folders = list(args)
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
        i2i_scripts = [x for x in t2i_scripts if os.path.basename(x.filename) in script_names_list]
        t2i_scripts_always = p_txt.scripts.alwayson_scripts.copy()
        i2i_scripts_always = [x for x in t2i_scripts_always if os.path.basename(x.filename) in script_names_list]
        p.scripts = p_txt.scripts
        p.script_args = p_txt.script_args
        p2.scripts = p_txt.scripts
        p2.script_args = p_txt.script_args
        
        p_txt.extra_generation_params["Tile upscale value"] = scalevalue
        p_txt.extra_generation_params["Tile upscale width"] = rewidth
        p_txt.extra_generation_params["Tile upscale height"] = reheight
        p_txt.extra_generation_params["Tile upscale overlap"] = overlap
        p_txt.extra_generation_params["Tile upscale upscaler"] = upscaler.name
        
        print(f"DDetailer {p.width}x{p.height}.")
        
        output_images = []
        result_images = []
        state.begin()
        state.job_count += ddetail_count
        for n in range(ddetail_count):
            devices.torch_gc()
            start_seed = seed + n
            print(f"Processing initial image for output generation {n + 1} (T2I).")
            p_txt.seed = start_seed
            p_txt.scripts.scripts = t2i_scripts.copy()
            p_txt.scripts.alwayson_scripts = t2i_scripts_always.copy()
            if not disable_random_control_net:
                for con_n, conet in enumerate(controlnet_args):
                    if len(controlnet_image_files[con_n]) > 0:
                        cn_image = Image.open(choice(controlnet_image_files[con_n]))
                        cn_np = np.array(cn_image)
                        if cn_image.mode == 'RGB':
                            cn_np = np.concatenate([cn_np, 255*np.ones((cn_np.shape[0], cn_np.shape[1], 1), dtype=np.uint8)], axis=-1)
                        cn_np_image = copy.deepcopy(cn_np[:,:,:3])
                        cn_np_mask = copy.deepcopy(cn_np)
                        cn_np_mask[:,:,:3] = 0
                        conet.image = {'image':cn_np_image,'mask':cn_np_mask}
            state.job = f'{n+1} image T2I Generate'
            processed = processing.process_images(p_txt)
            initial_info.append(processed.info)
            posi, nega = processed.all_prompts[0], processed.all_negative_prompts[0]
            
            initial_prompt.append(posi)
            initial_negative.append(nega)
            output_images.append(processed.images[0])
            
            if shared.opts.data.get('save_ddsd_working_on_images', False):
                images.save_image(output_images[n], p.outpath_samples, "", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
                
            if not disable_detailer:
                assert dino_detection_prompt, 'Please Input DINO Detect Prompt(Enable Logic Gate(OR,AND,XOR,NOR,NAND))(A OR B AND (C XOR D) NOR E NAND F)'
                p.scripts.scripts = i2i_scripts.copy()
                p.scripts.alwayson_scripts = i2i_scripts_always.copy()
                dino_detect_list = [x for x in dino_detection_prompt.split(';') if len(x) > 0]
                assert len(dino_detect_list) > 0, 'Please Input DINO Detect Prompt(ex - A;B)'
                dino_detect_positive_list = dino_detection_positive.split(';')
                while len(dino_detect_positive_list) < len(dino_detect_list):
                    dino_detect_positive_list.append('')
                dino_detect_negative_list = dino_detection_negative.split(';')
                while len(dino_detect_negative_list) < len(dino_detect_list):
                    dino_detect_negative_list.append('')
                dino_detect_positive_list = [x.strip() for x in dino_detect_positive_list]
                dino_detect_negative_list = [x.strip() for x in dino_detect_negative_list]
                
                init_img = output_images[-1]
                for detect_index, detect in enumerate(dino_detect_list):
                    detect = detect.split('$')
                    detect[0] = detect[0].strip()
                    p.denoising_strength = try_convert(detect[1], float, 0.4, 0, 1) if len(detect) > 1 else 0.4
                    mask = dino_detect_from_prompt(detect[0], detailer_sam_model, detailer_dino_model, init_img)
                    p.prompt = dino_detect_positive_list[detect_index] if dino_detect_positive_list[detect_index] else initial_prompt[-1]
                    p.negative_prompt = dino_detect_negative_list[detect_index] if dino_detect_negative_list[detect_index] else initial_negative[-1]
                    p.init_images = [init_img]
                    if mask is not None:
                        p.image_mask = mask
                        state.job_count += 1
                        state.job = f'{detect_index + 1} DINO I2I Inpaint Generate'
                        processed = processing.process_images(p)
                        p.seed = processed.seed + 1
                        init_img = processed.images[0]
                    initial_info[n] += ', '.join(['',f'{detect_index+1} DINO : {detect[0]}', 
                                                   f'{detect_index+1} DINO Positive : {processed.all_prompts[0] if dino_detect_positive_list[detect_index] else "original"}', 
                                                   f'{detect_index+1} DINO Negative : {processed.all_negative_prompts[0] if dino_detect_negative_list[detect_index] else "original"}',
                                                   f'{detect_index+1} DINO Denoising : {p.denoising_strength}'])
                    if shared.opts.data.get('save_ddsd_working_on_images', False):
                        images.save_image(init_img, p.outpath_samples, "", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
                    output_images[n] = init_img
                    
            if not disable_upscaler:
                p2.init_images = [output_images[n]]
                p2.prompt = initial_prompt[n]
                p2.negative_prompt = initial_negative[n]
                
                init_img = output_images[n]

                if(upscaler.name != "None"): 
                    img = upscaler.scaler.upscale(init_img, scalevalue, upscaler.data_path)
                else:
                    img = init_img

                devices.torch_gc()
                state.job_count += 1
                grid = images.split_grid(img, tile_w=rewidth, tile_h=reheight, overlap=overlap)

                batch_size = p2.batch_size

                work = []

                for y, h, row in grid.tiles:
                    for tiledata in row:
                        work.append(tiledata[2])

                batch_count = math.ceil(len(work) / batch_size)
                state.job_count += batch_count

                print(f"Tile upscaling will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches (I2I).")

                p2.seed = start_seed
                p2.scripts.scripts = i2i_scripts.copy()
                p2.scripts.alwayson_scripts = i2i_scripts_always.copy()
                
                work_results = []
                for i in range(batch_count):
                    p2.batch_size = batch_size
                    p2.init_images = work[i*batch_size:(i+1)*batch_size]

                    state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                    processed = processing.process_images(p2)

                    p2.seed = processed.seed + 1
                    work_results += processed.images

                image_index = 0
                for y, h, row in grid.tiles:
                    for tiledata in row:
                        tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (rewidth, reheight))
                        image_index += 1
                output_images[n] = images.combine_grid(grid)
                if shared.opts.data.get('save_ddsd_working_on_images', False):
                    images.save_image(output_images[n], p.outpath_samples, "", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
            result_images.append(output_images[n])
            images.save_image(result_images[-1], p.outpath_samples, "", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
        state.end()
        p_txt.scripts.scripts = original_scripts.copy()
        p_txt.scripts.alwayson_scripts = original_scripts_always.copy()
        return Processed(p_txt, result_images, start_seed, initial_info[0], all_prompts=initial_prompt, all_negative_prompts=initial_negative, infotexts=initial_info)
    



def on_ui_settings():
    section = ('ddsd_script', "DDSD")
    shared.opts.add_option("save_ddsd_working_on_images", shared.OptionInfo(
        False, "Save all images you are working on", gr.Checkbox, {"interactive": True}, section=section))

modules.script_callbacks.on_ui_settings(on_ui_settings)