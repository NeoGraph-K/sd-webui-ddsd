import os
import sys
import cv2
import math
import copy
from random import choice
from glob import glob

import modules.scripts as scripts
import gradio as gr
import numpy as np
from PIL import Image

from modules import processing, shared, sd_samplers, images, devices, scripts, script_callbacks, modelloader
from modules.processing import Processed, process_images, fix_seed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.shared import opts, cmd_opts, state

from modules.sd_models import model_hash
from modules.paths import models_path
from basicsr.utils.download_util import load_file_from_url

dd_models_path = os.path.join(models_path, "mmdet")


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
    from launch import is_installed, run
    if not is_installed("mmdet"):
        python = sys.executable
        run(f'"{python}" -m pip install -U openmim==0.3.7', desc="Installing openmim", errdesc="Couldn't install openmim")
        run(f'"{python}" -m mim install mmcv-full==1.7.1', desc=f"Installing mmcv-full", errdesc=f"Couldn't install mmcv-full")
        run(f'"{python}" -m pip install mmdet==2.28.2', desc=f"Installing mmdet", errdesc=f"Couldn't install mmdet")

    if (len(list_models(dd_models_path)) == 0):
        print("No detection models found, downloading...")
        bbox_path = os.path.join(dd_models_path, "bbox")
        segm_path = os.path.join(dd_models_path, "segm")
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/bbox/mmdet_anime-face_yolov3.pth", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/bbox/mmdet_anime-face_yolov3.py", bbox_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/resolve/main/mmdet/segm/mmdet_dd-person_mask2former.pth", segm_path)
        load_file_from_url("https://huggingface.co/dustysys/ddetailer/raw/main/mmdet/segm/mmdet_dd-person_mask2former.py", segm_path)

startup()

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

class Script(scripts.Script):
    def title(self):
        return "ddetailer + sdupscale"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        import modules.ui

        sample_list = [x.name for x in shared.list_samplers()]
        sample_list = [x for x in sample_list if x not in ['PLMS','UniPC','DDIM']]
        sample_list.insert(0,"Original")
        model_list = list_models(dd_models_path)
        model_list.insert(0, "None")
        ret = []
        
        with gr.Group():
            control_net_info = gr.HTML('<br><p style="margin-bottom:0.75em">T2I Control Net random image process</p>')
            disable_random_control_net = gr.Checkbox(label='Disable Random Controlnet', value=True, visible=True)
            cn_models_num = shared.opts.data.get("control_net_max_models_num", 1)
            image_detectors = []
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
            disable_upscaler = gr.Checkbox(label='Disable Upscaler', value=False, visible=True)
            upscaler_sample = gr.Dropdown(label='Upscaler Sampling', choices=sample_list, value=sample_list[0], visible=True, type="value")
            upscaler_index = gr.Dropdown(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[-1].name, type="index")
            scalevalue = gr.Slider(minimum=1, maximum=16, step=0.5, label='Resize', value=2)
            overlap = gr.Slider(minimum=0, maximum=256, step=32, label='Tile overlap', value=32)
            rewidth = gr.Slider(minimum=0, maximum=1024, step=64, label='Width', value=512)
            reheight = gr.Slider(minimum=0, maximum=1024, step=64, label='Height', value=512)
            denoising_strength = gr.Slider(minimum=0, maximum=1.0, step=0.01, label='Denoising strength', value=0.1)

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
            disable_detailer = gr.Checkbox(label='Disable DDetailer', value=False, visible=True)
            detailer_sample = gr.Dropdown(label='Detailer Sampling', choices=sample_list, value=sample_list[0], visible=True, type="value")
            if not is_img2img:
                with gr.Row():
                    dd_prompt = gr.Textbox(label="dd_prompt", elem_id="t2i_dd_prompt", show_label=False, lines=3, placeholder="Ddetailer Prompt")

                with gr.Row():
                    dd_neg_prompt = gr.Textbox(label="dd_neg_prompt", elem_id="t2i_dd_neg_prompt", show_label=False, lines=2, placeholder="Ddetailer Negative prompt")

            with gr.Row():
                dd_model_a = gr.Dropdown(label="Primary detection model (A)", choices=model_list,value = model_list[2], visible=True, type="value")
            
            with gr.Row():
                dd_conf_a = gr.Slider(label='Detection confidence threshold % (A)', minimum=0, maximum=100, step=1, value=30, visible=True)
                dd_dilation_factor_a = gr.Slider(label='Dilation factor (A)', minimum=0, maximum=255, step=1, value=4, visible=True)

            with gr.Row():
                dd_offset_x_a = gr.Slider(label='X offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=True)
                dd_offset_y_a = gr.Slider(label='Y offset (A)', minimum=-200, maximum=200, step=1, value=0, visible=True)
            
            with gr.Row():
                dd_bitwise_op = gr.Radio(label='Bitwise operation', choices=['None', 'A&B', 'A-B'], value="A&B", visible=True)  
        
            br = gr.HTML("<br>")
            
            with gr.Row():
                dd_model_b = gr.Dropdown(label="Secondary detection model (B) (optional)", choices=model_list,value = model_list[1], visible =True, type="value")

            with gr.Row():
                dd_conf_b = gr.Slider(label='Detection confidence threshold % (B)', minimum=0, maximum=100, step=1, value=30, visible=True)
                dd_dilation_factor_b = gr.Slider(label='Dilation factor (B)', minimum=0, maximum=255, step=1, value=4, visible=True)
            
            with gr.Row():
                dd_offset_x_b = gr.Slider(label='X offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=True)
                dd_offset_y_b = gr.Slider(label='Y offset (B)', minimum=-200, maximum=200, step=1, value=0, visible=True)
        
        with gr.Group():
            with gr.Row():
                dd_mask_blur = gr.Slider(label='Mask blur ', minimum=0, maximum=64, step=1, value=4, visible=(not is_img2img))
                dd_denoising_strength = gr.Slider(label='Denoising strength (Inpaint)', minimum=0.0, maximum=1.0, step=0.01, value=0.4, visible=(not is_img2img))
            
            with gr.Row():
                dd_inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution ', value=True, visible = (not is_img2img))
                dd_inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels ', minimum=0, maximum=256, step=4, value=32, visible=(not is_img2img))

        disable_detailer.change(
            lambda disable,modelname_a,modelname_b:{
                detailer_sample:gr_show((not disable)),
                dd_prompt:gr_show((not disable)),
                dd_neg_prompt:gr_show((not disable)),
                dd_model_a:gr_show((not disable)),
                dd_conf_a:gr_show((not disable) and (modelname_a != 'None')),
                dd_dilation_factor_a:gr_show((not disable) and (modelname_a != 'None')),
                dd_offset_x_a:gr_show((not disable) and (modelname_a != 'None')),
                dd_offset_y_a:gr_show((not disable) and (modelname_a != 'None')),
                dd_bitwise_op:gr_show((not disable) and (modelname_b != 'None')),
                dd_model_b:gr_show((not disable)),
                dd_conf_b:gr_show((not disable) and (modelname_b != 'None')),
                dd_dilation_factor_b:gr_show((not disable) and (modelname_b != 'None')),
                dd_offset_x_b:gr_show((not disable) and (modelname_b != 'None')),
                dd_offset_y_b:gr_show((not disable) and (modelname_b != 'None')),
                dd_mask_blur:gr_show((not disable)),
                dd_denoising_strength:gr_show((not disable)),
                dd_inpaint_full_res:gr_show((not disable)),
                dd_inpaint_full_res_padding:gr_show((not disable))
            },
            inputs=[disable_detailer,dd_model_a,dd_model_b],
            outputs=[
                detailer_sample,
                dd_prompt,
                dd_neg_prompt,
                dd_model_a,
                dd_conf_a,
                dd_dilation_factor_a,
                dd_offset_x_a,
                dd_offset_y_a,
                dd_bitwise_op,
                dd_model_b,
                dd_conf_b,
                dd_dilation_factor_b,
                dd_offset_x_b,
                dd_offset_y_b,
                dd_mask_blur,
                dd_denoising_strength,
                dd_inpaint_full_res,
                dd_inpaint_full_res_padding
            ]
        )
        
        dd_model_a.change(
            lambda modelname: {
                dd_model_b:gr_show( modelname != "None" ),
                dd_conf_a:gr_show( modelname != "None" ),
                dd_dilation_factor_a:gr_show( modelname != "None"),
                dd_offset_x_a:gr_show( modelname != "None" ),
                dd_offset_y_a:gr_show( modelname != "None" )

            },
            inputs= [dd_model_a],
            outputs =[dd_model_b, dd_conf_a, dd_dilation_factor_a, dd_offset_x_a, dd_offset_y_a]
        )

        dd_model_b.change(
            lambda modelname: {
                dd_bitwise_op:gr_show( modelname != "None" ),
                dd_conf_b:gr_show( modelname != "None" ),
                dd_dilation_factor_b:gr_show( modelname != "None"),
                dd_offset_x_b:gr_show( modelname != "None" ),
                dd_offset_y_b:gr_show( modelname != "None" )
            },
            inputs= [dd_model_b],
            outputs =[dd_bitwise_op, dd_conf_b, dd_dilation_factor_b, dd_offset_x_b, dd_offset_y_b]
        )
        
        ret += [disable_random_control_net, disable_upscaler, disable_detailer, enable_script_names, scalevalue, upscaler_sample, detailer_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength]
        ret += [dd_model_a, 
                dd_conf_a, dd_dilation_factor_a,
                dd_offset_x_a, dd_offset_y_a,
                dd_bitwise_op, 
                br,
                dd_model_b,
                dd_conf_b, dd_dilation_factor_b,
                dd_offset_x_b, dd_offset_y_b,  
                dd_mask_blur, dd_denoising_strength,
                dd_inpaint_full_res, dd_inpaint_full_res_padding
        ]
        ret += [dd_prompt, dd_neg_prompt]
        ret += image_detectors

        return ret

    def run(self, p, disable_random_control_net, disable_upscaler, disable_detailer, enable_script_names, 
                     scalevalue, upscaler_sample, detailer_sample, overlap, upscaler_index, rewidth, reheight, denoising_strength,
                     dd_model_a, 
                     dd_conf_a, dd_dilation_factor_a,
                     dd_offset_x_a, dd_offset_y_a,
                     dd_bitwise_op, 
                     br,
                     dd_model_b,
                     dd_conf_b, dd_dilation_factor_b,
                     dd_offset_x_b, dd_offset_y_b,  
                     dd_mask_blur, dd_denoising_strength,
                     dd_inpaint_full_res, dd_inpaint_full_res_padding,
                     dd_prompt, dd_neg_prompt, *args):
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
                denoising_strength = dd_denoising_strength,
                mask = None,
                mask_blur= dd_mask_blur,
                inpainting_fill = 1,
                inpaint_full_res = dd_inpaint_full_res,
                inpaint_full_res_padding= dd_inpaint_full_res_padding,
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
            mask_blur=dd_mask_blur,
            inpainting_fill=1,
            resize_mode=0,
            denoising_strength=denoising_strength,
            inpaint_full_res=dd_inpaint_full_res,
            inpaint_full_res_padding=dd_inpaint_full_res_padding,
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
                controlnet_image_files[con_n] = files.copy()
        
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
        state.job_count += ddetail_count
        for n in range(ddetail_count):
            devices.torch_gc()
            start_seed = seed + n
            print(f"Processing initial image for output generation {n + 1} (T2I).")
            p_txt.seed = start_seed
            p_txt.scripts.scripts = t2i_scripts
            p_txt.scripts.alwayson_scripts = t2i_scripts_always
            if not disable_random_control_net:
                for con_n, conet in enumerate(controlnet_args):
                    if len(controlnet_image_files[con_n]) > 0:
                        cn_image = Image.open(choice(controlnet_image_files[con_n]))
                        cn_np = np.array(cn_image)
                        if cn_image.mode == 'RGB':
                            cn_np = np.concatenate([cn_np, 255*np.ones((cn_np.shape[0], cn_np.shape[1], 1), dtype=np.uint8)], axis=-1)
                        cn_np_image = cn_np[:,:,:3].copy()
                        cn_np_mask = cn_np
                        cn_np_mask[:,:,:3] = 0
                        conet.image = {'image':cn_np_image,'mask':cn_np_mask}
            processed = processing.process_images(p_txt)
            initial_info.append(processed.info)
            posi, nega = processed.all_prompts[0], processed.all_negative_prompts[0]
            initial_prompt.append(posi)
            initial_negative.append(nega)
            p.prompt = posi if dd_prompt == '' else dd_prompt
            p.negative_prompt = nega if dd_neg_prompt == '' else dd_neg_prompt
            init_image = processed.images[0]
            
            output_images.append(init_image)
            masks_a = []
            if not disable_detailer:
                if (dd_model_a != "None"):
                    label_a = "A"
                    if (dd_model_b != "None" and dd_bitwise_op != "None"):
                        label_a = dd_bitwise_op
                    results_a = inference(init_image, dd_model_a, dd_conf_a/100.0, label_a)
                    masks_a = create_segmasks(results_a)
                    masks_a = dilate_masks(masks_a, dd_dilation_factor_a, 1)
                    masks_a = offset_masks(masks_a,dd_offset_x_a, dd_offset_y_a)
                    if (dd_model_b != "None" and dd_bitwise_op != "None"):
                        label_b = "B"
                        results_b = inference(init_image, dd_model_b, dd_conf_b/100.0, label_b)
                        masks_b = create_segmasks(results_b)
                        masks_b = dilate_masks(masks_b, dd_dilation_factor_b, 1)
                        masks_b = offset_masks(masks_b,dd_offset_x_b, dd_offset_y_b)
                        if (len(masks_b) > 0):
                            combined_mask_b = combine_masks(masks_b)
                            for i in reversed(range(len(masks_a))):
                                if (dd_bitwise_op == "A&B"):
                                    masks_a[i] = bitwise_and_masks(masks_a[i], combined_mask_b)
                                elif (dd_bitwise_op == "A-B"):
                                    masks_a[i] = subtract_masks(masks_a[i], combined_mask_b)
                                if (is_allblack(masks_a[i])):
                                    del masks_a[i]
                                    for result in results_a:
                                        del result[i]
                                        
                        else:
                            print("No model B detections to overlap with model A masks")
                            results_a = []
                            masks_a = []
                    
                    if (len(masks_a) > 0):
                        results_a = update_result_masks(results_a, masks_a)
                        gen_count = len(masks_a)
                        state.job_count += gen_count
                        print(f"Processing {gen_count} model {label_a} detections for output generation {n + 1} (I2I).")
                        p.seed = start_seed
                        p.init_images = [init_image]

                        for i in range(gen_count):
                            p.image_mask = masks_a[i]
                            
                            p.scripts.scripts = i2i_scripts
                            p.scripts.alwayson_scripts = i2i_scripts_always
                            processed = processing.process_images(p)
                            p.seed = processed.seed + 1
                            p.init_images = processed.images
                        
                        if (gen_count > 0):
                            output_images[n] = processed.images[0]
    
                    else: 
                        print(f"No model {label_a} detections for output generation {n} with current settings.")
                    
            state.job = f"Generation {n + 1} out of {state.job_count} DDetailer"
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

                work_results = []
                for i in range(batch_count):
                    p2.batch_size = batch_size
                    p2.init_images = work[i*batch_size:(i+1)*batch_size]

                    state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                    p2.scripts.scripts = i2i_scripts
                    p2.scripts.alwayson_scripts = i2i_scripts_always
                    processed = processing.process_images(p2)

                    p2.seed = processed.seed + 1
                    work_results += processed.images

                image_index = 0
                for y, h, row in grid.tiles:
                    for tiledata in row:
                        tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (rewidth, reheight))
                        image_index += 1
                output_images[n] = images.combine_grid(grid)
            result_images.append(output_images[n])
            images.save_image(result_images[-1], p.outpath_samples, "", start_seed, initial_prompt[n], opts.samples_format, info=initial_info[n], p=p_txt)
        p_txt.scripts.scripts = original_scripts
        return Processed(p_txt, result_images, start_seed, initial_info[0], all_prompts=initial_prompt, all_negative_prompts=initial_negative, infotexts=initial_info)
        
def modeldataset(model_shortname):
    path = modelpath(model_shortname)
    if ("mmdet" in path and "segm" in path):
        dataset = 'coco'
    else:
        dataset = 'bbox'
    return dataset

def modelpath(model_shortname):
    model_list = modelloader.load_models(model_path=dd_models_path, ext_filter=[".pth"])
    model_h = model_shortname.split("[")[-1].split("]")[0]
    for path in model_list:
        if ( model_hash(path) == model_h):
            return path

def update_result_masks(results, masks):
    for i in range(len(masks)):
        boolmask = np.array(masks[i], dtype=bool)
        results[2][i] = boolmask
    return results

def is_allblack(mask):
    cv2_mask = np.array(mask)
    return cv2.countNonZero(cv2_mask) == 0

def bitwise_and_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def subtract_masks(mask1, mask2):
    cv2_mask1 = np.array(mask1)
    cv2_mask2 = np.array(mask2)
    cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
    mask = Image.fromarray(cv2_mask)
    return mask

def dilate_masks(masks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return masks
    dilated_masks = []
    kernel = np.ones((dilation_factor,dilation_factor), np.uint8)
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        dilated_masks.append(Image.fromarray(dilated_mask))
    return dilated_masks

def offset_masks(masks, offset_x, offset_y):
    if (offset_x == 0 and offset_y == 0):
        return masks
    offset_masks = []
    for i in range(len(masks)):
        cv2_mask = np.array(masks[i])
        offset_mask = cv2_mask.copy()
        offset_mask = np.roll(offset_mask, -offset_y, axis=0)
        offset_mask = np.roll(offset_mask, offset_x, axis=1)
        
        offset_masks.append(Image.fromarray(offset_mask))
    return offset_masks

def combine_masks(masks):
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
    
    combined_mask = Image.fromarray(combined_cv2_mask)
    return combined_mask

def create_segmasks(results):
    segms = results[2]
    segmasks = []
    for i in range(len(segms)):
        cv2_mask = segms[i].astype(np.uint8) * 255
        mask = Image.fromarray(cv2_mask)
        segmasks.append(mask)

    return segmasks

import mmcv
from mmdet.core import get_classes
from mmdet.apis import (inference_detector,
                        init_detector)

def get_device():
    device_id = shared.cmd_opts.device_id
    if device_id is not None:
        cuda_device = f"cuda:{device_id}"
    else:
        cuda_device = "cpu"
    return cuda_device

def inference(image, modelname, conf_thres, label):
    path = modelpath(modelname)
    if ( "mmdet" in path and "bbox" in path ):
        results = inference_mmdet_bbox(image, modelname, conf_thres, label)
    elif ( "mmdet" in path and "segm" in path):
        results = inference_mmdet_segm(image, modelname, conf_thres, label)
    return results

def inference_mmdet_segm(image, modelname, conf_thres, label):
    model_checkpoint = modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = get_device()
    model = init_detector(model_config, model_checkpoint, device=model_device)
    mmdet_results = inference_detector(model, np.array(image))
    bbox_results, segm_results = mmdet_results
    dataset = modeldataset(modelname)
    classes = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_results)
    ]
    n,m = bbox_results[0].shape
    if (n == 0):
        return [[],[],[]]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_results)
    segms = mmcv.concat_list(segm_results)
    filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label + "-" + classes[labels[i]])
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results

def inference_mmdet_bbox(image, modelname, conf_thres, label):
    model_checkpoint = modelpath(modelname)
    model_config = os.path.splitext(model_checkpoint)[0] + ".py"
    model_device = get_device()
    model = init_detector(model_config, model_checkpoint, device=model_device)
    results = inference_detector(model, np.array(image))
    cv2_image = np.array(image)
    cv2_image = cv2_image[:, :, ::-1].copy()
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for (x0, y0, x1, y1, conf) in results[0]:
        cv2_mask = np.zeros((cv2_gray.shape), np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)
    
    n,m = results[0].shape
    if (n == 0):
        return [[],[],[]]
    bboxes = np.vstack(results[0])
    filter_inds = np.where(bboxes[:,-1] > conf_thres)[0]
    results = [[],[],[]]
    for i in filter_inds:
        results[0].append(label)
        results[1].append(bboxes[i])
        results[2].append(segms[i])

    return results