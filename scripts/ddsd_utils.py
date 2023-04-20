import re
import numpy as np
import cv2
import gc
from scripts.sam import sam_predict
from modules.devices import torch_gc
from skimage import measure

from modules.processing import StableDiffusionProcessingImg2Img

token_split = re.compile(r"(AND|OR|NOR|XOR|NAND)")
token_first = re.compile(r'\(([^()]+)\)')
token_match = re.compile(r'(\d+)GROUPMASK')

def try_convert(data, type, default, min, max):
    try:
        convert = type(data)
        if convert < min: return min
        if convert > max: return max
        return convert
    except (ValueError, TypeError):
        return default
    
def prompt_spliter(prompt:str, split_token:str, count:int):
    spliter = prompt.split(split_token)
    while len(spliter) < count:
        spliter.append('')
    return spliter[:count]

def combine_masks(mask, combine_masks_option, mask2):
    if combine_masks_option == 'AND': return cv2.bitwise_and(mask, mask2)
    if combine_masks_option == 'OR': return cv2.bitwise_or(mask, mask2)
    if combine_masks_option == 'XOR': return cv2.bitwise_xor(mask, mask2)
    if combine_masks_option == 'NOR': return cv2.bitwise_not(cv2.bitwise_or(mask, mask2))
    if combine_masks_option == 'NAND': return cv2.bitwise_not(cv2.bitwise_and(mask,mask2))

def dino_detect_from_prompt(prompt:str, detailer_sam_model, detailer_dino_model, init_image, disable_mask_paint_mode, inpaint_mask_mode, image_mask):
    gc.collect()
    torch_gc()
    image_np_zero = np.array(init_image.convert('L'))
    image_np_zero[:,:] = 0
    image_np = np.array(init_image)
    image_np_rgb = image_np[:,:,:3].copy()
    image_set = (init_image, image_np, image_np_rgb, image_np_zero)
    model_set = (detailer_sam_model, detailer_dino_model)
    result = dino_prompt_detector(prompt, model_set, image_set)
    if np.array_equal(result, image_np_zero): return None
    if not disable_mask_paint_mode: return result
    image_mask = np.array(image_mask.resize((result.shape[1],result.shape[0])).convert('L'))
    image_mask = np.resize(image_mask, result.shape)
    if inpaint_mask_mode == 'Inner': return cv2.bitwise_and(result, image_mask)
    if inpaint_mask_mode == 'Outer': return cv2.bitwise_and(result, cv2.bitwise_not(image_mask))
    return None
    

def dino_prompt_detector(prompt:str, model_set, image_set):
    find = token_first.search(prompt)
    result_group = {}
    result_count = 0
    while find:
        result_group[f' {result_count}GROUPMASK '] = dino_prompt_detector(find.group(1), model_set, image_set)
        prompt = prompt.replace(find.group(), f' {result_count}GROUPMASK ')
        result_count += 1
        find = token_first.search(prompt)
        
    spliter = token_split.split(prompt)
    
    while len(spliter) > 1:
        left, operator, right = spliter[:3]
        if not isinstance(left, np.ndarray):
            match = token_match.match(left)
            if match is None:
                dino_text, sam_level, dino_box_threshold, dilation = prompt_spliter(left, ':', 4)
                left = sam_predict(model_set[0], model_set[1], image_set[0], image_set[1], image_set[2], dino_text, 
                                   try_convert(dino_box_threshold.strip(), float, 0.3, 0, 1.0), 
                                   try_convert(dilation.strip(), int, 8, 0, 128), 
                                   try_convert(sam_level.strip(), int, 0, 0, 2))
                if left is None: left = image_set[3].copy()
            else:
                left = result_group[left]
        if not isinstance(right, np.ndarray):
            match = token_match.match(right)
            if match is None:
                dino_text, sam_level, dino_box_threshold, dilation = prompt_spliter(right, ':', 4)
                right = sam_predict(model_set[0], model_set[1], image_set[0], image_set[1], image_set[2], dino_text, 
                                   try_convert(dino_box_threshold.strip(), float, 0.3, 0, 1.0), 
                                   try_convert(dilation.strip(), int, 8, 0, 128), 
                                   try_convert(sam_level.strip(), int, 0, 0, 2))
                if right is None: right = image_set[3].copy()
            else:
                right = result_group[right]
        spliter[:3] = [combine_masks(left, operator, right)]
        gc.collect()
        torch_gc()
    if isinstance(spliter[0], np.ndarray): return spliter[0]
    dino_text, sam_level, dino_box_threshold, dilation = prompt_spliter(spliter[0], ':', 4)
    target = sam_predict(model_set[0], model_set[1], image_set[0], image_set[1], image_set[2], dino_text, 
                                   try_convert(dino_box_threshold.strip(), float, 0.3, 0, 1.0), 
                                   try_convert(dilation.strip(), int, 8, 0, 128), 
                                   try_convert(sam_level.strip(), int, 0, 0, 2))
    if target is None: return image_set[3].copy()
    return target

def mask_spliter_and_remover(mask, area):
    gc.collect()
    torch_gc()
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    
    for r in regions:
        if r.area < area:
            for coord in r.coords:
                labels[coord[0], coord[1]] = 0
    
    num_labels = np.max(labels)
    
    label_images = []
    for x in range(num_labels):
        label_image = np.zeros_like(mask, dtype=np.uint8)
        label_image[labels == (x + 1)] = 255
        label_images.append(label_image)
    return label_images
    
def I2I_Generator_Create(p, i2i_sample, i2i_mask_blur, full_res_inpainting, inpainting_padding, init_image, denoise, cfg, steps, width, height, tiling, scripts, scripts_list, alwaysonscripts_list, script_args, positive, negative):
    i2i = StableDiffusionProcessingImg2Img(
                init_images = [init_image],
                resize_mode = 0,
                denoising_strength = 0,
                mask = None,
                mask_blur= i2i_mask_blur,
                inpainting_fill = 1,
                inpaint_full_res = full_res_inpainting,
                inpaint_full_res_padding= inpainting_padding,
                inpainting_mask_invert= 0,
                sd_model=p.sd_model,
                outpath_samples=p.outpath_samples,
                outpath_grids=p.outpath_grids,
                restore_faces=p.restore_faces,
                prompt='',
                negative_prompt='',
                styles=p.styles,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=i2i_sample,
                n_iter=1,
                batch_size=1,
                steps=steps,
                cfg_scale=cfg,
                width=width,
                height=height,
                tiling=tiling,
            )
    i2i.denoising_strength = denoise
    i2i.do_not_save_grid = True
    i2i.do_not_save_samples = True
    i2i.override_settings = {}
    i2i.override_settings_restore_afterwards = {}
    i2i.scripts = scripts
    i2i.scripts.scripts = scripts_list.copy()
    i2i.scripts.alwayson_scripts = alwaysonscripts_list.copy()
    i2i.script_args = script_args
    i2i.prompt = positive
    i2i.negative_prompt = negative
    
    return i2i