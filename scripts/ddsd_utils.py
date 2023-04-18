import re
import numpy as np
import cv2
import gc
from PIL import Image
from scripts.sam import sam_predict
from modules.devices import torch_gc

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

def dino_detect_from_prompt(prompt:str, detailer_sam_model, detailer_dino_model, init_image):
    image_np_zero = np.array(init_image.convert('L'))
    image_np_zero[:,:] = 0
    image_np = np.array(init_image)
    image_np_rgb = image_np[:,:,:3].copy()
    image_set = (init_image, image_np, image_np_rgb, image_np_zero)
    model_set = (detailer_sam_model, detailer_dino_model)
    result = dino_prompt_detector(prompt, model_set, image_set)
    if np.array_equal(result, image_np_zero): return None
    return Image.fromarray(result)

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
                                   try_convert(dilation.strip(), int, 4, 0, 128), 
                                   try_convert(sam_level.strip(), int, 1, 0, 2))
                if left is None: left = image_set[3].copy()
            else:
                left = result_group[left]
        if not isinstance(right, np.ndarray):
            match = token_match.match(right)
            if match is None:
                dino_text, sam_level, dino_box_threshold, dilation = prompt_spliter(right, ':', 4)
                right = sam_predict(model_set[0], model_set[1], image_set[0], image_set[1], image_set[2], dino_text, 
                                   try_convert(dino_box_threshold.strip(), float, 0.3, 0, 1.0), 
                                   try_convert(dilation.strip(), int, 4, 0, 128), 
                                   try_convert(sam_level.strip(), int, 1, 0, 2))
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
                                   try_convert(dilation.strip(), int, 4, 0, 128), 
                                   try_convert(sam_level.strip(), int, 1, 0, 2))
    if target is None: return image_set[3].copy()
    return target