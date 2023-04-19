import os
import numpy as np
import torch
import gc
import copy

from modules import shared
from modules.paths import models_path
from modules.safe import unsafe_torch_load, load
from modules.devices import device, torch_gc, cpu

from PIL import Image
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from segment_anything import SamPredictor, sam_model_registry
from scripts.dino import dino_predict_internal, clear_dino_cache

sam_model_cache = OrderedDict()
sam_model_dir = os.path.join(models_path, "sam")

def sam_model_list():
    return [x for x in os.listdir(sam_model_dir) if x.endswith('.pth')]

def load_sam_model(sam_checkpoint):
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    torch.load = load
    return sam

def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()
    
def clear_cache():
    clear_sam_cache()
    clear_dino_cache()

def dilate_mask(mask, dilation):
    x, y = np.meshgrid(np.arange(dilation), np.arange(dilation))
    center = dilation // 2
    dilation_kernel = ((x - center) ** 2 + (y - center) ** 2 <= center ** 2).astype(np.uint8)
    
    dilated_bin_img = binary_dilation(mask, dilation_kernel)
    
    return dilated_bin_img.astype(np.uint8) * 255

def init_sam_model(sam_model_name):
    print('Initializing SAM')
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if(shared.cmd_opts.lowvram):
            sam.to(device=device)
        return sam
    elif sam_model_name in sam_model_list():
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        Exception(f'{sam_model_name} not found, please download model to models/sam')

def sam_predict(sam_model_name, dino_model_name, image, image_np, image_np_rgb, dino_text, dino_box_threshold, dilation, sam_level):
    print('Start SAM Processing')
    
    assert dino_text, 'Please input dino text'
    
    boxes = dino_predict_internal(image, dino_model_name, dino_text, dino_box_threshold)
    
    if boxes.shape[0] < 1: return None
    
    sam = init_sam_model(sam_model_name)
    
    print(f'Running SAM Inference {image_np_rgb.shape}')
    predictor = SamPredictor(sam)
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = True
    )
    
    masks = masks.permute(1,0,2,3).cpu().numpy()
    
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    clear_sam_cache()
    
    return dilate_mask(Image.fromarray(np.any(masks[sam_level], axis=0)),dilation)