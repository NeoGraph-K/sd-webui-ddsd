import os
import gc
import torch
from collections import OrderedDict

from modules import shared
from modules.devices import device, torch_gc, cpu

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from modules.paths import models_path
from groundingdino.util.utils import clean_state_dict

dino_model_cache = OrderedDict()
grounding_models_dir = os.path.join(models_path, "grounding")

def dino_model_list():
    return [x for x in os.listdir(grounding_models_dir) if x.endswith('.pth')]

def dino_config_file_name(dino_model_name:str):
    return dino_model_name.replace('.pth','.py')

def clear_dino_cache():
    dino_model_cache.clear()
    gc.collect()
    torch_gc()

def load_dino_model(dino_checkpoint):
    print(f"Initializing GroundingDINO {dino_checkpoint}")
    if dino_checkpoint in dino_model_cache:
        dino = dino_model_cache[dino_checkpoint]
        if shared.cmd_opts.lowvram:
            dino.to(device=device)
    else:
        clear_dino_cache()
        args = SLConfig.fromfile(os.path.join(grounding_models_dir,dino_config_file_name(dino_checkpoint)))
        dino = build_model(args)
        checkpoint = torch.load(os.path.join(grounding_models_dir,dino_checkpoint),map_location='cpu')
        dino.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        dino.to(device=device)
        dino_model_cache[dino_checkpoint] = dino
    dino.eval()
    return dino


def load_dino_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def get_grounding_output(model, image, caption, box_threshold):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    if shared.cmd_opts.lowvram:
        model.to(cpu)
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    return boxes_filt.cpu()


def dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold):
    print("Running GroundingDINO Inference")
    dino_image = load_dino_image(input_image.convert("RGB"))
    dino_model = load_dino_model(dino_model_name)

    boxes_filt = get_grounding_output(
        dino_model, dino_image, text_prompt, box_threshold
    )

    H, W = input_image.size[1], input_image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    gc.collect()
    torch_gc()
    return boxes_filt
