from __future__ import annotations

import os
import torch

import mediapipe as mp
import numpy as np

from PIL import Image, ImageDraw
from ultralytics import YOLO

from modules import safe
from modules.shared import cmd_opts
from modules.paths import models_path

yolo_models_path = os.path.join(models_path, 'yolo')

def mediapipe_face_detect(image, model_type, confidence):
    width, height = image.size
    image_np = np.array(image)
    
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=model_type, min_detection_confidence=confidence) as face_detector:
        predictor = face_detector.process(image_np)
    
    if predictor.detections is None: return None
    
    bboxes = []
    for detection in predictor.detections:
        
        bbox = detection.location_data.relative_bounding_box
        x1 = bbox.xmin * width
        y1 = bbox.ymin * height
        x2 = x1 + bbox.width * width
        y2 = y1 + bbox.height * height
        bboxes.append([x1,y1,x2,y2])
    
    return create_mask_from_bbox(image, bboxes)

def ultralytics_predict(image, model_type, confidence, device):
    models = [os.path.join(yolo_models_path,x) for x in os.listdir(yolo_models_path) if (x.endswith('.pt') or x.endswith('.pth')) and os.path.splitext(os.path.basename(x))[0].upper() == model_type]
    if len(models) == 0: return None
    model = YOLO(models[0])
    predictor = model(image, conf=confidence, show_labels=False, device=device)
    bboxes = predictor[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0: return None
    bboxes = bboxes.tolist()
    return create_mask_from_bbox(image, bboxes)

def create_mask_from_bbox(image, bboxes):
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        draw.rectangle(bbox, fill=255)
    return np.array(mask)
        
def bs_model(image, model_type, confidence):
    print(model_type, confidence)
    image = Image.fromarray(image)
    orig = torch.load
    torch.load = safe.unsafe_torch_load
    if model_type == 'FACE_MEDIA_FULL':
        mask = mediapipe_face_detect(image, 1, confidence)
    elif model_type == 'FACE_MEDIA_SHORT':
        mask = mediapipe_face_detect(image, 0, confidence)
    else:
        device = ''
        if getattr(cmd_opts, 'lowvram', False) or getattr(cmd_opts, 'medvram', False):
            device = 'cpu'
        mask = ultralytics_predict(image, model_type, confidence, device)
    torch.load = orig
    return mask