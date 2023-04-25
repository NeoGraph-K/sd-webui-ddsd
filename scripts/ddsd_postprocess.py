import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pillow_lut import load_cube_file
from scipy.interpolate import UnivariateSpline

from modules.paths import models_path

lut_model_dir = os.path.join(models_path, "lut")

def lut_model_list():
    return [x for x in os.listdir(lut_model_dir) if x.lower().endswith('.cube')]

def saturation_image(image:Image.Image, strength:float) -> Image.Image: # 채도 조절
    return ImageEnhance.Color(image).enhance(strength)
def sharpening_image(image:Image.Image, radius:float, percent:int, threshold:float) -> Image.Image: # 선명도 조절
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
def gaussian_blur_image(image:Image.Image, radius:float) -> Image.Image: # 흐림도 조절
    return image.filter(ImageFilter.GaussianBlur(radius=radius))
def brightness_image(image:Image.Image, strength:float) -> Image.Image: # 밝기 조절
    return ImageEnhance.Brightness(image).enhance(strength)
def color_image(image:Image.Image, strength:float) -> Image.Image: # 색조 조절
    return ImageEnhance.Color(image).enhance(strength)
def contrast_image(image:Image.Image, strength:float) -> Image.Image: # 대비 조절
    return ImageEnhance.Contrast(image).enhance(strength)
def color_extraction_image(image:Image.Image, lower:tuple[int,int,int], upper:tuple[int,int,int], strength:float) -> Image.Image: # 색상 추출 및 변화
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_np, lower, upper)
    image_np = image_np.astype(np.float64)
    image_np[mask != 0] *= strength
    image_np = image_np.astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_HSV2RGB))
def hue_image(image:Image.Image, strength:float) -> Image.Image: # Hue 조절
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    image_np[..., 0] = (image_np[..., 0] + strength * 180) % 180
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_HSV2RGB))
def inversion_image(image:Image.Image) -> Image.Image: # 반전
    return ImageOps.invert(image)
def bilateral_image(image:Image.Image, sigmaC:int, sigmaS:int) -> Image.Image: # 양방향 필터
    image_np = np.array(image)
    return Image.fromarray(cv2.bilateralFilter(image_np, -1, sigmaC, sigmaS))
def color_tint_lut_image(image:Image.Image, lut_file:str) -> Image.Image: # 색상 조절
    lut = load_cube_file(os.path.join(lut_model_dir, lut_file))
    return image.filter(lut)
def color_tint_type_image(image:Image.Image, type:str) -> Image.Image: # 색온도 조절(Warm, Cool)
    increase = UnivariateSpline([0,64,128,192,256],[0,70,140,210,256])(range(256))
    decrease = UnivariateSpline([0,64,128,192,256],[0,30,80,120,192])(range(256))
    image_np = np.array(image)
    r, g, b = cv2.split(image_np)
    r = cv2.LUT(r, increase if type == 'warm' else decrease).astype(np.uint8)
    b = cv2.LUT(b, decrease if type == 'warm' else increase).astype(np.uint8)
    image_np = cv2.merge((r, g, b))
    h, s, v = cv2.split(cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV))
    s = cv2.LUT(s, increase if type == 'warm' else decrease).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB))

def ddsd_postprocess(image:Image.Image, pptype:str,
                     saturation_strength:float,
                     sharpening_radius:float, sharpening_percent:int, sharpening_threshold:float,
                     gaussian_blur_radius:float,
                     brightness_strength:float,
                     color_strength:float,
                     contrast_strength:float,
                     #color_extraction_lower:tuple[int,int,int], color_extraction_upper:tuple[int,int,int], color_extraction_strength:float,
                     hue_strength:float,
                     bilateral_sigmaC:int, bilateral_sigmaS:int,
                     color_tint_lut_file:str,
                     color_tint_type_name:str) -> Image.Image:
    if pptype == 'saturation': return saturation_image(image, saturation_strength)
    if pptype == 'sharpening': return sharpening_image(image, sharpening_radius, sharpening_percent, sharpening_threshold)
    if pptype == 'gaussian blur': return gaussian_blur_image(image, gaussian_blur_radius)
    if pptype == 'brightness': return brightness_image(image, brightness_strength)
    if pptype == 'color': return color_image(image, color_strength)
    if pptype == 'contrast': return contrast_image(image, contrast_strength)
    #if pptype == 'color extraction': return color_extraction_image(image, color_extraction_lower, color_extraction_upper, color_extraction_strength)
    if pptype == 'hue': return hue_image(hue_strength)
    if pptype == 'inversion': return inversion_image(image)
    if pptype == 'bilateral': return bilateral_image(image, bilateral_sigmaC, bilateral_sigmaS)
    if pptype == 'color tint(type)': return color_tint_type_image(image, color_tint_type_name)
    if pptype == 'color tint(lut)': return color_tint_lut_image(image, color_tint_lut_file)
    return image