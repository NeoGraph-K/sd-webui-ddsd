import numpy as np
import cv2
from PIL import Image

def combine_masks(masks, combine_masks_option):
    if len(masks) == 0: return None
    initial_cv2_mask = np.array(masks[0])
    combined_cv2_mask = initial_cv2_mask
    for i in range(1, len(masks)):
        cv2_mask = np.array(masks[i])
        if combine_masks_option == 'AND':
            combined_cv2_mask = cv2.bitwise_and(combined_cv2_mask, cv2_mask)
        if combine_masks_option == 'OR':
            combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
    return Image.fromarray(combined_cv2_mask)