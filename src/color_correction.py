import cv2
import numpy as np
import math


def correct_colors(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    :param img: Image to correct colors for (not modified).
    :return: Image with corrected colors.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray_img
    blur_inverted_gray = cv2.GaussianBlur(inverted_gray, (121, 121), 0)
    blur_gray = 255 - blur_inverted_gray
    img_corrected = cv2.divide(gray_img, blur_gray, scale=256)

    return img_corrected
