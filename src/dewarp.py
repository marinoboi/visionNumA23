import cv2
import numpy as np


def dewarp_page(img: np.ndarray, contours: np.ndarray, **kwargs) -> np.ndarray:
    """
    Dewarp and crop the image using the contours.
    :param img: Image to dewarp (not modified).
    :return: Dewarped image
    """
    return img
