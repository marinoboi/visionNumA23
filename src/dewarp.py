import cv2
import numpy as np


def dewarp_page(img: np.ndarray, contours: np.ndarray, text_lines: np.ndarray) -> np.ndarray:
    """
    Dewarp and crop the image using the contours and the text lines.
    This includes correcting the orientation if slightly rotated.
    :param img: Image to dewarp (not modified).
    :return: Dewarped image
    """
    return img
