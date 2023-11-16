import cv2
import numpy as np


def detect_text_lines(img: np.ndarray) -> np.ndarray:
    """
    Detect text lines in an image
    :param img: Image to detect text for (not modified).
    :return: NxMx2 numpy array containing N lines of M points (X,Y) in the image.
    """
    return np.zeros((1, 1, 2), dtype=np.uint32)
