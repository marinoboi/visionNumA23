import cv2
import numpy as np


def detect_contours(img: np.ndarray) -> np.ndarray:
    """
    Detect page contours in an image.
    :param img: Image to detect contours in.
    :return: Nx2 numpy array of points on the contour in the image.
    """
    # TODO
    return np.zeros((1, 2), dtype=np.uint32)
