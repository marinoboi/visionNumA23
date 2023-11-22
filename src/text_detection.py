import logging

import cv2
import numpy as np


def detect_text_lines(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Detect lines of text segments in an image
    :param img: Image to detect text for (not modified).
    :return: NxMx2 array of lines
    """
    # Scale image to accelerate processing
    scale = 512.0 / img.shape[0]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_size = np.uint32(np.array(img_gray.shape[::-1]) * scale)
    img_small = cv2.resize(img_gray, new_size)

    # Blur with horizontal kernel to remove gradient between characters on the same line
    img_blur = cv2.blur(img_small, ksize=(20, 3))
    # Edge detect lines (most edges will be horizontal)
    img_edges = cv2.Canny(img_blur, 20, 100)

    # Find line segments in edge detection
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(img_edges)[0]
    if lines is None:
        lines = np.zeros((0, 1, 4))
    # Remove lines with extreme angles (too vertical, >15°)
    line_angles = np.abs(np.arctan((lines[:, 0, 1] - lines[:, 0, 3]) /
                                   (lines[:, 0, 0] - lines[:, 0, 2])))
    lines = lines[line_angles < np.radians(30), :, :]
    logging.info(f"{lines.shape[0]} text line segments detected")

    # cv2.imshow("Blurred", img_blur)
    # cv2.imshow("Canny edges", img_edges)
    # img_lines = lsd.drawSegments(img_small, lines)
    # cv2.imshow("Lines overlay", img_lines)

    lines /= scale

    return lines
