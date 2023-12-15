import logging

import cv2
import numpy as np


def fix_orientation(img: np.ndarray, lines: np.ndarray, **kwargs) -> np.ndarray:
    """
    Fix the page orientation using lines of text segments.
    :param img: Image to correct
    :return: Corrected image
    """
    if len(lines) == 0:
        # No lines detected, can't fix orientation.
        return img

    # +1e-9 to avoid division by zero
    line_angles = np.arctan((lines[:, 0, 1] - lines[:, 0, 3]) / (lines[:, 0, 0] - lines[:, 0, 2] + 1e-9))
    avg_angle = np.degrees(np.mean(line_angles))
    logging.info(f"Page orientation: {avg_angle:.1f} deg")

    img_center = img.shape[1] / 2, img.shape[0] / 2
    mat_rot = cv2.getRotationMatrix2D(img_center, avg_angle, 1)
    img_rotated = cv2.warpAffine(img, mat_rot, img.shape[1::-1], flags=cv2.INTER_CUBIC)

    return img_rotated
