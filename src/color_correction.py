import cv2
import numpy as np


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


def remove_fingers(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    :param img: Image to remove the thumb from.
    :return: Image where the thumb is now white pixels.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin_value = np.array([0, 70, 80], dtype="uint8")
    upper_skin_value = np.array([30, 255, 255], dtype="uint8")

    skin_mask = cv2.inRange(img_hsv, lower_skin_value, upper_skin_value)
    # remove noise from the mask by applying a series of erosions and dilations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # fill the mask with white
    white = 255 * np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    fingers_mask = cv2.bitwise_and(white, white, mask=skin_mask)

    return fingers_mask
