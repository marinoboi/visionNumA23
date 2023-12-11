import numpy as np
import cv2
import logging

def crop_image_with_text(img: np.ndarray, lines: np.ndarray, crop_top_bottom:bool) -> np.ndarray:
    """
    Crop the image so that the distance between the text and the sides of the image is equal.
    :param img: Original image.
    :param lines: NxMx2 array of lines detected in the image.
    :return: Cropped image.
    """
    if lines.shape[0] == 0:
        return img

    x_min = np.min(lines[:, :, 0])
    x_max = np.max(lines[:, :, 0])
    y_min = np.min(lines[:, :, 1])
    y_max = np.max(lines[:, :, 1])

    left_dist = x_min
    right_dist = img.shape[1] - x_max
    top_dist = y_min
    bottom_dist = img.shape[0] - y_max

    min_dist = min(left_dist, right_dist, top_dist, bottom_dist)

    x1 = max(0, x_min - min_dist)
    x2 = min(img.shape[1], x_max + min_dist)
    y1 = max(0, y_min - min_dist)
    y2 = min(img.shape[0], y_max + min_dist)

    if not(crop_top_bottom):
        y1, y2 = 0, img.shape[0]

    cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

    return cropped_img

# img = cv2.imread("path_to_image.jpg")
# lines = detect_text_lines(img)
# cropped_img = crop_image_to_text(img, lines)
# cv2.imshow("Cropped Image", cropped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()