import cv2
import numpy as np
import math


def correct_colors(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    :param img: Image to correct colors for (not modified).
    :return: Image with corrected colors.
    """
    gamma = 2.2
    print(gamma)

    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    img_corrected = cv2.LUT(img, lookUpTable)

    '''cv2.imshow('Original Image', img)
    cv2.imshow('Gamma image', img_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return img_corrected
