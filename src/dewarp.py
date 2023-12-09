import cv2
import numpy as np


def dewarp_page(img: np.ndarray, contours: np.ndarray, **kwargs) -> np.ndarray:
    """
    Dewarp and crop the image using the contours.
    :param img: Image to dewarp (not modified).
    :return: Dewarped image
    """
    # Find the extremities
    corners = find_extremities(contours)
    rect, dst, width, height = calculate_params(corners)

    M = cv2.getPerspectiveTransform(rect, dst)

    return cv2.warpPerspective(img, M, (int(width), int(height)))



def find_extremities(contours: np.ndarray) -> np.ndarray:
    """
    Find the top-left, top-right, bottom-right and bottom-left points.
    :param points: Points to find the extremities from.
    :return: numpy array of the 4 extremities in order from topLeft to bottomRight.
    """
    extremities: np.ndarray = np.zeros((4, 2), dtype="float32")
    points = contours.reshape(-1, 2)

    extremities[0] = min(points, key=lambda p: p[0] + p[1])
    extremities[1] = max(points, key=lambda p: p[0] - p[1])
    extremities[2] = max(points, key=lambda p: p[1] - p[0])
    extremities[3] = max(points, key=lambda p: p[0] + p[1])

    return extremities

def calculate_params(pts: np.ndarray):
    """
    Calculate the parameters of the rectangle.
    :param pts: Points of the rectangle.
    :return: Tuple of the rectangle, its width and height.

    """
    rect: np.ndarray = np.zeros((4, 2), dtype="float32")


    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]


    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))

    dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]],
    dtype="float32")


    return rect, dst, width, height

def page_curvature(img: np.ndarray, contours: np.ndarray, **kwargs) -> float:
    """
    Calculate the curvature of the page.
    :param contours: Contours of the page.
    :return: Curvature of the page.
    """
    corners = find_extremities(contours)
    #rows et cols c'est la taille de l'image en pixels
    #donc les deux boucles en dessous c'est pour parcourir l'image
    #faire des grid et faire des warpPerspectives pour ajuster la courbure
    rows, cols, ch = img.shape
    grid_size = 100


    for i in range(0, rows, grid_size):
        for j in range(0, cols, grid_size):
            src_pts = np.float32([[j, i], [j + grid_size, i], [j, i + grid_size], [j + grid_size, i + grid_size]])

            #Y va falloir utiliser la detection des lignes pour trouver les points de destination
            dst_pts = np.float32([])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (cols, rows))

            img[i:i + grid_size, j:j + grid_size] = warped[i:i + grid_size, j:j + grid_size]

    return img

