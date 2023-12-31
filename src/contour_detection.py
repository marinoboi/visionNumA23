import cv2
import imutils
import numpy as np


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def remove_edge(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove the edge of the photo to detect component at the edge of the image

    Args:
        img (np.ndarray): The image to remove the edges from

    Returns:
        np.ndarray: The image with the edges removed
    """
    iteration_x = 1
    iteration_y = 1
    for i in range(img.shape[0]):
        for j in range(iteration_x):
            img[i][j] = 255
            img[i][img.shape[1] - j - 1] = 255
    for i in range(img.shape[1]):
        for j in range(iteration_y):
            img[j][i] = 255
            img[img.shape[0] - j - 1][i] = 255

    return img


def filter_contours(input_contours: list[np.ndarray], min_area: float, min_perimeter: float, min_width: float,
                    max_width: float, min_height: float, max_height: float, solidity: list[float, float],
                    max_vertex_count: float, min_vertex_count: float, min_ratio: float, max_ratio: float) \
        -> list[np.ndarray]:
    """Filters out contours that do not meet certain criteria.
    Args:
        input_contours (list[np.ndarray]): Contours as a list of numpy.ndarray.
        min_area (float): The minimum area of a contour that will be kept.
        min_perimeter (float): The minimum perimeter of a contour that will be kept.
        min_width (float): Minimum width of a contour.
        max_width (float): MaxWidth maximum width.
        min_height (float): Minimum height.
        max_height (float): Maximimum height.
        solidity (list[float, float]): The minimum and maximum solidity of a contour.
        min_vertex_count (float): Minimum vertex Count of the contours.
        max_vertex_count (float): Maximum vertex Count.
        min_ratio (float): Minimum ratio of width to height.
        max_ratio (float): Maximum ratio of width to height.
    Returns:
        Contours as a list of numpy.ndarray.
    """
    output = []
    for contour in input_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or w > max_width:
            continue
        if h < min_height or h > max_height:
            continue
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if cv2.arcLength(contour, True) < min_perimeter:
            continue
        hull = cv2.convexHull(contour)
        solid = 100 * area / cv2.contourArea(hull)
        if solid < solidity[0] or solid > solidity[1]:
            continue
        if len(contour) < min_vertex_count or len(contour) > max_vertex_count:
            continue
        ratio = float(w) / h
        if ratio < min_ratio or ratio > max_ratio:
            continue
        output.append(contour)
    return output


def detect_contours(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Detect page contours in an image.
    :param img: Image to detect contours in.
    :return: Nx2 numpy array of points on the contour in the image.
    """
    image_height_resize: int = 1500

    orig: np.ndarray = image.copy()
    image: np.ndarray = imutils.resize(image, height=image_height_resize)
    image = cv2.bilateralFilter(image, 13, 200, 200)
    output_contour: np.ndarray = None
    area_img: int = image.shape[0] * image.shape[1]

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # print("STEP 1: Edge Detection")
    for tresh in range(120, 59, -15):
        output: np.ndarray = orig.copy()
        found: bool = False
        edged = cv2.Canny(gray, tresh, tresh + 10, apertureSize=5, L2gradient=False)
        edged = cv2.dilate(edged, kernel=None, anchor=(-1, -1), iterations=2, borderType=cv2.BORDER_CONSTANT,
                           borderValue=(-1))
        edged = remove_edge(edged)

        contours: tuple = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours: tuple = imutils.grab_contours(contours)
        cnts: list = sorted(contours, key=cv2.contourArea, reverse=True)

        if True:
            filter_contours_min_area: float = image.shape[0] * image.shape[1] * 0.1
            filter_contours_min_perimeter: float = 100.0
            filter_contours_min_width: float = image.shape[1] * 0.30
            filter_contours_max_width: float = image.shape[1] * 0.99
            filter_contours_min_height: float = image.shape[0] * 0.30
            filter_contours_max_height: float = image.shape[0] * 0.99
            filter_contours_solidity: list[float, float] = [0, 100]
            filter_contours_max_vertices: float = 1000000.0
            filter_contours_min_vertices: float = 0.0
            filter_contours_min_ratio: float = 0.0
            filter_contours_max_ratio: float = 1000
            cnts = filter_contours(cnts,
                                   filter_contours_min_area,
                                   filter_contours_min_perimeter,
                                   filter_contours_min_width,
                                   filter_contours_max_width,
                                   filter_contours_min_height,
                                   filter_contours_max_height,
                                   filter_contours_solidity,
                                   filter_contours_max_vertices,
                                   filter_contours_min_vertices,
                                   filter_contours_min_ratio,
                                   filter_contours_max_ratio)
        # loop over the contours
        screenCnt: np.ndarray = None
        contour: np.ndarray
        for contour in cnts:
            hull = cv2.convexHull(contour)
            contour = hull
            peri: int = cv2.arcLength(contour, True)
            approx: np.ndarray = cv2.approxPolyDP(contour, epsilon=0.02 * peri, closed=True)

            if 4 <= len(approx) <= 4:
                contour = (contour * orig.shape[0]) // image_height_resize
                cv2.drawContours(output, [contour], -1, (0, 255, 0), orig.shape[0] // 200)
                screenCnt: np.ndarray = approx
                screenCnt = (screenCnt * orig.shape[0]) // image_height_resize
                # (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
                found = True
                output_contour = contour.copy()
                break
            elif len(approx) < 4 or len(approx) > 4:
                cv2.drawContours(output, [contour], -1, (0, 0, 255), orig.shape[0] // 200)
        if found:
            break
    # show the contour (outline) of the piece of paper
    if output_contour is None:
        output_contour = np.array([[0, 0], [image.shape[1], 0],
                                   [image.shape[1], image.shape[0]], [0, image.shape[0]]])
    else:
        output_contour = output_contour.squeeze(1)
    return output_contour
