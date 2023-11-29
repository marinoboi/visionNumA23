import cv2
import numpy as np
from skimage.filters import threshold_local
from imutils.object_detection import non_max_suppression
import imutils
from tkinter import Tk, Label, Frame
from PIL import Image, ImageTk
import os
from os import listdir
import time
from threading import Thread, current_thread

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
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
		[0, maxHeight - 1]], dtype = "float32")
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
    iteration_x = img.shape[0] // 200
    iteration_y = img.shape[1] // 200
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
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio: float = image.shape[0] / 500.0
    orig: np.ndarray = image.copy()
    image: np.ndarray = imutils.resize(image, height = 1500)
    image = cv2.bilateralFilter(image, 13, 200, 200)
    output_contour: np.ndarray = None
    area_img: int = image.shape[0] * image.shape[1]
    # convert the image to grayscale, blur it, and find edges in the image
    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # edged = cv2.Canny(gray, 75, 200, apertureSize=3, L2gradient=True) # kinda good


    # edged2: np.ndarray = cv2.Canny(gray, 70, 150, apertureSize=5, L2gradient=True)

    # edged = cv2.bitwise_or(edged2, edged)
    print("STEP 1: Edge Detection")
    for tresh in range(120, 59, -15):
        output: np.ndarray = image.copy()
        found: bool = False
        edged = cv2.Canny(gray, tresh, tresh + 10, apertureSize=5, L2gradient=False)
        # show the original image and the edge detected image
        # cv2.imwrite('./src./images/Image.png', image)
        # cv2.imwrite('./src./images/gray.png', gray)
        # _, edged = cv2.threshold(edged, 127, 255, cv2.THRESH_BINARY)
        edged = cv2.dilate(edged, kernel=None, anchor=(-1, -1), iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))
        # edged = cv2.medianBlur(edged, 3)
        # edged = cv2.GaussianBlur(edged, (3, 3), 0)

        #edged = cv2.erode(edged, kernel=None, anchor=(-1, -1), iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))
        #edged = cv2.dilate(edged, kernel=None, anchor=(-1, -1), iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))

        #edged = cv2.dilate(edged, kernel=None, anchor=(-1, -1), iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))
        #edged = cv2.erode(edged, kernel=None, anchor=(-1, -1), iterations=2, borderType=cv2.BORDER_CONSTANT, borderValue=(-1))
        edged = remove_edge(edged)
        # cv2.imwrite('./src./images/Edged.png', edged)

        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        contours: tuple = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours: tuple = imutils.grab_contours(contours)
        cnts: list = sorted(contours, key = cv2.contourArea, reverse = True)

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
            filter_contours_max_ratio: float = 6
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
        print(len(cnts))
        # loop over the contours
        screenCnt: np.ndarray = None
        contour: np.ndarray
        for contour in cnts:
            # area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            # hull_area = cv2.contourArea(hull)
            # solidity = float(area)/hull_area
            contour = hull
            # approximate the contour
            peri: int = cv2.arcLength(contour, True)
            # approx: np.ndarray = cv2.approxPolyDP(contour, 0.007 * peri, True)
            # approx: np.ndarray = cv2.approxPolyDP(contour, epsilon=0.05 * peri, closed=True)
            approx: np.ndarray = cv2.approxPolyDP(contour, epsilon=0.02 * peri, closed=True)
            # print(len(approx))

            # x,y,w,h = cv2.boundingRect(contour)
            # area_bounding = w*h

            # if solidity < 0.7:
            #     print('SOLIDITY')
            #     continue
            # if area_bounding / area > 1.20:
            #     print('AREA')
            #     continue
            if 4 <= len(approx) <= 4:
                cv2.drawContours(output, [contour], -1, (0, 255, 0), 10)
                screenCnt: np.ndarray = approx
                # (x,y),(MA,ma),angle = cv2.fitEllipse(contour)
                # M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), -min(angle%90, abs(90-angle), abs(180-angle)), 1.0)
                # image = cv2.warpAffine(image, M, (width, height))
                found = True
                # print('area_bounding / area: ' + str(area_bounding / area))
                # print('solidity: ' + str(solidity))
                output_contour = contour.copy()
                print(tresh)
                break
            elif len(approx) < 4 or len(approx) > 4:
                cv2.drawContours(output, [contour], -1, (0, 0, 255), 10)
        if found:
            break
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imwrite('./src./images/Outline.png', image)

    output: np.ndarray = imutils.resize(output, height = 400)
    edged: np.ndarray = imutils.resize(edged, height = 400)
    gray: np.ndarray = imutils.resize(gray, height = 400)
    return output_contour #, output, edged
    if screenCnt is None:
        return output_contour #, output, edged

    # apply the four point transform to obtain a top-down
    # view of the original image
    # orig = imutils.resize(orig, height = 400)
    # screenCnt *= orig.shape[0] // output.shape[0]
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255
    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    # cv2.imwrite('./src./images/Original.png', orig)
    # cv2.imwrite('./src./images/Scanned.png', warped)

    warped: np.ndarray = imutils.resize(warped, height = 400)
    return output_contour #, output, warped


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required = True,
    #     help = "Path to the image to be scanned")
    # args = vars(ap.parse_args())
    # image = cv2.imread(args["image"])

    # detect_contours(image)

    # Create an instance of TKinter Window or frame
    win = Tk()

    # Set the size of the window
    win.geometry("1280x720")

    # Create a Label to capture the Video frames
    label =Label(win)
    label2 =Label(win)
    label.grid(row=0, column=0)
    label2.grid(row=0, column=1)
    capture: cv2.VideoCapture = cv2.VideoCapture(0)
    width = 1280
    height = 720
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    # get the path/directory
    folder_dir = "C:/Users/Vince/OneDrive/Bureau/University/GIF/Session 7/Vision/Projet/images"
    folder_dir = "../images"
    image_dir: list[str] = []
    for image in os.listdir(folder_dir):
        if 'IMG' in image or True:
            image_dir.append(folder_dir + '/' + image)

    def show_frames(counter: int) -> None:
        # Get the latest frame and convert into Image
        img: np.ndarray = capture.read()[1]
        if counter >= len(image_dir):
            counter = 0
        img: np.ndarray = cv2.imread(image_dir[counter])
        print(image_dir[counter])

        contours, img, edged = detect_contours(img)
        # cv2.imwrite(os.getcwd() + '/output/original.png', img)
        # cv2.imwrite(os.getcwd() + '/output/filtered.png', black_and_white)
        cv2image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2edged= cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
        edged = Image.fromarray(cv2edged)
        # Convert image to PhotoImage
        edgedtk = ImageTk.PhotoImage(image=edged)

        image = Image.fromarray(cv2image)
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=image)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        label2.imgtk = edgedtk
        label2.configure(image=edgedtk)
        # Repeat after an interval to capture continiously
        label.after(500, lambda: show_frames(counter + 1))


    show_frames(0)
    frame: Frame = Frame()
    win.mainloop()