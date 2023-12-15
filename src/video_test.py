import argparse

import cv2
import numpy as np

from color_correction import correct_colors
from contour_detection import detect_contours
from dewarp import dewarp_page
from fix_orientation import fix_orientation
from text_detection import detect_text_lines

parser = argparse.ArgumentParser()
parser.add_argument("input", action="store", type=str, help="Input video")
parser.add_argument("output", action="store", type=str, help="Output video")

def _detect_text_lines(state: dict) -> np.ndarray:
    text_lines = detect_text_lines(state["img"], intermediates=state)
    state["text_lines"] = text_lines
    return state["img_text_lines"]


def _fix_orientation(state: dict) -> np.ndarray:
    img_rotated = fix_orientation(state["img"], state["text_lines"], intermediates=state)
    state["img"] = img_rotated
    return img_rotated

def _dewarp_grid(state: dict) -> np.ndarray:
    img = state["img"]
    contour = detect_contours(img, intermediates=state)
    state["img"] = dewarp_page(img, contour, intermediates=state)
    return state["img_dewarp_grid"]

def _dewarp_result(state: dict) -> np.ndarray:
    return state["img"]

def _correct_colors(state: dict) -> np.ndarray:
    img_corrected = correct_colors(state["img"], intermediates=state)
    state["img"] = img_corrected
    return img_corrected


OPERATIONS = [
    _detect_text_lines,
    _fix_orientation,
    _dewarp_grid,
    _dewarp_result,
    _correct_colors,
]

FRAME_HEIGHT = 960


def main() -> None:
    args = parser.parse_args()

    # logging.getLogger().setLevel(logging.INFO)

    cap_in = cv2.VideoCapture(args.input)

    # Find frame size from operations
    # Operations must always return the same frame size!
    in_frame_size = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    state = {"img": np.zeros((*in_frame_size, 3), dtype=np.uint8)}
    frame_widths = np.zeros(len(OPERATIONS), dtype=np.uint32)
    for i, op in enumerate(OPERATIONS):
        frame = op(state)
        frame_widths[i] = round(frame.shape[0] * (FRAME_HEIGHT / frame.shape[1]))
    frame_width = np.sum(frame_widths)

    # Create video output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_count = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_out = cv2.VideoWriter(args.output, fourcc, cap_in.get(cv2.CAP_PROP_FPS),
                              (frame_width, FRAME_HEIGHT), True)

    # Process all frames
    for i in range(frame_count):
        ret, frame = cap_in.read()

        # Build new frame
        new_frame = np.zeros((FRAME_HEIGHT, frame_width, 3), dtype=np.uint8)
        x = 0
        state = {"img": frame}
        for j, op in enumerate(OPERATIONS):
            op_result = op(state)
            width = frame_widths[j]
            resized = cv2.resize(op_result, (width, FRAME_HEIGHT))
            if len(resized.shape) == 2:
                resized = np.tile(resized[:, :, np.newaxis], (1, 1, 3))
            new_frame[:, x:x + width, :] = resized
            x += width

        cap_out.write(new_frame)
        print(f"{i + 1} / {frame_count}")

    cap_in.release()
    cap_out.release()


if __name__ == '__main__':
    main()
