import argparse

import cv2
import numpy as np

from src.fix_orientation import fix_orientation
from src.text_detection import detect_text_lines

parser = argparse.ArgumentParser()
parser.add_argument("input", action="store", type=str, help="Input video")
parser.add_argument("output", action="store", type=str, help="Output video")


def operation1(img: np.ndarray) -> np.ndarray:
    text_lines = detect_text_lines(img)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    if text_lines.shape[0] > 0:
        result = lsd.drawSegments(img.copy(), text_lines)
    else:
        result = img
    return result


def operation2(img: np.ndarray) -> np.ndarray:
    text_lines = detect_text_lines(img)
    img_rotated = fix_orientation(img, text_lines)
    return img_rotated


OPERATIONS = [
    operation1,
    operation2,
]

FRAME_HEIGHT = 960


def main() -> None:
    args = parser.parse_args()

    # logging.getLogger().setLevel(logging.INFO)

    cap_in = cv2.VideoCapture(args.input)

    # Find frame size from operations
    # Operations must always return the same frame size!
    in_frame_size = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dummy_img = np.zeros((*in_frame_size, 3), dtype=np.uint8)
    frame_widths = np.zeros(len(OPERATIONS), dtype=np.uint32)
    for i, op in enumerate(OPERATIONS):
        frame = op(dummy_img)
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
        for j, op in enumerate(OPERATIONS):
            op_result = op(frame.copy())
            width = frame_widths[j]
            resized = cv2.resize(op_result, (width, FRAME_HEIGHT))
            new_frame[:, x:x + width, :] = resized
            x += width

        cap_out.write(new_frame)
        print(f"{i + 1} / {frame_count}")

    cap_in.release()
    cap_out.release()


if __name__ == '__main__':
    main()
