import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from color_correction import correct_colors
from contour_detection import detect_contours
from dewarp import dewarp_page
from text_detection import detect_text_lines
from utils import ScannerException

parser = argparse.ArgumentParser(description="Post-processing tool for scanning grayscale books")
parser.add_argument("input", action="store", type=str, nargs="+", help="Input files or folders")
parser.add_argument("-o", "--output", action="store", type=str, help="Output folder", default="out")
parser.add_argument("-v", action="store", dest="log_level", type=str,
                    choices=["info", "debug", "warn", "error"], help="Log level", default="warn")

ALLOWED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff"]


@dataclass
class Config:
    files: list[Path]
    output: Path
    log_level: int


def create_config() -> Config:
    args = parser.parse_args()

    # Get all input file paths
    def iter_input_files():
        for input_path in args.input:
            path = Path(input_path)
            if not path.exists():
                raise ScannerException(f"input file '{path}' does not exist")
            elif path.is_dir():
                for p in path.iterdir():
                    if p.is_file():
                        yield p
            else:
                yield path

    files = [p for p in iter_input_files() if p.suffix.lower() in ALLOWED_EXTENSIONS]

    if len({p.name for p in files}) != len(files):
        raise ScannerException("duplicate name in input files")

    if not files:
        raise ScannerException("no input files provided")

    # Create output folder
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    log_level = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARN,
        "error": logging.ERROR,
    }[args.log_level]

    return Config(files, output, log_level)


def process_image(img: np.ndarray) -> np.ndarray:
    contours = detect_contours(img)
    text_lines = detect_text_lines(img)
    img_dewarped = dewarp_page(img, contours, text_lines)
    img_corrected = correct_colors(img_dewarped)
    return img_corrected


def main() -> None:
    config = create_config()

    # Setup logging
    logging.getLogger().setLevel(config.log_level)
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')

    for i, in_path in enumerate(config.files):
        # Read the image
        try:
            img = cv2.imread(str(in_path))
        except cv2.Error as e:
            raise ScannerException(f"failed to read image '{in_path}': {e}")

        # Perform the operations
        img_processed = process_image(img)

        # Save the corrected image
        out_path = config.output / in_path.name
        try:
            cv2.imwrite(str(out_path), img_processed)
        except cv2.Error as e:
            raise ScannerException(f"failed to save image '{out_path}': {e}")

        logging.info(f"Processed image {i + 1} / {len(config.files)}")


if __name__ == '__main__':
    try:
        main()
    except ScannerException as ex:
        print(f"ERROR: {ex}")
        exit(1)