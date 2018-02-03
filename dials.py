import argparse
import sys

import cv2
import time

import numpy as np


def process_image(img, args):
    img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("%s_input.png" % args.prefix, img)
    b, g, r = cv2.split(img)
    cv2.imwrite("%s_green.png" % args.prefix, g)

    gray = cv2.bilateralFilter(g, 5, 150, 150)
    ret, gray = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    for_circles = gray.copy()

    cv2.imwrite("%s_thresh.png" % args.prefix, gray)
    for_contours = gray.copy()

    _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter_contours(contours)

    contoured = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(contoured, contours, -1, (0, 255, 0))
    for each in contours:
        rect = cv2.minAreaRect(each)
        x, y, angle = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        contoured = cv2.drawContours(contoured, [box], 0, (0, 0, 255), 2)


    cv2.imwrite("%s_contours.png" % args.prefix, contoured)

    if args.show_images:
        show_images(img, contoured)


def show_images(original, contoured):
    from matplotlib import pyplot as plt

    plt.subplot(121), plt.imshow(original, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(contoured, cmap='gray')
    plt.title('Contours'), plt.xticks([]), plt.yticks([])

    plt.show()


def process_dials(image):
    image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
    b, g, r = cv2.split(image)
    gray = cv2.bilateralFilter(g, 5, 150, 150)
    ret, gray = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    for_contours = gray.copy()
    _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter_contours(contours)
    contoured = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contoured, contours, -1, (0, 255, 0))
    for each in contours:
        rect = cv2.minAreaRect(each)
        x, y, angle = rect
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        contoured = cv2.drawContours(contoured, [box], 0, (0, 0, 255), 2)

    return contoured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--show-images",
        action="store_true",
        help="Show images after processing source image"
    )
    parser.add_argument(
        "-t", "--test-image",
        help="Test an image from file, rather than capturing"
    )
    parser.add_argument(
        "-p", "--prefix",
        default="dials_" + time.strftime("%Y%m%d_%H%M%S"),
        help="Prefix for image names"
    )

    args = parser.parse_args()

    if args.test_image:
        img = cv2.imread(args.test_image)
        process_image(img, args)


if __name__ == "__main__":
    sys.exit(main())
