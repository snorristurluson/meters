import argparse
import sys
import time

import cv2
import numpy as np

from digits import extract_digits, find_aligned_bounding_boxes, get_bounding_boxes_for_contours


class HotWaterMeter(object):
    def __init__(self):
        self.image = None
        self.digit_pos_min = 800
        self.digit_pos_max = 0
        self.last_known_digit_bounding_boxes = []
        self.digit_vertical_pos = 0


    def process_image(self, image):
        self.image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.bilateralFilter(self.gray, 5, 150, 150)
        self.output = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        self.process_digits()
        self.process_dials()

        return self.output


    def process_digits(self):

        threshold = cv2.medianBlur(self.gray, 5)
        threshold = cv2.adaptiveThreshold(threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 3)

        for_contours = threshold.copy()
        _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = get_bounding_boxes_for_contours(contours)
        digit_bounding_boxes = self.find_digit_bounding_boxes(bounding_boxes)
        digits = extract_digits(digit_bounding_boxes, self.gray)

        for bb in digit_bounding_boxes:
            pt1 = (bb[0], bb[1])
            pt2 = (bb[0] + bb[2], bb[1] + bb[3])
            self.output = cv2.rectangle(self.output, pt1, pt2, (0, 255, 0))

        x = 8
        for digit in digits:
            digit = cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)
            self.output[8:24, x:x + 16] = digit
            x += 18

    def process_dials(self):
        ret, threshold = cv2.threshold(self.gray, 40, 255, cv2.THRESH_BINARY_INV)
        for_contours = threshold.copy()
        _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = self.filter_dial_contours(contours)

        self.output = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(self.output, contours, -1, (0, 255, 0))
        for each in contours:
            rect = cv2.minAreaRect(each)
            x, y, angle = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            self.output = cv2.drawContours(self.output, [box], 0, (0, 0, 255), 2)

    def filter_dial_contours(self, contours):
        filtered = []
        for each in contours:
            bb = cv2.boundingRect(each)
            x, y, w, h = bb
            if y < self.digit_vertical_pos:
                continue
            if y > self.digit_vertical_pos + 300:
                continue
            if x < self.digit_pos_min - 100:
                continue
            if x > self.digit_pos_max + 100:
                continue
            if w < 15:
                continue
            if h < 15:
                continue
            if w > 60:
                continue
            if h > 60:
                continue

            rect = cv2.minAreaRect(each)
            pos, dim, angle = rect
            w, h = dim
            if abs(w - h) < 8:
                continue

            filtered.append(each)

        return filtered

    def find_digit_bounding_boxes(self, bounding_boxes):
        longest_chain = []
        for bb in bounding_boxes:
            aligned = find_aligned_bounding_boxes(bb, bounding_boxes)
            if len(aligned) > len(longest_chain):
                longest_chain = aligned

        if len(longest_chain) < 5:
            return self.last_known_digit_bounding_boxes

        first_digit = longest_chain[0]
        last_digit = longest_chain[-1]

        if self.digit_pos_min is None:
            self.digit_pos_min = first_digit[0]

        if self.digit_pos_max is None:
            self.digit_pos_max = last_digit[0] + last_digit[2]

        if first_digit[0] < self.digit_pos_min:
            self.digit_pos_min = first_digit[0]
            print("X:", self.digit_pos_min)

        if last_digit[0] + last_digit[2] > self.digit_pos_max:
            self.digit_pos_max = last_digit[0] + last_digit[2]

        if first_digit[1] > self.digit_vertical_pos:
            self.digit_vertical_pos = first_digit[1] + first_digit[2]
            print("Y:", self.digit_vertical_pos)

        self.last_known_digit_bounding_boxes = longest_chain
        return longest_chain[:6]


def find_rotation_angle(gray):
    return 0, [], gray

    gray = cv2.medianBlur(gray, 3)
    ret, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_TRIANGLE)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    lines = cv2.HoughLines(gray, 1, np.pi/180, 200)
    if lines is None:
        print("No lines found, can't determine rotation angle")
        return 0, [], gray

    print("Found %d lines" % len(lines))
    candidates = []
    for each in lines:
        rho, theta = each[0]
        theta_deg = theta * 180 / np.pi
        if 70 < theta_deg < 110:
            candidates.append(each)

    lines = candidates
    if not lines:
        print("No lines found, can't determine rotation angle")
        return 0, [], gray

    line = lines[0]
    rho, theta = line[0]
    rotation_angle = theta
    rotation_angle *= 180 / np.pi
    rotation_angle -= 90

    return rotation_angle, lines, gray


def drawlines(img, lines):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


def save_digits(digits, args):
    i = 0
    for digit in digits:
        cv2.imwrite("%s_digit_%s.png" % (args.prefix, i), digit)
        i += 1


def process_image(img, args):
    img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("%s_input.png" % args.prefix, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rotation_angle, lines, for_angle = find_rotation_angle(gray)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    m = cv2.getRotationMatrix2D((400, 300), rotation_angle, 1)
    gray = cv2.warpAffine(gray, m, (800, 600))
    for_digits = cv2.warpAffine(img, m, (800, 600))
    for_digits = cv2.cvtColor(for_digits, cv2.COLOR_BGR2GRAY)
    for_contours = gray.copy()
    _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = get_bounding_boxes_for_contours(contours)
    digit_bounding_boxes = find_digit_bounding_boxes(bounding_boxes)
    digits = extract_digits(digit_bounding_boxes, for_digits)

    contoured = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for bb in digit_bounding_boxes:
        pt1 = (bb[0], bb[1])
        pt2 = (bb[0] + bb[2], bb[1] + bb[3])
        contoured = cv2.rectangle(contoured, pt1, pt2, (0, 255, 0))
    cv2.imwrite("%s_contours.png" % args.prefix, contoured)

    if args.save_digits:
        save_digits(digits, args)

    if args.show_images:
        for_angle = cv2.cvtColor(for_angle, cv2.COLOR_GRAY2BGR)
        drawlines(for_angle, lines)
        show_images(img, for_angle, contoured, digits)


def show_images(original, for_angle, contoured, digits):
    from matplotlib import pyplot as plt

    plt.subplot(331), plt.imshow(original, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(332), plt.imshow(for_angle, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(333), plt.imshow(contoured, cmap='gray')
    plt.title('Contours'), plt.xticks([]), plt.yticks([])
    plt.subplot(334), plt.imshow(digits[0], cmap='gray')
    plt.title('Digit'), plt.xticks([]), plt.yticks([])
    plt.subplot(335), plt.imshow(digits[1], cmap='gray')
    plt.title('Digit'), plt.xticks([]), plt.yticks([])
    plt.subplot(336), plt.imshow(digits[2], cmap='gray')
    plt.title('Digit'), plt.xticks([]), plt.yticks([])
    plt.subplot(337), plt.imshow(digits[3], cmap='gray')
    plt.title('Digit'), plt.xticks([]), plt.yticks([])
    plt.subplot(338), plt.imshow(digits[4], cmap='gray')
    plt.title('Digit'), plt.xticks([]), plt.yticks([])
    plt.subplot(339), plt.imshow(digits[5], cmap='gray')
    plt.title('Digit'), plt.xticks([]), plt.yticks([])
    plt.show()


def take_picture():
    cap = cv2.VideoCapture(0)
    for i in range(30):
        cap.read()
    ret, img = cap.read()
    cap.release()
    return img, ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--show-images",
        action="store_true",
        help="Show images after processing source image"
    )
    parser.add_argument(
        "-s", "--save-digits",
        action="store_true",
        help="Store digit images rather than identifying them"
    )
    parser.add_argument(
        "-t", "--test-image",
        help="Test an image from file, rather than capturing"
    )
    parser.add_argument(
        "-p", "--prefix",
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="Prefix for image names"
    )
    args = parser.parse_args()

    if args.test_image:
        img = cv2.imread(args.test_image)
        process_image(img, args)

    else:
        while True:
            img, ret = take_picture()

            if ret:
                args.prefix = time.strftime("%Y%m%d_%H%M%S")
                process_image(img, args)
            else:
                print("Couldn't read frame", ret)

            time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())


