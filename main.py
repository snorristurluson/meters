import argparse
import sys
import time

import cv2
import numpy as np

SOURCE_IMAGE_WIDTH = 800
SOURCE_IMAGE_HEIGHT = 600

DIAL_AREA_LEFT_OFFSET = 50
DIAL_AREA_TOP_OFFSET = 50
DIAL_AREA_HEIGHT = 180


class HotWaterMeter(object):
    def __init__(self):
        self.image = None
        self.background = "dials_threshold"
        self.digit_pos_min = SOURCE_IMAGE_WIDTH
        self.digit_pos_max = 0
        self.last_known_digit_bounding_boxes = []
        self.digit_vertical_pos = 0
        self.dial_images = []
        self.dial_bounds = [(SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0),
                            (SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0),
                            (SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0),
                            (SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0)]


    def process_image(self, image):
        self.image = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.bilateralFilter(self.gray, 5, 150, 150)

        self.process_digits()
        self.process_dials()

        background_image = getattr(self, self.background)
        self.output = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
        self.show_digits()
        self.show_dials()
        self.show_dials_boxes()
        self.show_dials_contours()
        self.show_dials_hulls()
        self.show_dials_lines()

        return self.output


    def process_digits(self):
        self.digits_threshold = cv2.medianBlur(self.gray, 5)
        self.digits_threshold = cv2.adaptiveThreshold(
            self.digits_threshold, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            5, 3)

        for_contours = self.digits_threshold.copy()
        _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        bounding_boxes = get_bounding_boxes_for_contours(contours)
        self.digit_bounding_boxes = self.find_digit_bounding_boxes(bounding_boxes)
        self.digits = extract_digits(self.digit_bounding_boxes, self.gray)

    def process_dials(self):
        b, g, r = cv2.split(self.image)
        ret, self.dials_threshold = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        for_contours = self.dials_threshold.copy()
        _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        self.dial_contours = self.filter_dial_contours(contours)

        if len(self.dial_contours) == 4:
            dial_angles = []
            ix = 0
            for each in self.dial_contours:
                rect = cv2.minAreaRect(each)
                pos, dim, angle = rect
                dial_angles.append(angle)
                bb = cv2.boundingRect(each)

                current_bounds = self.dial_bounds[ix]
                cx, cy, cw, ch = current_bounds
                x, y, w, h = bb
                if x < cx:
                    cx = x
                if y < cy:
                    cy = y
                if x + w > cx + cw:
                    cw = x + w - cx
                if y + h > cy + ch:
                    ch = y + h - cy
                self.dial_bounds[ix] = (cx, cy, cw, ch)
                print(cx, cy, cw, ch)
                ix += 1

            print("Dials: {:.4}  {:.4}  {:.4}  {:.4}".format(
                dial_angles[0], dial_angles[1], dial_angles[2], dial_angles[3]
            ))

            self.dial_images = self.extract_images(self.dials_threshold, self.dial_bounds, (32, 32))
        else:
            print("Incorrect number of dials detected, skipping")

    def show_digits(self):
        for bb in self.digit_bounding_boxes:
            pt1 = (bb[0], bb[1])
            pt2 = (bb[0] + bb[2], bb[1] + bb[3])
            self.output = cv2.rectangle(self.output, pt1, pt2, (0, 255, 0))
        x = 8
        for digit in self.digits:
            digit = cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)
            self.output[8:24, x:x + 16] = digit
            x += 18

    def show_dials(self):
        x = 8
        for dial in self.dial_images:
            dial = cv2.cvtColor(dial, cv2.COLOR_GRAY2BGR)
            self.output[32:64, x:x + 32] = dial

            x += 40

    def show_dials_boxes(self):
        for each in self.dial_bounds:
            x, y, w, h = each
            self.output = cv2.rectangle(self.output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def show_dials_ellipses(self):
        for each in self.dial_contours:
            ellipse = cv2.fitEllipse(each)
            self.output = cv2.ellipse(self.output, ellipse, (0, 0, 255), 2)

    def show_dials_hulls(self):
        for each in self.dial_contours:
            hull = cv2.convexHull(each)
            self.output = cv2.drawContours(self.output, [hull], -1, (0, 0, 255), 2)

    def show_dials_lines(self):
        x0 = 32
        y0 = 64
        for each in self.dial_contours:
            hull = cv2.convexHull(each)
            line = cv2.fitLine(hull, cv2.DIST_L2, 0, 0.01, 0.01)
            [vx, vy, x, y] = line

            pt1 = (x0, y0)
            pt2 = (x0 + vx * 24, y0 + vy * 24)
            self.output = cv2.line(self.output, pt1, pt2, (0, 0, 255), 2)
            x0 += 32

    def is_dial_inverted(self, dial_contours):
        moments = cv2.moments(dial_contours)
        cx, cy = (
            int(moments['m10'] / moments['m00']),
            int(moments['m01'] / moments['m00'])
        )

        self.output = cv2.drawMarker(self.output, (cx, cy), (255, 0, 0), 2)

        rect = cv2.minAreaRect(dial_contours)
        pos, dim, angle = rect
        _, h = dim
        x, y = pos

        dist = np.sqrt((cx - x)**2 - (cy - y)**2)
        if dist > h/2:
            return True
        else:
            return False

    def show_dials_contours(self):
        cv2.drawContours(self.output, self.dial_contours, -1, (0, 255, 0))

    def show_dials_area(self):
        cv2.line(
            self.output,
            (self.digit_pos_min - DIAL_AREA_LEFT_OFFSET, self.digit_vertical_pos),
            (self.digit_pos_max, self.digit_vertical_pos),
            (0, 0, 255),
            2
        )
        cv2.line(
            self.output,
            (self.digit_pos_min - DIAL_AREA_LEFT_OFFSET, self.digit_vertical_pos + DIAL_AREA_HEIGHT),
            (self.digit_pos_max, self.digit_vertical_pos + DIAL_AREA_HEIGHT),
            (0, 0, 255),
            2
        )
        cv2.line(
            self.output,
            (self.digit_pos_min - DIAL_AREA_LEFT_OFFSET, self.digit_vertical_pos),
            (self.digit_pos_min - DIAL_AREA_LEFT_OFFSET, self.digit_vertical_pos + DIAL_AREA_HEIGHT),
            (0, 0, 255),
            2
        )
        cv2.line(
            self.output,
            (self.digit_pos_max, self.digit_vertical_pos),
            (self.digit_pos_max, self.digit_vertical_pos + DIAL_AREA_HEIGHT),
            (0, 0, 255),
            2
        )

    def filter_dial_contours(self, contours):
        filtered = []
        for each in contours:
            aabb = cv2.boundingRect(each)
            aax, aay, _, _ = aabb

            rect = cv2.minAreaRect(each)
            pos, dim, angle = rect
            x, y = pos
            w, h = dim

            # Reject nearly square area - like the spinning flow indicator
            if w > 30 and h > 30:
                continue

            # The digit positions give us a clue to where the dials might be
            if aay < self.digit_vertical_pos + DIAL_AREA_TOP_OFFSET:
                continue
            if aay > self.digit_vertical_pos + DIAL_AREA_HEIGHT:
                continue
            if aax < self.digit_pos_min - DIAL_AREA_LEFT_OFFSET:
                continue
            if aax > self.digit_pos_max:
                continue

            # We know the approximate size of the dials
            if w < 10:
                continue
            if h < 10:
                continue
            if w > 80:
                continue
            if h > 80:
                continue

            filtered.append(each)

        filtered.sort(key=lambda x: cv2.boundingRect(x)[0])
        return filtered

    def find_digit_bounding_boxes(self, bounding_boxes):
        longest_chain = []
        for bb in bounding_boxes:
            aligned = find_aligned_bounding_boxes(bb, bounding_boxes)
            if len(aligned) > len(longest_chain):
                longest_chain = aligned

        if len(longest_chain) < 5:
            return self.last_known_digit_bounding_boxes

        if len(longest_chain) > 5:
            longest_chain = longest_chain[:5]

        first_digit = longest_chain[0]
        last_digit = longest_chain[-1]

        if self.digit_pos_min is None:
            self.digit_pos_min = first_digit[0]

        if self.digit_pos_max is None:
            self.digit_pos_max = last_digit[0] + last_digit[2]

        if first_digit[0] < self.digit_pos_min:
            self.digit_pos_min = first_digit[0]
            self.digit_pos_max = last_digit[0] + last_digit[2]
            print("X:", self.digit_pos_min)

        if first_digit[1] > self.digit_vertical_pos:
            self.digit_vertical_pos = first_digit[1] + first_digit[2]
            print("Y:", self.digit_vertical_pos)

        self.last_known_digit_bounding_boxes = longest_chain
        return longest_chain[:5]

    def extract_images(self, source, bounding_boxes, size):
        images = []
        for bb in bounding_boxes:
            x, y, w, h = bb
            img = source[y:y+h, x:x+w].copy()
            img = cv2.resize(img, size)
            images.append(img)
        return images


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


def find_aligned_bounding_boxes(bb, bounding_boxes):
    result = []
    x0, y0, w0, h0 = bb
    center = y0 + h0 / 2
    for candidate in bounding_boxes:
        x, y, w, h = candidate
        if abs(y - y0) < 10:
            result.append(candidate)
    result.sort()

    # check for gaps and overlaps
    final_result = [bb]
    for candidate in result:
        x, y, w, h = candidate
        if x < x0 + w0 + 5:
            if x > x0 and x + w > x0 + w0:
                final_result.append(candidate)
                x0 = x
                w0 = w

    return final_result


def get_bounding_boxes_for_contours(contours):
    bounding_boxes = []
    for each in contours:
        bb = cv2.boundingRect(each)
        x, y, w, h = bb
        if w < 15:
            continue
        if h < 15:
            continue
        if w > 40:
            continue
        if h > 60:
            continue
        if w > h:
            continue
        bounding_boxes.append(bb)
    return bounding_boxes


def extract_digits(digit_bounding_boxes, img):
    digits = []
    for bb in digit_bounding_boxes:
        x, y, w, h = bb
        digit = img[y:y+h, x:x+w].copy()
        digit = cv2.resize(digit, (16, 16))
        digits.append(digit)
    return digits