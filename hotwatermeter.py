import math

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
        self.gray = None
        self.digits_threshold = None
        self.dials_threshold = None
        self.output = None
        self.background = "dials_threshold"
        self.digits = []
        self.digit_pos_min = SOURCE_IMAGE_WIDTH
        self.digit_pos_max = 0
        self.digit_bounding_boxes = []
        self.last_known_digit_bounding_boxes = []
        self.digit_vertical_pos = 0
        self.dial_contours = []
        self.dial_images = []
        self.dial_angles = [0, 0, 0, 0]
        self.dial_bounds = [(SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0),
                            (SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0),
                            (SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0),
                            (SOURCE_IMAGE_WIDTH, SOURCE_IMAGE_HEIGHT, 0, 0)]

    def process_image(self, image):
        h, w, _ = image.shape
        x0 = int(w / 3)
        y0 = int(h / 2)
        roi_width = int(w / 3)
        roi_height = int(h / 3)
        roi = image[y0:y0 + roi_height, x0:x0 + roi_width]

        self.image = cv2.resize(roi, (800, 600), interpolation=cv2.INTER_CUBIC)
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
        self.digits_threshold = cv2.medianBlur(self.gray, 3)
        self.digits_threshold = cv2.adaptiveThreshold(
            self.digits_threshold, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            5, 3)

        for_contours = self.digits_threshold.copy()
        _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        bounding_boxes = self.filter_digit_contours(contours)
        self.digit_bounding_boxes = self.find_digit_bounding_boxes(bounding_boxes)
        self.digits = self.extract_digits(self.digit_bounding_boxes, self.gray)

    def process_dials(self):
        b, g, r = cv2.split(self.image)
        ret, self.dials_threshold = cv2.threshold(g, 70, 255, cv2.THRESH_BINARY_INV)
        for_contours = self.dials_threshold.copy()
        _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
        self.dial_contours = self.filter_dial_contours(contours)
        self.dial_images = [None, None, None, None]
        if len(self.dial_contours) == 4:
            self.dial_angles = [0, 0, 0, 0]
            ix = 0
            for each in self.dial_contours:
                rect = cv2.minAreaRect(each)
                bb = cv2.boundingRect(each)

                self.accumulate_dial_bounds(bb, ix)

                dial = self.extract_rotated_image(
                    self.dials_threshold,
                    rect
                )
                self.dial_images[ix] = dial

                hull = cv2.convexHull(each)
                line = cv2.fitLine(hull, cv2.DIST_L2, 0, 0.01, 0.01)
                [vx, vy, _, _] = line

                angle = -math.atan2(vy, vx)
                angle_as_degrees = angle * 180 / math.pi
                h, w = dial.shape
                print(ix, w, h)
                w_2 = int(w / 2)
                h_2 = int(h / 2)
                if w > h:
                    # Dial is horizontal
                    left = dial[0:h, 0:w_2]
                    right = dial[0:h, w_2:w]
                    left_mean = cv2.mean(left)[0]
                    right_mean = cv2.mean(right)[0]
                    print(ix, "horizontal", left_mean, right_mean, angle_as_degrees)
                    if left_mean < right_mean:
                        # Image is darker on the left side, meaning the
                        # tip of the needle is on the left.
                        if abs(angle_as_degrees) < 90:
                            angle_as_degrees += 180
                    else:
                        if abs(angle_as_degrees) > 90:
                            angle_as_degrees += 180
                else:
                    # Dial is vertical
                    top = dial[0:h_2, 0:w]
                    bottom = dial[h_2:h, 0:w]
                    top_mean = cv2.mean(top)[0]
                    bottom_mean = cv2.mean(bottom)[0]
                    print(ix, "vertical", top_mean, bottom_mean, angle_as_degrees)
                    if top_mean < bottom_mean:
                        # Image is darker on the top, meaning the
                        # tip of the needle is on top
                        if angle_as_degrees < 0:
                            angle_as_degrees += 180
                    else:
                        if angle_as_degrees > 0:
                            angle_as_degrees += 180

                self.dial_angles[ix] = angle_as_degrees
                ix += 1

            print("Dials: {:.4}  {:.4}  {:.4}  {:.4}".format(
                self.dial_angles[0], self.dial_angles[1], self.dial_angles[2], self.dial_angles[3]
            ))

            # self.dial_images = self.extract_images(self.dials_threshold, self.dial_bounds, (32, 32))
        else:
            print("Incorrect number of dials detected, skipping")

    def accumulate_dial_bounds(self, bb, ix):
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
            if dial is not None:
                dial = cv2.cvtColor(dial, cv2.COLOR_GRAY2BGR)
                h = dial.shape[0]
                w = dial.shape[1]
                self.output[32:32 + h, x:x + w] = dial

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
        for angle_in_degrees in self.dial_angles:
            angle = angle_in_degrees * math.pi / 180
            pt1 = (x0, y0)
            pt2 = (x0 + int(math.cos(angle) * 24), int(y0 - math.sin(angle) * 24))
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

        dist = np.sqrt((cx - x) ** 2 - (cy - y) ** 2)
        if dist > h / 2:
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
            aligned = self.find_aligned_bounding_boxes(bb, bounding_boxes)
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
            img = source[y:y + h, x:x + w].copy()
            img = cv2.resize(img, size)
            images.append(img)
        return images

    def extract_rotated_image(self, source, rect):
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        w = rect[1][0]
        h = rect[1][1]

        xs = [i[0] for i in box]
        ys = [i[1] for i in box]
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        rotated = False
        angle = rect[2]

        if angle < -45:
            angle += 90
            rotated = True

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        size = (int(x2 - x1), int(y2 - y1))

        m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

        cropped = cv2.getRectSubPix(source, size, center)
        cropped = cv2.warpAffine(cropped, m, size)

        cropped_w = w if not rotated else h
        cropped_h = h if not rotated else w

        cropped_rotated = cv2.getRectSubPix(cropped, (int(cropped_w), int(cropped_h)),
                                            (size[0] / 2, size[1] / 2))
        return cropped_rotated


    def find_aligned_bounding_boxes(self, bb, bounding_boxes):
        result = []
        x0, y0, w0, h0 = bb
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


    def filter_digit_contours(self, contours):
        bounding_boxes = []
        for each in contours:
            bb = cv2.boundingRect(each)
            x, y, w, h = bb
            if w < 30:
                continue
            if h < 30:
                continue
            if w > 60:
                continue
            if h > 80:
                continue
            if w > h:
                continue
            bounding_boxes.append(bb)
        return bounding_boxes


    def extract_digits(self, digit_bounding_boxes, img):
        digits = []
        for bb in digit_bounding_boxes:
            x, y, w, h = bb
            digit = img[y:y + h, x:x + w].copy()
            digit = cv2.resize(digit, (16, 16))
            digits.append(digit)
        return digits
