import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_rotation_angle(gray):
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


def get_bounding_boxes_for_contours(contours):
    bounding_boxes = []
    for each in contours:
        bb = cv2.boundingRect(each)
        x, y, w, h = bb
        if w < 15:
            continue
        if h < 15:
            continue
        if w > 50:
            continue
        if h > 70:
            continue
        if w > h:
            continue
        bounding_boxes.append(bb)
    return bounding_boxes


def find_aligned_bounding_boxes(bb, bounding_boxes):
    result = []
    x0, y0, w0, h0 = bb
    center = y0 + h0 / 2
    for candidate in bounding_boxes:
        x, y, w, h = candidate
        if y <= center <= y + h and x > x0:
            result.append(candidate)
    result.sort()

    # check for gaps
    final_result = [bb]
    for candidate in result:
        x, y, w, h = candidate
        if x < x0 + w0 + 30:
            final_result.append(candidate)
            x0 = x
            w0 = w

    return final_result


def find_digit_bounding_boxes(bounding_boxes):
    longest_chain = []
    for bb in bounding_boxes:
        aligned = find_aligned_bounding_boxes(bb, bounding_boxes)
        if len(aligned) > len(longest_chain):
            longest_chain = aligned
    return longest_chain[:6]


def extract_digits(digit_bounding_boxes, img):
    digits = []
    for bb in digit_bounding_boxes:
        x, y, w, h = bb
        digit = img[y:y+h, x:x+w].copy()
        digit = cv2.resize(digit, (16, 16))
        digits.append(digit)
    return digits


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


def save_digits(digits):
    i = 0
    for digit in digits:
        cv2.imwrite("digit_%s.png" % i, digit)
        i += 1


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rotation_angle, lines, for_angle = find_rotation_angle(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    m = cv2.getRotationMatrix2D((400, 300), rotation_angle, 1)
    gray = cv2.warpAffine(gray, m, (800, 600))
    for_digits = cv2.warpAffine(img, m, (800, 600))
    for_digits = cv2.cvtColor(for_digits, cv2.COLOR_BGR2GRAY)
    for_contours = gray.copy()
    _, contours, _ = cv2.findContours(for_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    bounding_boxes = get_bounding_boxes_for_contours(contours)
    digit_bounding_boxes = find_digit_bounding_boxes(bounding_boxes)
    digits = extract_digits(digit_bounding_boxes, for_digits)
    save_digits(digits)
    contoured = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for_angle = cv2.cvtColor(for_angle, cv2.COLOR_GRAY2BGR)
    drawlines(for_angle, lines)
    for bb in digit_bounding_boxes:
        pt1 = (bb[0], bb[1])
        pt2 = (bb[0] + bb[2], bb[1] + bb[3])
        contoured = cv2.rectangle(contoured, pt1, pt2, (0, 255, 0))
    show_images(img, for_angle, contoured, digits)


def show_images(original, for_angle, contoured, digits):
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


def main():
    img = cv2.imread("09-20180124195013-01.jpg")
    img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
    process_image(img)


main()
