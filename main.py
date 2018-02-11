import cv2
import numpy as np


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


