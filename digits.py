import cv2

from main import get_bounding_boxes_for_contours, extract_digits


def process_digits(image):
    img = cv2.resize(image, (800, 600), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 150, 150)
    for_digits = gray.copy()
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
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

    x = 8
    for digit in digits:
        digit = cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR)
        contoured[8:24, x:x + 16] = digit
        x += 18

    return contoured


