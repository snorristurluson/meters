import cv2


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
        if x < x0 + w0 + 30:
            if x > x0 and x + w > x0 + w0:
                final_result.append(candidate)
                x0 = x
                w0 = w

    return final_result