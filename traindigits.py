import cv2
import numpy as np
import os


def get_digit(name):
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def get_digits(folder, n):
    filenames = os.listdir(folder)[:n]
    data = []
    for each in filenames:
        name = os.path.join(folder, each)
        data.append(get_digit(name))
    return data


def main():
    data0 = get_digits("digits/0", 3)
    data5 = get_digits("digits/5", 3)
    data7 = get_digits("digits/7", 3)
    data8 = get_digits("digits/8", 3)
    data9 = get_digits("digits/9", 3)
    combined = np.array(data0 + data5 + data7 + data8 + data9)
    trainData = combined.reshape(-1, 256).astype(np.float32)
    #print(trainData)

    knn = cv2.ml.KNearest_create()

    trainLabels = np.array([0, 0, 0, 5, 5, 5, 7, 7, 7, 8, 8, 8, 9, 9, 9])
    knn.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

    test = get_digit("digits/8/digit_4_20180218_023522.png")
    testData = np.array([test])
    testData = testData.reshape(-1, 256).astype(np.float32)
    #print(testData)

    print(knn.findNearest(testData, 5))

if __name__ == "__main__":
    main()
