import shutil

import cv2
import numpy as np
import os


def get_single_digit_from_file(name):
    img = cv2.imread(name)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def process_training_digit_folder(folder, n):
    filenames = os.listdir(folder)[:n]
    data = []
    for each in filenames:
        if each.endswith(".png"):
            name = os.path.join(folder, each)
            data.append(get_single_digit_from_file(name))
    return data


def process_training_digits(folder, n):
    combined = []
    labels = []
    folders = os.listdir(folder)
    for subfolder in folders:
        if len(subfolder) != 1:
            continue
        digit = int(subfolder)
        path = os.path.join(folder, subfolder)
        if os.path.isdir(path):
            entries = process_training_digit_folder(path, n)
            combined += entries
            labels += [digit] * len(entries)

    trainData = np.array(combined).reshape(-1, 256).astype(np.float32)
    return trainData, np.array(labels)


def process_folder(src, dst, knn):
    contents = os.listdir(src)
    for each in contents:
        if each.endswith(".png"):
            path = os.path.join(src, each)
            data = get_single_digit_from_file(path)

            testData = np.array([data])
            testData = testData.reshape(-1, 256).astype(np.float32)

            ret, result, neighbours, dist = knn.findNearest(testData, 5)
            digit = str(int(ret))
            avg_dist = np.average(dist)
            if avg_dist < 150000:
                dst_path = os.path.join(dst, digit)
            else :
                dst_path = os.path.join(dst, "unknown")

            if not os.path.isdir(dst_path):
                os.makedirs(dst_path)
            shutil.copy(path, os.path.join(dst_path, each))

def main():
    trainData, trainLabels = process_training_digits("training_data", 1000)

    knn = cv2.ml.KNearest_create()

    knn.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

    process_folder("digits", "sorted_digits", knn)

if __name__ == "__main__":
    main()
