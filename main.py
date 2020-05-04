import numpy as np
import matplotlib
import cv2
from os import listdir
from os.path import isfile, join
import sys


class Picture:
    def __init__(self, original_path, target_path):
        self.original_path = original_path
        self.target_path = target_path

def getPaths():
    healthy='pictures\healthy'
    healthymanual='pictures\healthy_manualsegm'
    pictures=[]

    healthy_paths=[join(healthy, f) for f in listdir(healthy)]
    healthymanual_paths = [join(healthymanual, f) for f in listdir(healthymanual)]
    for i in range(len(healthy_paths)):
        print(healthy_paths[i])
        print(healthymanual_paths[i])
        print()
        pictures.append(Picture(healthy_paths[i], healthymanual_paths[i]))
    return pictures

def resize(image, width):
    (h, w) = image.shape[:2]
    ratio = w / width
    height = int(np.round(h / ratio))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def read_image(path, method, size):
    image = cv2.imread(path, method)
    image = resize(image, size)
    return image


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def image_processing(path):
    image = read_image(path, cv2.IMREAD_COLOR, 800)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(image, 800)
    image = adjust_gamma(image, 1.07)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    high_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_threshold *= 0.75
    low_threshold = high_threshold / 2
    image = cv2.Canny(image, low_threshold, high_threshold)
    contours, hierarchy = cv2.findContours(image, 1, 2)
    cv2.drawContours(image, contours, -1, (255, 255, 0), 3)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    ##############################################################
    mask_path = '.\\pictures\\healthy_fovmask\\15_h_mask.tif'  # TODO adresowanie tego ładnie jak inne
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = resize(mask, 800)
    mask = cv2.erode(mask, kernel, iterations=3)
    #############################################################
    return cv2.bitwise_and(image, mask)


def norm(image):
    return [point for row in
            [list(map(int, row / 255)) for row in list(cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)[1])]
            for point in row]


def analysis(tested_values, actual_values):
    tn = 0
    fn = 0
    fp = 0
    tp = 0
    for pair in zip(tested_values, actual_values):
        if pair[0] == 0 and pair[1] == 0:
            tn += 1
        elif pair[0] == 0 and pair[1] == 1:
            fn += 1
        elif pair[0] == 1 and pair[1] == 0:
            fp += 1
        else:
            tp += 1
    return float(tn), float(fn), float(fp), float(tp)


def evaluation(processed, actual):
    tn, fn, fp, tp = analysis(norm(processed), norm(actual))
    # TODO macierze pomyłek
    # trafność // trafność jest dziwna - nie jest wartością, czułość i swoistość są jej miarami
    # https://pqstat.pl/?mod_f=diagnoza
    # naczynie - positive, tło - negative
    # accuracy = None  # wiec to chyba nieprzydatne
    sensitivity = tp / (tp + fn)     # czułość
    specificity = tn / (fp + tn)     # swoistość
    print('czułość: {}'.format(sensitivity), 'swoistość: {}'.format(specificity))
    # TODO miary dla danych niezrównoważonych



paths = getPaths()
test_image=image_processing(paths[0].original_path)
show_image(test_image)
actual_image = read_image(paths[0].target_path, cv2.IMREAD_GRAYSCALE, 800)
show_image(actual_image)
evaluation(test_image, actual_image)

# paths = get_paths('pictures')
# test_image = image_processing(paths[-1][0])
# # show_image(test_image)
# actual_image = read_image(paths[-1][1], cv2.IMREAD_GRAYSCALE, 800)
# # show_image(actual_image)
# evaluation(test_image, actual_image)
