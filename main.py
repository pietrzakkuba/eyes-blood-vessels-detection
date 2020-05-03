import numpy as np
import matplotlib
import cv2
from os import listdir
from os.path import isfile, join
import sys


def get_paths(dir_name):
    # zamknąłem Twoje rzeczy w funkcji
    # spoko to zrobiłeś
    # nie chce mi się tego próbować zrozumieć, ale działa gitówa XD
    class Picture:
        def __init__(self, original_path, target_paths):
            self.original_path = original_path
            self.target_paths = target_paths

    dir_list = [join(dir_name, f) for f in listdir(dir_name)]
    dir_list = [x for x in dir_list if 'fovmask' not in x][:-1]

    picture_list = []

    loc_dir_list = [join(dir_list[0], f) for f in listdir(dir_list[0])]
    for i in range(0, len(loc_dir_list), 3):
        picture_list.append(Picture(loc_dir_list[i], [loc_dir_list[i + 1], loc_dir_list[i + 2]]))

    for i in range(1, len(dir_list) - 1, 2):
        loc_dir_list = [[join(dir_list[i], f) for f in listdir(dir_list[i])],
                        [join(dir_list[i + 1], f) for f in listdir(dir_list[i + 1])]]
        for j in range(len(loc_dir_list[0])):
            picture_list.append(Picture(loc_dir_list[0][j], loc_dir_list[1][j]))

    return [[pic.original_path, pic.target_paths] for pic in picture_list]


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


def normalize(image):
    checker = lambda x: 1 if x > 0 else 0
    return None  # TODO ^ to na elemantach arrayu wykonać i go zwrocić


def evaluation(processed, actual):
    # TODO trafność // trafność jest dziwna - nie jest wartością, czułość i swoistość są jej miarami
    # TODO https://pqstat.pl/?mod_f=diagnoza
    # TODO naczynie - positive, tło - negative
    accuracy = None  # wiec to chyba nieprzydatne
    # TODO czułość TP/(TP+FN)
    sensitivity = None
    # TODO swoistość TN/(FP+TN)
    specificity = None
    # TODO miary dla danych niezrównoważonych


paths = get_paths('pictures')
test_image = image_processing(paths[-1][0])
# show_image(test_image)
actual_image = read_image(paths[-1][1], cv2.IMREAD_GRAYSCALE, 800)
show_image(actual_image)
evaluation(test_image, actual_image)
