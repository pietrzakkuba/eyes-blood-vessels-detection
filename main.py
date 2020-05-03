import numpy as np
import matplotlib
import cv2
from os import listdir
from os.path import isfile, join


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


def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def resize(image, width):
    (h, w) = image.shape[:2]
    ratio = w / width
    height = int(np.round(h / ratio))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def adjust_gamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def image_processing(image):
    # tu trzeba coś fajnego odjebać
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize(image, 800)  # resize bo wysoka rozdzielczcośc to tylko problemy XD
    image = adjust_gamma(image, 1.07)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    high_threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_threshold *= 0.75
    low_threshold = high_threshold / 2
    image = cv2.Canny(image, low_threshold, high_threshold)
    # kernel = np.ones((3, 3), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=3)
    # image = cv2.erode(image, kernel, iterations=4)
    return image


paths = get_paths('pictures')
test_image = cv2.imread(paths[-1][0], cv2.IMREAD_COLOR)  # healthy -> 15_h
test_image = image_processing(test_image)
show_image(test_image)
