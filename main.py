import numpy as np
import matplotlib
import cv2
from os import listdir
from os.path import isfile, join
import sys
from Picture import Picture


def get_pictures():
    healthy = 'pictures\\healthy'
    healthy_fovmask = 'pictures\\healthy_fovmask'
    healthy_manual = 'pictures\\healthy_manualsegm'
    healthy_paths = [join(healthy, f) for f in listdir(healthy)]
    healthy_fovmask_paths = [join(healthy_fovmask, f) for f in listdir(healthy_fovmask)]
    healthy_manual_paths = [join(healthy_manual, f) for f in listdir(healthy_manual)]
    return [Picture(healthy_paths[i], healthy_fovmask_paths[i], healthy_manual_paths[i])
            for i in range(len(healthy_paths))]


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
    print(sum((tn, fn, fp, tp)))
    sensitivity = tp / (tp + fn)  # czułość
    specificity = tn / (fp + tn)  # swoistość
    print('czułość: {}'.format(sensitivity), 'swoistość: {}'.format(specificity))
    # TODO miary dla danych niezrównoważonych


pictures = get_pictures()
pictures[0].process_image()
pictures[0].show_image()
pictures[0].show_target()

evaluation(pictures[0].image, pictures[0].target)

