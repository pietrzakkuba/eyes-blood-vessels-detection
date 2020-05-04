import numpy as np
import cv2


def read(path, method):
    return cv2.imread(path, method)


def resize(image, width, height=None, interpolation=cv2.INTER_AREA):
    if height is None:
        (h, w) = image.shape[:2]
        ratio = w / width
        height = int(np.round(h / ratio))
    return cv2.resize(image, (width, height), interpolation=interpolation)


class Picture:
    def __init__(self, original_path, fovmask_path, target_path):
        self.original_path = original_path
        self.fovmask_path = fovmask_path
        self.target_path = target_path
        self.image = None
        self.target = read(target_path, cv2.IMREAD_GRAYSCALE)

    def adjust_gamma(self, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def process_image(self):
        self.image = read(self.original_path, cv2.IMREAD_COLOR)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height = self.image.shape[0]
        width = self.image.shape[1]
        self.image = resize(self.image, 800)
        self.adjust_gamma(1.07)
        self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        high_threshold, _ = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_threshold *= 0.75
        low_threshold = high_threshold / 2
        self.image = cv2.Canny(self.image, low_threshold, high_threshold)
        contours, hierarchy = cv2.findContours(self.image, 1, 2)
        cv2.drawContours(self.image, contours, -1, (255, 255, 0), 3)
        kernel = np.ones((3, 3), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        self.image = cv2.erode(self.image, kernel, iterations=1)
        mask = read(self.fovmask_path, cv2.IMREAD_GRAYSCALE)
        mask = resize(mask, 800)
        mask = cv2.erode(mask, kernel, iterations=3)
        self.image = cv2.bitwise_and(self.image, mask)
        self.image = resize(self.image, width, height)

    def show_image(self):
        image = resize(self.image, 800)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def show_target(self):
        image = resize(self.target, 800)
        cv2.imshow('target', image)
        cv2.waitKey(0)


