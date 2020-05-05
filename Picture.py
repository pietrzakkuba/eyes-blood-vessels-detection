import numpy as np
import cv2
from random import randint, seed
from Segment import Segment
import sys


# imgToCut[:, :, 2]=0

class Picture:
    def __init__(self, original_path, fovmask_path, target_path):
        self.original_path = original_path
        self.fovmask_path = fovmask_path
        self.target_path = target_path
        self.basic_processing_image = None
        self.target = self.read(target_path, cv2.IMREAD_GRAYSCALE)
        self.original_image = self.read(self.original_path, cv2.IMREAD_GRAYSCALE)
        self.segments_list = []

    def adjust_gamma(self, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.basic_processing_image = cv2.LUT(self.basic_processing_image, table)

    def process_image(self):
        self.basic_processing_image = self.read(self.original_path, cv2.IMREAD_COLOR)
        self.basic_processing_image = cv2.cvtColor(self.basic_processing_image, cv2.COLOR_BGR2GRAY)
        height = self.basic_processing_image.shape[0]
        width = self.basic_processing_image.shape[1]
        self.basic_processing_image = self.resize(self.basic_processing_image, 800)
        self.adjust_gamma(1.07)
        self.basic_processing_image = cv2.GaussianBlur(self.basic_processing_image, (3, 3), 0)
        high_threshold, _ = cv2.threshold(self.basic_processing_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_threshold *= 0.75
        low_threshold = high_threshold / 2
        self.basic_processing_image = cv2.Canny(self.basic_processing_image, low_threshold, high_threshold)
        contours, hierarchy = cv2.findContours(self.basic_processing_image, 1, 2)
        cv2.drawContours(self.basic_processing_image, contours, -1, (255, 255, 0), 3)
        kernel = np.ones((3, 3), np.uint8)
        self.basic_processing_image = cv2.dilate(self.basic_processing_image, kernel, iterations=1)
        self.basic_processing_image = cv2.erode(self.basic_processing_image, kernel, iterations=1)
        mask = self.read(self.fovmask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.resize(mask, 800)
        mask = cv2.erode(mask, kernel, iterations=3)
        self.basic_processing_image = cv2.bitwise_and(self.basic_processing_image, mask)
        self.basic_processing_image = self.resize(self.basic_processing_image, width, height)

    def show_image(self):
        image = self.resize(self.basic_processing_image, 800)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def show_target(self):
        image = self.resize(self.target, 800)
        cv2.imshow('target', image)
        cv2.waitKey(0)

    @staticmethod
    def test_image(image, name='test', size=200):
        image = Picture.resize(image, size, size)
        cv2.imshow(name, image)

    @staticmethod
    def read(path, method):
        return cv2.imread(path, method)

    @staticmethod
    def resize(image, width, height=None, interpolation=cv2.INTER_AREA):
        if height is None:
            (h, w) = image.shape[:2]
            ratio = w / width
            height = int(np.round(h / ratio))
        return cv2.resize(image, (width, height), interpolation=interpolation)

    @staticmethod
    def cutMiddleSquare(image):
        h, w = image.shape[0:2]
        hcut = int((h - 2048) / 2)
        wcut = int((w - 2048) / 2)
        return image[hcut:h - hcut, wcut:w - wcut]

    def get_segments(self, size=5, quantity=300):
        seed()
        imgToCut = Picture.cutMiddleSquare(self.original_image)
        imgToCheck = Picture.cutMiddleSquare(self.target)
        imgToCheckBinary = imgToCheck / 255

        h, w = imgToCheck.shape

        positiveSegment = []
        negativeSegment = []

        while (len(positiveSegment) != quantity or len(negativeSegment) != quantity):
            x = randint(0, w - size)
            y = randint(0, w - size)

            value = int(imgToCheckBinary[x][y])

            if value and len(positiveSegment) != quantity:
                positiveSegment.append(
                    Segment(imgToCut[x:x + size, y:y + size], imgToCheck[x:x + size, y:y + size]))
            elif not value and len(negativeSegment) != quantity:
                negativeSegment.append(
                    Segment(imgToCut[x:x + size, y:y + size], imgToCheck[x:x + size, y:y + size]))

        return positiveSegment, negativeSegment

