import numpy as np
import cv2
from random import randint, seed
from Segment import Segment
import sys
import NeuralNetwork


# imgToCut[:, :, 2]=0

class Picture:
    def __init__(self, original_path, fovmask_path, target_path):
        self.size = 65
        self.original_path = original_path
        self.fovmask_path = fovmask_path
        self.target_path = target_path
        self.basic_processing_image = None
        self.advanced_processing_image = None
        self.network = NeuralNetwork.NeuralNetwork(self.size)
        self.target = self.read(target_path, cv2.IMREAD_GRAYSCALE)
        self.fovmask = self.read(fovmask_path, cv2.IMREAD_GRAYSCALE)
        self.true_original = self.read(self.original_path, cv2.IMREAD_COLOR)
        self.original_image = cv2.GaussianBlur(self.true_original[:, :, 1], (21, 21), 0)
        self.segments_list = []

    def adjust_gamma(self, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.basic_processing_image = cv2.LUT(self.basic_processing_image, table)

    def process_image_basic(self):
        self.basic_processing_image = cv2.adaptiveThreshold(self.original_image,
                                                          255,
                                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY_INV,
                                                          45,
                                                          2)
        Picture.showImage(self.true_original, 'original')
        Picture.showImage(self.original_image, 'original - green channel')
        cv2.imwrite('result\\basic processing result image.jpg', self.basic_processing_image)
        self.show_image()

    def process_image_advanced(self):
        # self.advanced_processing_image
        self.network.load_model('my_model2')
        self.advanced_processing_image = self.network.predictImage(self,
                                                                   self.size,
                                                                   'advanced processing result image')
        Picture.showImage(self.advanced_processing_image, 'advanced processing result')


    def show_image(self):
        image = self.resize(self.basic_processing_image, 800)
        cv2.imshow('basic processing result', image)
        cv2.waitKey(0)

    def show_target(self):
        image = self.resize(self.target, 800)
        cv2.imshow('target', image)
        cv2.waitKey(0)

    @staticmethod
    def test_image(image, name='test', size=200):
        image = Picture.resize(image, size)
        cv2.imshow(name, image)

    @staticmethod
    def showImage(image, windowname='test', newh=600):
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        h, w = image.shape[:2]
        neww = int((newh / h) * w)
        cv2.resizeWindow(windowname, neww, newh)
        cv2.imshow(windowname, image)

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

    def get_segments(self, size=65, quantity=500):
        seed()
        imgToCut = Picture.cutMiddleSquare(self.original_image)
        imgToCheck = Picture.cutMiddleSquare(self.target)
        imgToCheckBinary = imgToCheck / 255

        h, w = imgToCheck.shape

        segments=[]
        positive=0
        negative=0

        while positive != quantity or negative != quantity:
            x = randint(0, w - size)
            y = randint(0, w - size)
            value = int(imgToCheckBinary[x][y])
            if value and positive != quantity:
                segments.append(
                    Segment(imgToCut[x:x + size, y:y + size], value))
                positive+=1
            elif not value and negative != quantity:
                segments.append(
                    Segment(imgToCut[x:x + size, y:y + size], value))
                negative+=1
        return segments

