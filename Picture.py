import numpy as np
import cv2


class Picture:
    def __init__(self, original_path, fovmask_path, target_path):
        self.original_path = original_path
        self.fovmask_path = fovmask_path
        self.target_path = target_path
        self.image = None
        self.target = self.read(target_path, cv2.IMREAD_GRAYSCALE)

    def adjust_gamma(self, gamma):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def process_image(self):
        self.image = self.read(self.original_path, cv2.IMREAD_COLOR)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height = self.image.shape[0]
        width = self.image.shape[1]
        self.image = self.resize(self.image, 800)
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
        mask = self.read(self.fovmask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.resize(mask, 800)
        mask = cv2.erode(mask, kernel, iterations=3)
        self.image = cv2.bitwise_and(self.image, mask)
        self.image = self.resize(self.image, width, height)

    def show_image(self):
        image = self.resize(self.image, 800)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def show_target(self):
        image = self.resize(self.target, 800)
        cv2.imshow('target', image)
        cv2.waitKey(0)

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
        h, w = image.shape
        hcut = int((h - 2048) / 2)
        wcut = int((w - 2048) / 2)
        return image[hcut:h - hcut, wcut:w - wcut]

    @staticmethod
    def cutIntoSquares(image, square_size=512):
        div = int(2048 / square_size)
        squares = []
        for i in range(div):
            for j in range(div):
                squares.append(image[i * square_size:(i + 1) * square_size, j * square_size:(j + 1) * square_size])
        for i in range(len(squares)):
            squares = [cv2.resize(square, (256, 256)) for square in squares]
            cv2.putText(squares[i], str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                        cv2.LINE_AA)
            # cv2.imshow(str(i), squares[i])
        return squares

    def get_labels(self):
        labels = np.array([list(map(int, row / 255))
                           for row in list(cv2.threshold(self.target, 127, 255, cv2.THRESH_BINARY)[1])])
        return labels
