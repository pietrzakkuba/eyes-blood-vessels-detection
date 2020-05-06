from os import listdir
from os.path import isfile, join

import numpy as np
from numpy import float32

from Picture import Picture
from NeuralNetwork import NeuralNetwork
from efficiencyevaluation import evaluation
import cv2


def get_pictures():
    healthy = 'pictures\\healthy'
    healthy_fovmask = 'pictures\\healthy_fovmask'
    healthy_manual = 'pictures\\healthy_manualsegm'
    healthy_paths = [join(healthy, f) for f in listdir(healthy)]
    healthy_fovmask_paths = [join(healthy_fovmask, f) for f in listdir(healthy_fovmask)]
    healthy_manual_paths = [join(healthy_manual, f) for f in listdir(healthy_manual)]
    return [Picture(healthy_paths[i], healthy_fovmask_paths[i], healthy_manual_paths[i])
            for i in range(len(healthy_paths))]


pictures = get_pictures()
size = 65
data = pictures[0].get_segments(size, 900)
nnet = NeuralNetwork(size)
# nnet.train(data)
# nnet.train2(data, n_split=2)

nnet.load_model('my_model2')
nnet.predictImage(pictures[1], size)




# evaluation(pictures[0].basic_processing_image, pictures[0].target)

# original_image = Picture.cutMiddleSquare(cv2.imread(pictures[1].original_path, cv2.IMREAD_GRAYSCALE))
# target_image = Picture.cutMiddleSquare(cv2.imread(pictures[1].target_path, cv2.IMREAD_GRAYSCALE))

# cv2.imshow('a', cv2.resize(target_image, (400, 400)))
# Picture.cutIntoSquares(target_image)
# cv2.waitKey(0)
