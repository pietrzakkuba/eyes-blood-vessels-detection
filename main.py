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

nnet = NeuralNetwork(size)

for j in range(4):
    for i in range(10):
        print('trening na obrazku:', i)
        data=pictures[i].get_segments(size, 5000)
        nnet.train2(data, n_split=4)

# nnet.load_model('my_model2')
print('pic11')
nnet.predictImage(pictures[10], size, 'pic11')
print('pic12')
nnet.predictImage(pictures[11], size, 'pic12')
print('pic13')
nnet.predictImage(pictures[12], size, 'pic13')
print('pic14')
nnet.predictImage(pictures[13], size, 'pic14')
print('pic15')
nnet.predictImage(pictures[14], size, 'pic15')




# evaluation(pictures[0].basic_processing_image, pictures[0].target)

# original_image = Picture.cutMiddleSquare(cv2.imread(pictures[1].original_path, cv2.IMREAD_GRAYSCALE))
# target_image = Picture.cutMiddleSquare(cv2.imread(pictures[1].target_path, cv2.IMREAD_GRAYSCALE))

# cv2.imshow('a', cv2.resize(target_image, (400, 400)))
# Picture.cutIntoSquares(target_image)
# cv2.waitKey(0)
cv2.waitKey(0)