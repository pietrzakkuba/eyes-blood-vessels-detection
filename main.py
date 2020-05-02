import numpy
import matplotlib
import cv2
from os import listdir
from os.path import isfile, join


class Picture:
    def __init__(self, originalPath, targetPaths):
        self.originalPath=originalPath
        self.targetPaths=targetPaths

dirlist=[join('pictures', f) for f in listdir('pictures')]
dirlist=[x for x in dirlist if 'fovmask' not in x][:-1]
print(dirlist)

pictureList=[]

locdirlist = [join(dirlist[0], f) for f in listdir(dirlist[0])]
for i in range(0, len(locdirlist), 3):
    pictureList.append(Picture(locdirlist[i], [locdirlist[i + 1], locdirlist[i + 2]]))

for i in range(1, len(dirlist)-1, 2):
    locdirlist=[[join(dirlist[i], f) for f in listdir(dirlist[i])], [join(dirlist[i+1], f) for f in listdir(dirlist[i+1])]]
    for j in range(len(locdirlist[0])):
        pictureList.append(Picture(locdirlist[0][j], locdirlist[1][j]))

for pic in pictureList:
    print(pic.originalPath, '-', pic.targetPaths)