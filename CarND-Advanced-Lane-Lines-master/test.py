import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
from draw import draw
from perspective_transform import perspectiveTransform
from thresholding import thresholdingProcessing
from moviepy.editor import VideoFileClip
from histogram import fit_polynomial

if os.path.exists('calibration.p') is not True:
    print('Camera is not calibrated \nRun calibration.py')
    exit()

#Loading in the calibration profile
calibration = pickle.load(open('calibration.p', 'rb'))
mtx, dist = map(calibration.get, ('mtx', 'dist'))

#Original image
image = cv2.imread('test_images/test4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Calibrated image
undist = cv2.undistort(image, mtx, dist, None, mtx)

drawLines = draw(undist)

transform = perspectiveTransform(undist)

threshold = thresholdingProcessing(transform)

histogram = fit_polynomial(threshold)

plt.imshow(histogram[0])
#plt.imshow(undist)
plt.show()

print(histogram[1])
print(histogram[2])





