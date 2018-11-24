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
from color import init

if os.path.exists('calibration.p') is not True:
    print('Camera is not calibrated \nRun calibration.py')
    exit()

#Loading in the calibration profile
calibration = pickle.load(open('calibration.p', 'rb'))
mtx, dist = map(calibration.get, ('mtx', 'dist'))

#Original image
image = cv2.imread('test_images/straight_lines2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Calibrated image
undist = cv2.undistort(image, mtx, dist, None, mtx)

drawLines = draw(undist)

transform = perspectiveTransform(undist)

# threshold = thresholdingProcessing(transform)

test = init(transform)


histogram = fit_polynomial(test)

plt.imshow(test, cmap='gray')
#plt.imshow(undist)
plt.show()

# print(histogram[1])
# print(histogram[2])

# cap = cv2.VideoCapture('video.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     frame = init(frame)

#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




