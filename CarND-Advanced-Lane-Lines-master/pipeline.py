import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import os
from perspective_transform import perspectiveTransform, reverse
from moviepy.editor import VideoFileClip
from histogram import find_lanes
from color import init

if os.path.exists('calibration.p') is not True:
    print('Camera is not calibrated \nRun calibration.py')
    exit()

#Loading in the calibration profile
calibration = pickle.load(open('calibration.p', 'rb'))
mtx, dist = map(calibration.get, ('mtx', 'dist'))

def pipeline(image):
    
    #Calibrates the Image
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    transform = perspectiveTransform(undist)

    test = init(transform)

    histogram = find_lanes(test)

    result = cv2.addWeighted(undist, 1, histogram, 0.3, 0)

    return result

output = 'output2.mp4'
clip1 = VideoFileClip("project_video.mp4")

white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)





