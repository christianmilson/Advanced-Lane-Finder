import numpy as np
import cv2

def perspectiveTransform(img, reverse = False):

    img_size = (img.shape[1], img.shape[0])
    
    leftupperpoint  = [220,720]
    leftlowerpoint  = [1110,720]
    rightupperpoint = [570,470]
    rightlowerpoint = [722,470]
    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[200,720], [1000,720], [200,1], [1000, 1]])
    # Given src and dst points, calculate the perspective transform matrix
    if reverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    normal = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return normal