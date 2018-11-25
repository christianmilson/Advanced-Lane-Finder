import numpy as np
import cv2

def perspectiveTransform(img):

    img_size = (img.shape[1], img.shape[0])
    
    leftupperpoint  = [490,482]
    leftlowerpoint  = [810,482]
    rightupperpoint = [1250,720]
    rightlowerpoint = [40,720]
    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[0,0], [1280,0], [1250,720], [40,720]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped

def reverse(img):
    img_size = (img.shape[1], img.shape[0])
    
    leftupperpoint  = [490,482]
    leftlowerpoint  = [810,482]
    rightupperpoint = [1250,720]
    rightlowerpoint = [40,720]
    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[0,0], [1280,0], [1250,720], [40,720]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    normal = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return normal