import cv2

#Loading in the calibration profile

def draw(image):
    copy = image.copy()

    bottomY = 720
    topY = 465


    left1 = (191, bottomY)
    left2 = (566, topY)

    right1 = (721, topY)

    right2 = (1122, bottomY)

    color = [255, 0, 0]
    w = 2
    cv2.line(copy, left1, left2, color, w)
    cv2.line(copy, left2, right1, color, w)
    cv2.line(copy, right1, right2, color, w)
    cv2.line(copy, right2, left1, color, w)
    
    return copy
