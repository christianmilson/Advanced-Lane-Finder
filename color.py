import cv2
import numpy as np

def threshold(rgb_image):
    gray = convert_gray(rgb_image)

    white = whiteDetection(rgb_image)
    yellow = yellowDetection(rgb_image)

    color_space = np.zeros_like(yellow)
    color_space[(white == 1) | (yellow == 1)] = 1

    gradx = abs_sobel_thresh(gray, orient='x', thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', thresh=(20,100))
    mag_binary = mag_thresh(gray, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    
    edge_detection = np.zeros_like(dir_binary)
    edge_detection[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    result = np.zeros_like(edge_detection)
    result[(edge_detection == 1) | (color_space == 1)] = 1

    return result


def whiteDetection(image):

    l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)[:,:,0]
    thresh_min = 225
    thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1
    return l_binary

def yellowDetection(image):

    def bin_it(image, threshold):
        bin = np.zeros_like(image)
        bin[(image >= threshold[0]) & (image <= threshold[1])] = 1
        return bin

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    S = hls[:,:,2]
    H = hls[:,:,1]
    
    bin_thresh = [20,255]

    lower = np.array([20,100,80])
    upper = np.array([45,200,255])
    mask_HSL = cv2.inRange(hls, lower, upper)
    mask_HSL = cv2.bitwise_and(image, image, mask = mask_HSL)
    mask_HSL = cv2.cvtColor(mask_HSL, cv2.COLOR_HLS2RGB)
    mask_HSL = cv2.cvtColor(mask_HSL, cv2.COLOR_RGB2GRAY)
    mask_HSL = bin_it(mask_HSL, bin_thresh)

    lower = np.array([0,80,200])
    upper = np.array([40,255,255])
    mask_HSV = cv2.inRange(hsv, lower, upper)
    mask_HSV = cv2.bitwise_and(image, image, mask = mask_HSV)
    mask_HSV = cv2.cvtColor(mask_HSV, cv2.COLOR_HSV2RGB)
    mask_HSV = cv2.cvtColor(mask_HSV, cv2.COLOR_RGB2GRAY)
    mask_HSV = bin_it(mask_HSV, bin_thresh)

    mask = np.zeros_like(mask_HSV)
    mask[(mask_HSV == 1) | (mask_HSL == 1)] = 1
    return mask

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def convert_gray(rgb_image):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return gray