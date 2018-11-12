import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
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
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def whiteDetection(image):

    def bin_it(image, threshold):
        bin = np.zeros_like(image)
        bin[(image >= threshold[0]) & (image <= threshold[1])] = 1
        return bin

    bin_thresh = [20,255]

    lower = np.array([100,100,200])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.bitwise_and(image, image, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = bin_it(mask, bin_thresh)
    return mask

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

def thresholdingProcessing(image):

    ksize=3

    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(10, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(25, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(1.0, np.pi/2))

    yellow = yellowDetection(image)
    white = whiteDetection(image)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (white == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (yellow == 1) | (white == 1)] = 1
    return combined