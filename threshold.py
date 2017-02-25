import numpy as np
import cv2


def abs_sobel_thresh(img, sx_thresh):
    '''
    This function takes an image, and threshold
    min / max values and returns a sobel binary threshold.
    '''

    # This takes the derivative in x
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)

    # This takes absolute value of x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)

    # This rescales back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # This creates a copy and apply the threshold
    sxbinary = np.zeros_like(scaled_sobel)

    # Here I'm using inclusive (>=, <=) thresholds
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    return sxbinary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    '''
    This function returns the magnitude of the gradient
    for a given sobel kernal size and threshold values
    '''

    # This takes both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # This calculates the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # This rescales to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # This creates a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output



def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    This function thresholds an image for a given range and Sobel kernel
    '''

    # This calculates the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # This takes the absolute value of the gradient direction, apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

# def color_threshold(s_channel, s_thresh):
#     s_binary = np.zeros_like(s_channel)
#     s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
#
#     return s_binary