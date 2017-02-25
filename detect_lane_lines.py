from threshold import abs_sobel_thresh, mag_thresh, dir_threshold
from perspective_transorm import warp
from draw_lines import draw_lines
from find_lines import find_lines
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip


def process_video(image):
    '''
    This function takes in an image either from a
    sequence of images or a single image and calls
    all methods responsible for undistorting,
    warping, and thresholding the image.
    '''

    img = np.copy(image)

    # This converts the image to HLS
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # This grabs the binary values
    gradx = abs_sobel_thresh(l_channel, sx_thresh = (7, 255))
    mag_binary = mag_thresh(s_channel, sobel_kernel=15, mag_thresh=(50, 255))
    dir_binary = dir_threshold(s_channel, sobel_kernel=21, thresh=(0, 1.3))
    # s_binary = color_threshold(s_channel, s_thresh=(170, 255)) --NOT IN USE--

    # This combines all the binary thresholds into one so that each can contribute its advantages
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1



    # ############################## PERSPECTIVE TRANSFORM ############################## #

    # This reads in the saved camera matrix and distortion coefficients
    # These are the arrays calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "calibration_wide/19.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # This is the perspective transform method
    warped_im, undist, Minv = warp(combined, mtx, dist)

    # ############################## FIND LINES AND RADII ############################## #

    # This defines conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    def get_radius(left_fit, right_fit):
        '''
        This function determines the radii from the polyfit
        '''

        # This gets a new polyfit with pixel-meter conversions
        left_fit = np.polyfit(yaxis * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit = np.polyfit(yaxis * ym_per_pix, rightx * xm_per_pix, 2)

        # This defines a y-value where we want radius of curvature
        # The maximum y-value is chosen, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        return left_curverad, right_curverad


    # This finds the lines through sliding window search
    lines, yaxis, leftx, rightx, dfc = find_lines(warped_im)

    dfc = dfc * ym_per_pix * .01

    # Polyfit
    left_fit = np.polyfit(yaxis, leftx, 2)
    right_fit = np.polyfit(yaxis, rightx, 2)

    ploty = np.linspace(0, warped_im.shape[0] - 1, warped_im.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # This is the radius of curvature
    left_rad, right_rad = get_radius(leftx, rightx)


    # ############################## DRAW LINES ############################## #


    # This draws the lines on an image either of a sequence of images or on a single image.
    result = draw_lines(warped_im, image, left_fitx, right_fitx, ploty, Minv, undist, [left_rad, right_rad], dfc)

    return result

# This is used on only single images --NOT IN USE--
# image = mpimg.imread('test_images/test5.jpg')
# process_video(image)


# ############################## PLAY VIDEO ############################## #

# This reads in the video file and calls the function to process the video and detect the lane lines.
output = 'output3.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_video) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)