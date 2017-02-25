from threshold import abs_sobel_thresh, mag_thresh, dir_threshold
from perspective_transorm import warp
from draw_lines import draw_lines
from find_lines import find_lines
from Line import Line
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip

Line = Line()

def process_video(image):
    # image = mpimg.imread('test_images/test5.jpg')
    # image = mpimg.imread('test_images/straight_lines1.jpg')

    # This converts the image to grayscale --NOT IN USE--
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1 --NOT IN USE--

    # This plots the result
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    #
    # ax1.imshow(image)
    # ax1.set_title('Original Image', fontsize=50)
    #
    # ax2.imshow(combined, cmap='gray')
    # ax2.set_title('Thresholded Magnitude', fontsize=50)
    #
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()

    # ############################## PERSPECTIVE TRANSFORM ############################## #

    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open( "calibration_wide/19.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]


    # Read in an image
    # img = mpimg.imread('test_images/straight_lines1.jpg')
    # plt.imshow(img)

    warped_im, undist, Minv = warp(combined, mtx, dist)

    # ############################## FIND LINES AND RADII ############################## #

    def get_radius(left_fit, right_fit):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        return left_curverad, right_curverad
        # Example values: 1926.74 1908.48

    lines, yaxis, leftx, rightx = find_lines(warped_im, Line)

    if (len(Line.recent_xfitted) >= 50):
        Line.recent_xfitted[0].insert(0, leftx[0])
        Line.recent_xfitted[1].insert(0, rightx[0])
        Line.recent_xfitted[0].pop()
        Line.recent_xfitted[1].pop()

        Line.bestx[0] = np.average(Line.recent_xfitted[0])
        Line.bestx[1] = np.average(Line.recent_xfitted[1])
    else:
        if (len(Line.bestx) == 0):
            Line.bestx.append(leftx[0])
            Line.bestx.append(rightx[0])
        else:
            Line.bestx[0] = leftx[0]
            Line.bestx[1] = rightx[0]

        Line.recent_xfitted[0].append(leftx[0])
        Line.recent_xfitted[1].append(rightx[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Polyfit
    left_fit = np.polyfit(yaxis, leftx, 2)
    right_fit = np.polyfit(yaxis, rightx, 2)

    ploty = np.linspace(0, warped_im.shape[0] - 1, warped_im.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_rad, right_rad = get_radius(left_fitx, right_fitx)
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    #
    # plt.plot(leftx, ploty, color='yellow')
    # plt.plot(rightx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    #
    # ax1.set_title('Source Image')
    # ax1.imshow(warped_im)
    # ax2.set_title('Warped Image')
    # ax2.imshow(lines)
    # plt.show()

    # ############################## DRAW LINES ############################## #



    result = draw_lines(warped_im, image, left_fitx, right_fitx, ploty, Minv, undist)

    return result


# ############################## PLAY VIDEO ############################## #


output = 'output3.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
white_clip = clip1.fl_image(process_video) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)