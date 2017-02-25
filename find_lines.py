import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def find_lines(img):
    '''
    This takes in a thresholded image and
    finds the lanes lines with sliding window search
    '''

    # These are window parameters
    window_width = 50
    window_height = 80  # Breaks image into 9 vertical layers since image height is 720
    margin = 100  # How much to slide left and right for searching

    def window_mask(width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return output

    def find_window_centroids(image, window_width, window_height, margin):

        window_centroids = []  # Stores the (left,right) window centroid positions per level
        window = np.ones(window_width)  # Creates a window template that will be used for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # This sums quarter bottom of image to get slice
        l_sum = np.sum(img[int(3 * img.shape[0] / 4):, :int(img.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(img[int(3 * img.shape[0] / 4):, int(img.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(img.shape[1] / 2)

        # Distance from center (camera center - lane center)
        cc = int(img.shape[1] / 2)
        lc = int((r_center - l_center) / 2)
        dfc = abs(cc - lc)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # This goes through each layer looking for max pixel locations
        for level in range(1, (int)(img.shape[0] / window_height)):
            # convolves the window into the vertical slice of the image
            image_layer = np.sum(
                img[int(img.shape[0] - (level + 1) * window_height):int(img.shape[0] - level * window_height),
                :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Finds the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, img.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Finds the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, img.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what was found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids, dfc

    window_centroids, dfc = find_window_centroids(img, window_width, window_height, margin)

    # These are empty arrays that will later store values for the polyfit
    leftx = []
    rightx = []
    yaxis = []

    # If any window centers were found
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)


        # Goes through each level and draws the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Adds graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255
            ycalc = img.shape[0] - ((level * 2 + 1) * (window_height / 2))

            yaxis.append(ycalc)
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

        yaxis = np.array(yaxis)
        rightx = np.array(rightx)
        leftx = np.array(leftx)

        # This draws the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((img, img, img)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, this displays orginal road image
    else:
        output = np.array(cv2.merge((img, img, img)), np.uint8)


    return output, yaxis, leftx, rightx, dfc