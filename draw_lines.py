import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_lines(warped, image, left_fitx, right_fitx, ploty, Minv, undist, radii, dfc):
    '''
    This function takes in the warped image and actually fills in the lane
    on the image
    '''

    # This creates an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # This recasts the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # This draws the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # This warps the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # This combines the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    radius = (radii[0] + radii[1]) / 2
    text1 = "Radius of curvature is: " + str(round(radius, 2)) + 'm'
    text2 = "Vehicle is " + str(round(dfc, 2)) + "m from center"
    result = cv2.putText(result, text1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    result = cv2.putText(result, text2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))


    return result