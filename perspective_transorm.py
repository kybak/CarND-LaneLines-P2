import cv2
import numpy as np


def warp(img, mtx, mdist):
    '''
    This is the function responsible for perspective transform.
    '''

    img_size = (img.shape[1], img.shape[0])
    undist = cv2.undistort(img, mtx, mdist, None, mtx)

    # Four source coordinates
    src = np.float32(
        [[672, 442],
         [1108, 719],
         [207, 719],
         [600, 447]])

    dst = np.float32(
        [[980, 0],
         [980, 719],
         [330, 719],
         [330, 0]])

    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)


    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)


    return warped, undist, Minv