import cv2
import numpy as np

def warp(img, mtx, mdist):
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


# # Define a function that takes an image, number of x and y points,
# # camera matrix and distortion coefficients
# def corners_unwarp(img, nx, ny, mtx, dist):
#     # Use the OpenCV undistort() function to remove distortion
#     # Convert undistorted image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Search for corners in the grayscaled image
#     ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
#     # undist = cv2.undistort(gray, mtx, dist, None, mtx)
#     undist = gray
#
#     if ret == True:
#         # If we found corners, draw them! (just for fun)
#         cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
#         # Choose offset from image corners to plot detected corners
#         # This should be chosen to present the result at the proper aspect ratio
#         # My choice of 100 pixels is not exact, but close enough for our purpose here
#         offset = 100 # offset for dst points
#         # Grab the image shape
#         img_size = (gray.shape[1], gray.shape[0])
#
#         # For source points I'm grabbing the outer four detected corners
#         src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#         # For destination points, I'm arbitrarily choosing some points to be
#         # a nice fit for displaying our warped result
#         # again, not exact, but close enough for our purposes
#         dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
#                                      [img_size[0]-offset, img_size[1]-offset],
#                                      [offset, img_size[1]-offset]])
#         # Given src and dst points, calculate the perspective transform matrix
#         M = cv2.getPerspectiveTransform(src, dst)
#         # Warp the image using OpenCV warpPerspective()
#         warped = cv2.warpPerspective(undist, M, img_size)
#
#     # Return the resulting image and matrix
#     return warped, M