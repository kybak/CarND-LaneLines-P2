import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle

# This reads in all the calibration images and adds them to a list
images = glob.glob('camera_cal/calibration*.jpg')

#These are arrays that will store the object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# This prepare object points, like (0,0,0), (1,0,0), (2,0,0)....,(7,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinated

for idx, fname in enumerate(images):
    # This reads in each image file
    img = mpimg.imread(fname)

    # This converts image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # This finds the chessboard corners, taking in the grayscale chessboard image and the dimensions of the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # This adds object points to the imgpoints array if there are corners
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # This draws and displays the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        plt.imshow(img)
        # plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    file_path = "calibration_wide/" + str(idx) + '.p'
    pickle.dump(dist_pickle, open(file_path, "wb"))




