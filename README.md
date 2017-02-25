##Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in "camera_calibration.py"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in another_file.py). This process is broken up into four steps. First I converted the image to HSV because that gave better contrast under a variety of lighting conditions. The rest of the threshold steps can be found in "threshold.py" where I use three different thresholding techniques to further transform the image. Here's an example of my output for this step.
![alt text][image3]
####2. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one: 
![alt text][image2]

The code for distortion correction can be found in "perspective_transform.py". It is done with cv2.undistort given the mtx and mdist values that were calculated and saved during the camera calibration process.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called warp(), which appears on line 49 in the file "detect_lane_lines.py" and the innards are in "perspective_transform.py". The warp() function takes as inputs an image (combined), as well as matrix (mtx) and destination (dst) points. I chose to hardcode the mtx and destination points by grabbing them from previously saved values during the camera calibration.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I then found the lines of the warped image as seen in "find_lines.py". This function uses the sliding window search approach. It returns a yaxis, leftx, and rightx values in order to fid a 2nd order polynomial using the equation f(y) = Ay^2 + By + C. This resulted in the image below.

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 53 through 71 in my code in `detect_lane_lines.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step on line 94 in my code in `detect_lane_lines.py` in the function `draw_lines()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had problems on the challenge videos. I tried saving n past x values and using average values whenever my lines were being detected incorrectly. I determined they were incorrect by testing to see they difference between the left and right lines were below some threshold. If I could spend more time on this I would continue with this approach because I think I'm on the right track though I'm not quite sure where I went wrong. I just need more time. As it is my pipeline will definitely fail whenever the road is not pristine. For instance, if there is some semblance of a line that is actually a crack then it will detect it as a lane line. I would solve this by making sure the left and right lines are some distance away from the center. Of course there are many other things that could be revised or improved upon and I look forward to continuing with this project in the future. 
