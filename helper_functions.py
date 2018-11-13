import cv2
import numpy as np
import matplotlib.pyplot as plt


def fit_line(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    margin = 80
    windows = 9
    window_height = np.int(img.shape[0] // windows)
    
    # splitting histogram into left lane and right lane
    midpoint = np.int(histogram.shape[0]/2)

    # highest histogram point on left and right side of image
    # this is the x-value where the each line starts
    # we will change this as we step through the windows
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint

    # Identify non-zero pixels in image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # lists for lanes indices
    left_lane_points = []
    right_lane_points = []

    # looping through all windows
    for window in range(windows):
        upper_y_boundary = img.shape[0] - window_height*(window)
        lower_y_boundary = img.shape[0] - window_height*(window+1)

        # creating ;the window bounds
        leftx_lower_bound = leftx_current - margin
        leftx_upper_bound = leftx_current + margin
        rightx_lower_bound = rightx_current - margin
        rightx_upper_bound = rightx_current + margin

        # checking for nonzero points within the window bounds
        valid_left_points = ((nonzeroy >= lower_y_boundary) &
                            (nonzeroy < upper_y_boundary) & 
                            (nonzerox >= leftx_lower_bound) &
                            (nonzerox < leftx_upper_bound)).nonzero()[0]

        valid_right_points = ((nonzeroy >= lower_y_boundary) &
                            (nonzeroy < upper_y_boundary) & 
                            (nonzerox >= rightx_lower_bound) &
                            (nonzerox < rightx_upper_bound)).nonzero()[0]

        # add found points to cummulative list
        left_lane_points.append(valid_left_points)
        right_lane_points.append(valid_right_points)

        # if there are more than 50 valid points, next window center is the average of all the valid lane points found
        if len(valid_left_points) > 50:
            leftx_current = np.int(np.mean(nonzerox[valid_left_points]))
        if len(valid_right_points) > 50:
            rightx_current = np.int(np.mean(nonzerox[valid_right_points]))

    left_lane_points = np.concatenate(left_lane_points)
    right_lane_points = np.concatenate(right_lane_points)

    # get coordinates of valid pixels nonzero pixels
    leftx = nonzerox[left_lane_points]
    lefty = nonzeroy[left_lane_points] 
    rightx = nonzerox[right_lane_points]
    righty = nonzeroy[right_lane_points]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, left_fitx, right_fitx
