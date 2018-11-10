import cv2
import numpy as np
from thresholds import color_thresh, gradient_thresh

# Project Steps:
#     - Camera calibration
#     - Distortion correction
#     - Color/gradient thresholds
#            - SobelX Operator
#            - S Color Threshold
#            ? - Magnitude of Gradient
#            ? - Direction of Gradient
#            ? - R Color Threshold
#     - Perspective transform
#     - Detect lane lines
#     - Determine the lane curvature


# Change to birds-eye view
def perspective_transform(img):

    srcPoints = np.float32([
        # x-value, y-value
        [img.shape[1] / 2 - 55, img.shape[0] / 2 + 100], # top left corner
        [img.shape[1] / 2 + 55, img.shape[0] / 2 + 100], # top right corner
        [img.shape[1] * 5 / 6, img.shape[0]], # bottom right corner
        [img.shape[1] / 6, img.shape[0]] # bottom left corner
    ])

    destinationPoints = np.float32([
        [img.shape[1] / 4, 0], # top left corner
        [img.shape[1] * 3 / 4, 0], # top right corner
        [img.shape[1] *3 / 4, img.shape[0]], # bottom right corner
        [img.shape[1] / 4, img.shape[0]], # bottom left corner
    ])

    M = cv2.getPerspectiveTransform(srcPoints, destinationPoints)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return warped


def histogram(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    return histogram


def find_lanes(img, windows, margin, left_lane_x_starting_point, right_lane_x_starting_point, nonzerox, nonzeroy):
    # Current positions to be updated later for each window in nwindows
    leftx_current = left_lane_x_starting_point
    rightx_current = right_lane_x_starting_point

    window_height = np.int(img.shape[0] // windows)

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

    # combine all lists
    left_lane_points = np.concatenate(left_lane_points)
    right_lane_points = np.concatenate(right_lane_points)

    return left_lane_points, right_lane_points


def plot_lines(img):
    hist = histogram(img)
    
    # splitting histogram into left lane and right lane
    midpoint = np.int(hist.shape[0]/2)

    # highest histogram point on left and right side of image
    # this is the x-value where the each line starts
    left_lane_x_starting_point = np.argmax(hist[:midpoint])
    right_lane_x_starting_point = np.argmax(hist[midpoint:]) + midpoint

    # Identify non-zero pixels in image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_points, right_lane_points = find_lanes(img, 9, 100, left_lane_x_starting_point, right_lane_x_starting_point, nonzerox, nonzeroy)

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


def lane_curves():
    pass