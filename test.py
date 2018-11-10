import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from calibrate_camera import undistort_image
from thresholds import color_gradient_thresholds
from lane_finder import perspective_transform, plot_lines


# for imageName in os.listdir("test_images"):
    # img = mpimg.imread('test_images/' + imageName)
img = mpimg.imread('test_images/test4.jpg')
plt.imshow(img)
plt.show()


undistortedImg = undistort_image(img)
afterGradient = color_gradient_thresholds(undistortedImg)
transformedImg = perspective_transform(afterGradient)
ploty, left_fitx, right_fitx = plot_lines(transformedImg)

# Plots the left and right polynomials on the lane lines
plt.plot(left_fitx, ploty, color='red')
plt.plot(right_fitx, ploty, color='red')

plt.imshow(transformedImg)
plt.show()


