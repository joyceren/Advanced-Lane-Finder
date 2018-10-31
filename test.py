import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from calibrate_camera import calibrate_camera

ret, cameraMatrix, distortionCoefficents = calibrate_camera()



# for i in range(1, 20):
#     img = mpimg.imread("camera_cal/calibration" + str(i) + '.jpg')
#     plt.imshow(cv2.undistort(img, cameraMatrix, distortionCoefficents, None, cameraMatrix))
#     plt.show()

# for imageName in os.listdir("test_images"):
#     img = mpimg.imread('test_images/' + imageName)
#     plt.imshow(cv2.undistort(img, cameraMatrix, distortionCoefficents, None, cameraMatrix))
#     plt.show()
