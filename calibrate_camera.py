import cv2
import numpy as np

# chessboard size
nx = 9
ny = 6

# objpoints and imgpoints for all images
objpoints = []
imgpoints = []

# preparing objpoints, formatted (0, 0, 0), (1, 0, 0), (2, 0, 0), etc...
objp = np.zeros((nx * ny, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

# stepping through calibration images
for i in range(1, 20):
        img = cv2.imread("camera_cal/calibration" + str(i) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # finding chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # if corner was found, add obj and img points
        if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

# get camera calibration variables
ret, cameraMatrix, distortionCoefficents, rotationVector, translationVector = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistort_image(img):
    return cv2.undistort(img, cameraMatrix, distortionCoefficents, None, cameraMatrix)