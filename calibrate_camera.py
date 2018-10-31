import cv2
import numpy as np


def calibrate_camera():

    nx = 9
    ny = 6

    objpoints = []
    imgpoints = []

    for i in range(1, 20):
        img = cv2.imread("camera_cal/calibration" + str(i) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, cameraMatrix, distortionCoefficents, rotationVector, translationVector = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, cameraMatrix, distortionCoefficents
