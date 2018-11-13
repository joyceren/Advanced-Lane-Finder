import cv2
import numpy as np
import matplotlib.pyplot as plt

from thresholds import color_gradient_thresholds
from helper_functions import fit_line

# Project Steps:
#     - Camera calibration
#     - Distortion correction
#     - Color/gradient thresholds
#            - SobelX Operator
#            - S Color Threshold
#            - R Color Threshold
#            - Magnitude of Gradient
#            - Direction of Gradient
#     - Perspective transform
#     - Detect lane lines
#     - Determine the lane curvature
#     - Draw lines
#     - Perspective revert


class LaneFinder:
    def __init__(self, mtx, dist):
        self.cameraMatrix = mtx
        self.distCoeffs = dist

    def undistort(self, img):
        self.undistorted_img = cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)

    def thresholds(self):
        self.thresholds_binary = color_gradient_thresholds(self.undistorted_img)

    def perspective_transform(self):
        img = self.thresholds_binary
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

        self.M = cv2.getPerspectiveTransform(srcPoints, destinationPoints)
        self.Minv = cv2.getPerspectiveTransform(destinationPoints, srcPoints)
        self.warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0])) 

    def draw_lines(self):
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        self.color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(self.color_warp, np.int_([pts]), (0,255, 0))

    def perspective_revert(self):
        newwarp = cv2.warpPerspective(self.color_warp, self.Minv, (self.undistorted_img.shape[1], self.undistorted_img.shape[0]))
        self.result = cv2.addWeighted(self.undistorted_img, 1, newwarp, 0.3, 0)

    def run(self, img):
        self.undistort(img)
        self.thresholds()
        self.perspective_transform()
        self.ploty, self.left_fitx, self.right_fitx = fit_line(self.warped)
        self.draw_lines()
        self.perspective_revert()

        return self.result







