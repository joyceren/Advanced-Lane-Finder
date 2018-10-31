import cv2
import numpy as np

# Project Steps:
#     + Camera calibration
#     + Distortion correction
#     - Color/gradient threshold
#     - Perspective transform
#     - Detect lane lines
#     - Determine the lane curvature



def color_gradient_thresholds(img):
    


def perspective_transform(img):
        srcPoints = np.float32([
            [200/960*img.shape[1], 500/540*img.shape[0]),
            [420/960*img.shape[1], 340/540*img.shape[0]),
            [550/960*img.shape[1], 340/540*img.shape[0]),
            [800/960*img.shape[1], 500/540*img.shape[0]),
        ])

        destinationPoints = np.float32([
            [100, 100],
            [1200, 100],
            [1200, 875],
            [100, 875] ,
        ])

    M = cv2.getPerspectiveTransform(srcPoints, destinationPoints)
    warped = cv2.warpPerspective(img, M, img_size)

    return warped

def find_lane_lines():
    pass


def lane_curves():
    pass