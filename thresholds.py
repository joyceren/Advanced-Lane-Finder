import numpy as np
import cv2

def color_gradient_thresholds(img):

    # sobel threshold
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # abs = np.abs(sobelx)
    # scaled_sobel = np.uint8(255*abs/np.max(abs))


    # sobel thresholds
    kernal_size = 15
    sobel_thresh = (30, 100)
    mag_thresh = (30, 100)
    dir_thresh = (0.7, 1.3)

    # color thresholds
    s_thresholds = (150, 255)
    r_thresholds = (220, 255)

    color_binary = color_thresh(img, r_thresholds, s_thresholds)
    grad_binary = gradient_thresh(img, kernal_size, sobel_thresh, mag_thresh, dir_thresh)

    # drawing new image with only the pixels that are within the threshold bounds
    combined_binary = np.zeros_like(img[:,:,0])
    combined_binary[(color_binary == 1) | (grad_binary == 1)] = 1

    return combined_binary


def color_thresh(img, r_thresh=(0, 255), s_thresh=(0, 255)):
    r_color_channel = img[:,:,2]
    s_color_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]

    color_binary = np.zeros_like(img[:,:,0])
    color_binary[
        ((s_color_channel > s_thresh[0]) & (s_color_channel <= s_thresh[1])) |
        ((r_color_channel > r_thresh[0]) & (r_color_channel <= r_thresh[1]))
    ]

    return color_binary

def gradient_thresh(img, sobel_kernel=3, grad_thresh=(0, 255), mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs = np.abs(sobelx)
    scaled_sobel_x = np.uint8(255*abs/np.max(abs))

    mag = np.sqrt(sobelx**2 + sobely**2)

    scaled_mag = np.uint8(255*mag / np.max(mag))
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    grad_binary = np.zeros_like(img[:,:,0])
    grad_binary[((scaled_sobel_x > grad_thresh[0]) & (scaled_sobel_x < grad_thresh[1])) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return grad_binary