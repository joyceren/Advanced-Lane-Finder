import numpy as np
import cv2

def color_gradient_thresholds(img):

    color_binary = color_thresh(img, r_thresh=(200, 255), s_thresh=(200, 255))
    mag_dir_binary = mag_dir_thresh(img, mkernal=9, dkernal=15, mag_thresh=(30, 100), dir_thresh=(0.7, 1.3))
    grad_binary = gradient_thresh(img, skernal=9, sobel_thresh=(30, 100))

    # drawing new image with only the pixels that are within the threshold bounds
    combined_binary = np.zeros_like(img[:,:,0])
    combined_binary[(color_binary == 1) | (grad_binary == 1) | (mag_dir_binary == 1)] = 1

    return combined_binary


def color_thresh(img, r_thresh=(0, 255), s_thresh=(0, 255)):
    r_color_channel = img[0:,0:,0]
    s_color_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]

    color_binary = np.zeros_like(s_color_channel)
    color_binary[
        (s_color_channel > s_thresh[0]) & (s_color_channel <= s_thresh[1]) |
        ((r_color_channel > r_thresh[0]) & (r_color_channel <= r_thresh[1]))
    ] = 1

    return color_binary

def mag_dir_thresh(img, mkernal, dkernal, mag_thresh=(0, 255), dir_thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=mkernal)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=mkernal)

    mag = np.sqrt(sobelx**2 + sobely**2)

    scaled_mag = np.uint8(255*mag / np.max(mag))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=dkernal)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=dkernal)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    mag_dir_binary = np.zeros_like(scaled_mag)
    mag_dir_binary[
        (scaled_mag >= mag_thresh[0]) &
        (scaled_mag <= mag_thresh[1]) &
        (absgraddir >= dir_thresh[0]) &
        (absgraddir <= dir_thresh[1])
        ] = 1

    return mag_dir_binary


def gradient_thresh(img, skernal=3, sobel_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=skernal)

    abs = np.abs(sobelx)
    scaled_sobel_x = np.uint8(255*abs/np.max(abs))

    grad_binary = np.zeros_like(img[:,:,0])
    grad_binary[(scaled_sobel_x > sobel_thresh[0]) & (scaled_sobel_x < sobel_thresh[1]) ] = 1

    return grad_binary