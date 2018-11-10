import numpy as np
import cv2

def color_gradient_thresholds(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # sobel thresholds
    g_thresh = (30, 100)
    m_thresh = (30, 100)
    d_thresh = (0.7, 1.3)

    # color thresholds
    s_thresholds = (150, 255)
    r_thresholds = (220, 255)

    color_binary = color_thresh(img, r_thresholds, s_thresholds)
    grad_binary = gradient_thresh(gray, 5, g_thresh)
    mag_binary = mag_thresh(gray, 9, m_thresh)
    dir_binary = dir_thresh(gray, 15, d_thresh)

    print("g=", grad_binary)
    print("m=", mag_binary)
    print("d=", dir_binary)

    combined_grad_binary = np.zeros_like(img[:,:,0])
    combined_grad_binary[grad_binary == 1 | (mag_binary==1 & dir_binary==1)] = 1

    # drawing new image with only the pixels that are within the threshold bounds
    combined_binary = np.zeros_like(img[:,:,0])
    combined_binary[color_binary == 1 | combined_grad_binary == 1] = 1

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

def sobelxy(gray, kernal_size):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernal_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernal_size)

    return sobelx, sobely

def gradient_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    sobelx = sobelxy(gray, sobel_kernel)
    abs = np.abs(sobelx)
    scaled_sobel_x = np.uint8(255*abs/np.max(abs))

    grad_binary = np.zeros_like(scaled_sobel_x[:,:,0])
    grad_binary[(scaled_sobel_x >= thresh[0]) & (scaled_sobel_x <= thresh[1])] = 1

    return grad_binary

def mag_thresh(gray, kernal=3, thresh=(0, 255)):
    sobelx, sobely = sobelxy(gray, kernal)
    mag = np.sqrt(sobelx**2 + sobely**2)

    scaled_mag = np.uint8(255*mag / np.max(mag))
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1

    return mag_binary

def dir_thresh(gray, kernal=3, thresh=(0, np.pi/2)):
    sobelx, sobely = sobelxy(gray, kernal)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

