# -*- coding=utf-8 -*-
# py3


import cv2
import numpy as np
import time


def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Filter white pixels
    # white_threshold = 130  # 200  # 130
    # lower_white = np.array([white_threshold, white_threshold, white_threshold])
    # upper_white = np.array([255, 255, 255])
    # white_mask = cv2.inRange(image, lower_white, upper_white)
    # white_image = cv2.bitwise_and(image, image, mask=white_mask)

    lower_white = np.array([0, 0, 43], dtype=np.uint8)
    upper_white = np.array([180, 43, 255], dtype=np.uint8)
    white_mask = cv2.inRange(img_hsv, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # kernel = np.ones((3, 3), np.uint8)
    # white_image = cv2.morphologyEx(white_image, cv2.MORPH_ERODE, kernel)  # 开运算

    # Filter red pixels
    redLower01 = np.array([0, 43, 46], dtype=np.uint8)  # 部分红
    redUpper01 = np.array([10, 255, 255], dtype=np.uint8)
    redLower02 = np.array([156, 43, 46], dtype=np.uint8)  # 部分红
    redUpper02 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask01 = cv2.inRange(img_hsv, redLower01, redUpper01)
    red_mask02 = cv2.inRange(img_hsv, redLower02, redUpper02)
    red_mask = red_mask01 + red_mask02

    red_image = cv2.bitwise_and(image, image, mask=red_mask)

    # Combine the two above images
    image2 = cv2.addWeighted(white_image, 1., red_image, 1., 0.)
    cv2.imshow("asdfaa", image2)
    cv2.waitKey(500000)

    return image2


img_path = "image//41.png"  # 41 67
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

image = filter_colors(img)

# Read in and grayscale the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
