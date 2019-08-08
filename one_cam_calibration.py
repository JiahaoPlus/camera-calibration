# -*- coding: utf-8 -*-
"""
Single camera calibration

Input: all checkerboard images from one camera
Output:    mtx: intrinsic matrix of this camera (3x3)
           dist: distortion matrix of this camera (1x5) 
           
Reference: OpenCV-Python Tutorials » Camera Calibration and 3D Reconstruction
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://opencv.org/license/
"""

import numpy as np
import cv2
import glob

IMAGES_PATH = 'image/*left.jpg'
PARA_SAVE_PATH = "parameters/parameters_left.yaml"

# Input all checkerboard images from one camera (prefer 10-20 images)
images = glob.glob(IMAGES_PATH)

# termination criteria (each small square in our checkerboard is 20mm x 20mm)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# our checkerboard has 7*9 corner points (see image folder)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
correct_idx = []

print("Manually check whether the corners are correct")
print("push 'a' for correct, push other key or wait 5 second for wrong")

for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,7), corners2,ret)
        cv2.putText(img, str(idx), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv2.imshow('img',img)
        key = cv2.waitKey(5000)
        if key & 0xFF == ord('a'):
            correct_idx.append(idx)
            objpoints.append(objp)
            imgpoints.append(corners2)
            continue
        else:
            continue

print(correct_idx)
# Only use the images with "correct" corners for calibration 
# In example images, for left, group_5_22,23,27_left are wrong; 
# for right, group_5_21,22,23,27_right are wrong
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# ---------- Saving the calibration -----------------
cv_file = cv2.FileStorage(PARA_SAVE_PATH, cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)
# note you *release* you don't close() a FileStorage object
cv_file.release()
