# camera-calibration
This repo aims to get the 3D distance of two points in the real world using two cameras.

1. (one_cam_calibration.py) Get the intrinsic matrix and distortion matrix of the two cameras separately.
2. (pose_estimation.py) Get the extrinsic matrix (R and T) of the two cameras according to one checkerboard.
3. (point2d_3d.py) Calculate the 3D distance when given the 2D position of the two points in two images.

References:
https://github.com/njanirudh/Aruco_Tracker
OpenCV-Python Tutorials » Camera Calibration and 3D Reconstruction
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
https://opencv.org/license/
