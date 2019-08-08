# -*- coding: utf-8 -*-
"""
Use two cameras to get the 3D distance of points

For point1, get its u, v (pixel position) in left and right images independently
then write it as point1 = [ul, vl, ur, vr]
point2, point3, ..., pointn are similar.

Input: points position in two images (from left and right cameras)
Output: 3D distance between points
"""

import numpy as np
import cv2

# checkerboard corner point position (ul, vl, ur, vr)
# l for group_5_19_left.jpg, r for group_5_19_right.jpg
# The 3D distance of the following two points are 20mm (ground truth)
points = [
[598.563,	436.873,	525.596,	380.127], 
[631.825,	427.795,	557.187,	383.776],
]

# intrinsic matrix of the left and right cameras (got from around 20 images)
intrinsic_left = np.array([[948.209,0,617.72],[0,960.912,374.665],[0,0,1]])
intrinsic_right = np.array([[960.613,0,653.568],[0,968.804,369.677],[0,0,1]])

distortion_left = np.array([0.0487857,-0.279364,0.00329971,-0.0073942,0.367614])
distortion_right = np.array([0.0400987,-0.218146,0.00793717,0.00552612,0.133228])

# rotation and translation matrix of camera1&2 related to checkerboard (group_5_18)
Rl = cv2.Rodrigues(np.array([-0.78103995,-0.30358092,-0.05871129]))[0]
Tl = [[2.9147771],[0.09855708],[26.59388227]]

Rr = cv2.Rodrigues(np.array([-0.83922768, 0.19875148, 0.47898573]))[0]
Tr = [[0.261736697],[0.0165916331],[29.8146047]]

def point2d_3d(ul, vl, ur, vr):
    """ 2D position in images to real world 3D position
    A function transfer the coordinate of a point 
    in left and right images (1280x720) to real world coordinate
    Input: the position of a point in left and right images (pixel)
    Output: 3D position of the point in real world (mm), 
            according to the left camera position
    """

    # Zc_left * [[ul],[vl],[1]] = Pl * [[X],[Y],[Z],[1]]
    Pl = np.dot(intrinsic_left, np.hstack((Rl, Tl)))
    Pr = np.dot(intrinsic_right, np.hstack((Rr, Tr)))

    # solve AX = B
    A_eq = [[ul*Pl[2][0]-Pl[0][0], ul*Pl[2][1]-Pl[0][1], ul*Pl[2][2]-Pl[0][2]],\
        [vl*Pl[2][0]-Pl[1][0], vl*Pl[2][1]-Pl[1][1], vl*Pl[2][2]-Pl[1][2]],\
        [ur*Pr[2][0]-Pr[0][0], ur*Pr[2][1]-Pr[0][1], ur*Pr[2][2]-Pr[0][2]],\
        [vr*Pr[2][0]-Pr[1][0], vr*Pr[2][1]-Pr[1][1], vr*Pr[2][2]-Pr[1][2]]] 
    B_eq = [Pl[0][3]-ul*Pl[2][3], Pl[1][3]-vl*Pl[2][3], Pr[0][3]-ur*Pr[2][3], Pr[1][3]-vr*Pr[2][3]]

    answer = np.linalg.lstsq(A_eq, B_eq, rcond=-1)
    X = 20*answer[0][0]
    Y = 20*answer[0][1]
    Z = 20*answer[0][2]
#     print(X,Y,Z,end='\n')
#     print(np.dot(A_eq, [[X],[Y],[Z]]))

    return X, Y, Z

def undistort_image(img, intrinsic, distortion):
    h, w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(intrinsic,distortion,(w,h),0,(w,h))
    dst = cv2.undistort(img, intrinsic, distortion, None, newcameramtx)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def distance_3D(point0_2d, point1_2d):
    X0, Y0, Z0 = point2d_3d(point0_2d[0],point0_2d[1],point0_2d[2],point0_2d[3])
    X1, Y1, Z1 = point2d_3d(point1_2d[0],point1_2d[1],point1_2d[2],point1_2d[3])

    distance = np.sqrt((X0 - X1)*(X0 - X1) + (Y0 - Y1)*(Y0 - Y1) + (Z0 - Z1)*(Z0 - Z1))
    print(distance) 

def main():
    print("Distance in 3D (mm)") 
    for point_2d in points[1:]:
        distance_3D(points[0], point_2d)


if __name__ == '__main__':
    main()
    