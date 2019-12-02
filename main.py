import glob

import numpy as np
import cv2

import config as cfg


# Initialization of parameters
board_size_x = cfg.board_size_x
board_size_y = cfg.board_size_y

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_size_y*board_size_x, 3), np.float32)
objp[:,:2] = np.mgrid[0:board_size_x, 0:board_size_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(cfg.img_location)

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, cfg.img_resize)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (board_size_x,board_size_y),None)
    print(ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.imshow('Original', img)
        img = cv2.drawChessboardCorners(img, (board_size_x, board_size_y), corners2, ret)
        cv2.imshow('Find Corner', img)
        cv2.waitKey(cfg.imshow_time)

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Undistortion
for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, cfg.img_resize)
    h, w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    # print(dst)
    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(cfg.imshow_time)
    # cv2.imwrite('calibresult.png',dst)