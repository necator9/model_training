# Created by Ivan Matveev at 01.07.20
# E-mail: ivan.matveev@hs-anhalt.de

# Find the camera matrix which can provide sufficient AOVs based on OpenCV calibration matrix

import cv2
import matplotlib.pyplot as plt
import numpy as np

cam_param = {'rpi': {'mtx': np.array([[602.17434328, 0., 511.32476428],   # Optical center was corrected manually
                                      [0., 601.27444228, 334.8572872],
                                      [0., 0., 1.]]),
                     'base_res': (1024, 768),
                     'dist': np.array([[-0.321267, 0.11775163, 0.00091285, 0.0007689, -0.02101163]]),
                     'image': 'scene_images/lamp_pole_1_2.png'},

             'hd_3000': {'mtx':  np.array([[693.38863768, 0., 339.53274061],
                                          [0., 690.71040995, 236.18033069],
                                          [0., 0., 1.]]),
                         'base_res': (640, 480),
                         'dist': np.array([[0.21584076, -1.58033256, -0.00369491,  0.00366677,  2.94284061]])}}


camera = cam_param['rpi']

intrinsic = camera['mtx']
dist = camera['dist']
image_path = camera['image']

image = cv2.imread(image_path, 0)

h, w = image.shape[: 2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, dist, (w, h), 0.5, (w, h))
dst = cv2.undistort(image, intrinsic, dist, None, newcameramtx)

plt.imshow(dst, cmap='gray', vmin=0, vmax=255)
plt.show()

cv2.imwrite('scene_images/lamp_pole_opt_mtx_2.png', dst)
print(newcameramtx)
