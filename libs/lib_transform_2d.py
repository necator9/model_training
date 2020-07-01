# Created by Ivan Matveev at 05.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Functions to manipulate 2d image plane including plotting

import cv2
import numpy as np
from matplotlib import pyplot as plt


def plt_2d_projections(vertices):
    fig, ax_lst = plt.subplots(2, 2, squeeze=True, figsize=(8.0, 13.0))
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ax_to_plot = [[[x, y], [z, y]],
                  [[x, z]]]
    fig.delaxes(ax_lst[1][1])

    color = 'k'
    s = 0.1
    marg = 0.1

    for (m, n), subplot in np.ndenumerate(ax_lst):
        if m == 1 and n == 1:
            break
        subplot.scatter(*ax_to_plot[m][n], color=color, s=s)
        subplot.set_aspect('equal', adjustable='box')
        min_x = min(ax_to_plot[m][n][0])
        max_x = max(ax_to_plot[m][n][0])
        subplot.set_xlim(min_x - marg, max_x + marg)
        subplot.set_ylim(min(ax_to_plot[m][n][1]) - marg, max(ax_to_plot[m][n][1]) + marg)
        subplot.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_image_plane(mask, img_res):
    plt.imshow(mask, cmap='gray')
    plt.xlim(0, img_res[0]), plt.ylim(img_res[1], 0)
    plt.show()


def generate_image_plane(img_points, faces, k_size, img_res):
    mask = np.zeros((img_res[1], img_res[0]), np.uint8)
    poly = np.take(img_points, faces, axis=0)
    for p in poly:
        cv2.fillConvexPoly(mask, p, color=255)
    # cv2.fillPoly(mask, pts=poly, color=255) # Works faster but not as desired
    cv2.dilate(mask, cv2.getStructuringElement(0, (int(k_size), int(k_size))), dst=mask)

    return mask


def find_basic_params(mask):
    cnts, _ = cv2.findContours(mask, mode=0, method=1)
    c_areas = [cv2.contourArea(cnt) for cnt in cnts]
    b_rects = [cv2.boundingRect(b_r) for b_r in cnts]

    return np.asarray(c_areas), np.asarray(b_rects)
