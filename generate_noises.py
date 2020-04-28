from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import itertools
import numpy as np
import pandas as pd

import logging
import sys
import os

from generate_features import get_status

# Set up logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_hull(cnt_data):
    """
    Find convex hull for a given set of points (features for a selected object class).
    :param cnt_data: n-dim array of features
    :return: convex hull wrapping the features
    """
    hull = ConvexHull(cnt_data)
    hull = cnt_data[hull.vertices, :]
    hull = Delaunay(hull)

    return hull


def in_hull(p, hull):
    """
    Check if a point belongs to a convex.
    :param p: n-dim point
    :param hull: n-dim convex hull
    :return: boolean affiliation
    """
    return hull.find_simplex(p) >= 0


def find_rg(rg, margin=1.5):
    """
    Find range for noise generation: values of (passed range + margin) > 0.
    :param rg: input range (f_min, f_fax) corresponding to selected feature
    :param margin: extra space to be added into range
    :return: passed range + margin
    """
    # Change to a positive value if a range border <= 0
    def check_val(val):
        return 0.01 if val <= 0 else val

    res_rg = check_val(min(rg) - margin), max(rg) + margin

    return res_rg


def gen_w_h(hulls_, points_amount_, w_rg_, h_rg_):
    """
    Generate such w and h features which do not belong to any convex hull among given.
    :param hulls_: list of hulls corresponding to different classes but the same scene scenario
    :param points_amount_: points to generate
    :param w_rg_: working range for width
    :param h_rg_: working range for height
    :return: list of [[w, h],...] features
    """
    noises = []  # Resulting output list
    while True:
        point = [np.random.uniform(*w_rg_), np.random.uniform(*h_rg_)]  # Suppose a point within declared ranges
        res_ = [in_hull(point, hull) for hull in hulls_]  # Check if point is in hulls

        # If point does not belong to any hulls, add to resulting list. Continue till enough points collected
        if not (any(res_)):
            noises.append(point)
            if len(noises) >= points_amount_:
                return np.array(noises)


POINTS_AMOUNT = 3000

# Mapping of keys in csv file
cam_a_k = 'cam_a'         # Camera angle relative to the ground surface in range [0, -90] deg.
# 0 deg. - the camera is parallel to the ground surface; -90 deg. - camera points perpendicularly down
cam_y_k = 'y'             # Ground surface offset (negative camera height) relative to camera origin in range [-3, -n] m
w_k = 'width_est'         # Feature - estimated object width
h_k = 'height_est'        # Feature - estimated object height
ca_k = 'rw_ca_est'        # Feature - estimated object contour area
z_k = 'z_est'             # Feature - estimated object distance from a camera
o_class_k = 'o_class'     # Object class as an integer, where 0 is a noise class

csv_file = sys.argv[1]  # Path to targeted csv file passed as cl argument
obj_features = pd.read_csv(csv_file)  # Read generated features
# Find available camera angles and heights
angle_rg = obj_features[cam_a_k].unique()
height_rg = obj_features[cam_y_k].unique()
it_params = itertools.product(angle_rg, height_rg)

o_classes = obj_features[o_class_k].unique()

total_iterations = len(angle_rg) * len(height_rg)
logger.info('Total iterations: {}'.format(total_iterations))
logger.info('Camera height range: {}\nCamera angle range: {}'.format(height_rg, angle_rg))

out_data_temp = list()

it = 0
for angle, height in it_params:
    try:
        # Select one slice with particular angle and height
        a_h_data = obj_features[(obj_features[cam_a_k] == angle) & (obj_features[cam_y_k] == height)]
        # Find border values for ranges of important basic features
        w_rg = find_rg((a_h_data[w_k].min(), a_h_data[w_k].max()))
        h_rg = find_rg((a_h_data[h_k].min(), a_h_data[h_k].max()))

        hulls = []
        # Iterate through the object classes
        for i in o_classes:
            try:
                # Select and transform to required form
                df = a_h_data[a_h_data[o_class_k] == i]
                df = np.vstack([df[w_k], df[h_k]]).T

                hulls.append(get_hull(df))  # Generate 2-dim hull for each class

            except Exception:  # Lol
                continue

        if len(hulls) == 0:
            continue

        # Generate features of width and height for a class of noise
        w_h = gen_w_h(hulls, POINTS_AMOUNT, w_rg, h_rg)
        # Find available distance range
        d_rg = find_rg((a_h_data[z_k].min(), a_h_data[z_k].max()), margin=0.5)
        # d = np.expand_dims(np.array([round(i) for i in np.random.uniform(*d_rg, size=[points_amount, 1])]), axis=1)
        d = np.random.uniform(*d_rg, size=[POINTS_AMOUNT, 1])  # Fill the distance range uniformly

        # Generate contour area for a class of noise, parameters are chosen empirically
        mu, sigma = 0.5, 0.1
        ca = np.random.normal(mu, sigma, size=[POINTS_AMOUNT, 1]) * np.expand_dims(w_h[:, 0], axis=1) * \
             np.expand_dims(w_h[:, 1], axis=1)

        res = np.hstack((w_h, ca, d, np.ones((POINTS_AMOUNT, 1)) * height, np.ones((POINTS_AMOUNT, 1)) * angle))

        out_data_temp.extend(res.tolist())

        it += 1
        logger.info(get_status(it, total_iterations))

    except ValueError:
        logger.warning("No such angle {} or height {}".format(angle, height))

dir_path = os.path.split(csv_file)[0]  # Extract dir path form input path
in_file_name = os.path.split(csv_file)[1]  # Extract filename form input path
out_file_name = os.path.join(dir_path, 'n_{}'.format(in_file_name))

noise = pd.DataFrame(out_data_temp, columns=[w_k, h_k, ca_k, z_k, cam_y_k, cam_a_k])
noise[o_class_k] = 0

noise = noise.round({z_k: 2, ca_k: 3, w_k: 2, h_k: 2, cam_y_k: 2, cam_a_k: 1, o_class_k: 0})
noise.to_csv(out_file_name, index=False)
