#!/usr/bin/env python3.7

# Created by Ivan Matveev at 05.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Script to generate noises class' features to model training
# Noises are generated to surround target features

from scipy.spatial import qhull
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import itertools
import numpy as np
import pandas as pd

import logging
import argparse

from feat_gen import get_status
import map as cfg

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
    hull = ConvexHull(cnt_data)  # Get indices of external vertices
    hull = cnt_data[hull.vertices, :]  # Extract points by their indices
    hull = Delaunay(hull)  # The Delaunay class has a method to check if a point belongs to a hull

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


def gen_noises(iterator, features, classes, n_points):
    out_data_temp = list()
    it = 0
    for angle, height in iterator:
        try:
            # Select one slice with particular angle and height
            a_h_data = features[(features[cfg.cam_a_k] == angle) & (features[cfg.cam_y_k] == height)]
            # Find border values for ranges of important basic features
            w_rg = find_rg((a_h_data[cfg.w_k].min(), a_h_data[cfg.w_k].max()))
            h_rg = find_rg((a_h_data[cfg.h_k].min(), a_h_data[cfg.h_k].max()))
            hulls = []
            # Iterate through the object classes
            for i in classes:
                try:
                    # Select and transform to required form
                    df = a_h_data[a_h_data[cfg.o_class_k] == i]
                    df = np.vstack([df[cfg.w_k], df[cfg.h_k]]).T
                    hulls.append(get_hull(df))  # Generate 2-dim hull for each class

                # Skip cases when the surface is flat (values one of some column are not varying)
                except qhull.QhullError:
                    logger.warning('Skipping the case. Angle {}, height {}, class {} '.format(angle, height, i))
                    continue

            if len(hulls) == 0:
                continue

            # Generate features of width and height for a class of noise
            w_h = gen_w_h(hulls, n_points, w_rg, h_rg)
            # Find available distance range
            d_rg = find_rg((a_h_data[cfg.z_k].min(), a_h_data[cfg.z_k].max()), margin=0.5)
            # d = np.expand_dims(np.array([round(i) for i in np.random.uniform(*d_rg, size=[points_amount, 1])]), axis=1)
            d = np.random.uniform(*d_rg, size=[n_points, 1])  # Fill the distance range uniformly

            # Generate contour area for a class of noise, parameters are chosen empirically
            # mu, sigma = 0.5, 0.1
            # ca = np.random.normal(mu, sigma, size=[POINTS_AMOUNT, 1]) * np.expand_dims(w_h[:, 0], axis=1) * \
            #      np.expand_dims(w_h[:, 1], axis=1)
            ca = np.expand_dims(np.random.uniform(0, w_h[:, 0].max() * w_h[:, 1].max(), n_points), axis=1)

            res = np.hstack((w_h, ca, d, np.ones((n_points, 1)) * height, np.ones((n_points, 1)) * angle))
            out_data_temp.extend(res.tolist())

            it += 1
            logger.info(get_status(it, total_iterations))

        except ValueError:
            logger.warning("No such angle {} or height {}".format(angle, height))

    return out_data_temp


parser = argparse.ArgumentParser(description='Generate noises around features')
parser.add_argument('features', action='store', help="path to the features csv file")
parser.add_argument('-n', '--noises', action='store', help="path to the output csv file containing noises features"
                                                           " (default:noises.csv )", default='noises.csv')
parser.add_argument('-p', '--points', action='store', type=int, help="amount of points per hull (default: 40000)",
                    default=40000)
args = parser.parse_args()

obj_features = pd.read_csv(args.features)  # Read generated features
# Find available camera angles and heights
angle_rg = obj_features[cfg.cam_a_k].unique()
height_rg = obj_features[cfg.cam_y_k].unique()
it_params = itertools.product(angle_rg, height_rg)

o_classes = obj_features[cfg.o_class_k].unique()

total_iterations = len(angle_rg) * len(height_rg)
logger.info('Total iterations: {}'.format(total_iterations))
logger.info('Camera height range: {}\nCamera angle range: {}'.format(height_rg, angle_rg))
noise = gen_noises(it_params, obj_features, o_classes, args.points)
noise = pd.DataFrame(noise, columns=[cfg.w_k, cfg.h_k, cfg.ca_k, cfg.z_k, cfg.cam_y_k, cfg.cam_a_k])
noise[cfg.o_class_k] = 0

noise = noise.round({cfg.z_k: 2, cfg.ca_k: 3, cfg.w_k: 2, cfg.h_k: 2, cfg.cam_y_k: 2, cfg.cam_a_k: 1, cfg.o_class_k: 0})
noise.to_csv(args.noises, index=False)
