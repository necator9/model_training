#!/usr/bin/env python3.7

# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Train a logistic regression classifier based on previously generated data

import sys
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Set up logging,
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('generator.log')
ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(file_handler)


if len(sys.argv) != 4:
    print('\n\nUsage: ./script [path_to_features.csv] [path_to_noises.csv] [classifier_path_name]\n'
          'All arguments are obligatory.\n')
    sys.exit()


def clean_by_margin(df_data_or, b_rec_k, margin=1, img_res=(1280, 720)):
    """
    # Remove objects which have intersections with frame borders
    :param df_data_or: Input dataframe to filter
    :param b_rec_k: Parameters of a bounding rectangle on image plane
    :param margin: Offset from horizontal and vertical frame borders
    :param img_res: Working image resolution
    :return: filtered dataframe
    """
    x_px, y_px, w_px, h_px = b_rec_k
    df_data_p = df_data_or[(df_data_or[x_px] > margin) & (df_data_or[x_px] + df_data_or[w_px] < img_res[0] - margin) &
                           (df_data_or[y_px] > margin) & (df_data_or[y_px] + df_data_or[h_px] < img_res[1] - margin)]
    return df_data_p


# Mapping the keys in csv file
cam_a_k = 'cam_a'         # Camera angle relative to the ground surface in range [0, -90] deg.
# 0 deg. - the camera is parallel to the ground surface; -90 deg. - camera points perpendicularly down
cam_y_k = 'y'             # Ground surface offset (negative camera height) relative to camera origin in range [-3, -n] m
w_k = 'width_est'         # Feature - estimated object width
h_k = 'height_est'        # Feature - estimated object height
ca_k = 'rw_ca_est'        # Feature - estimated object contour area
z_k = 'z_est'             # Feature - estimated object distance from a camera
o_class_k = 'o_class'     # Object class as an integer, where 0 is a noise class
o_name_k = 'o_name'       # Object name as a string

# Read the source data and filter it
target_df = pd.read_csv(sys.argv[1])
noises_df = pd.read_csv(sys.argv[2])
b_rec_k = ('x_px', 'y_px', 'w_px', 'h_px')
target_df = clean_by_margin(target_df, b_rec_k)
dt = pd.concat([noises_df, target_df])
logger.info('Data shape: {}'.format(dt.shape))
logger.info('Cases: {}, {}'.format(dt[cam_a_k].unique(), dt[cam_y_k].unique()))

# Prepare data for training
# All meaningful features
training_data = np.stack((dt[w_k], dt[h_k], dt[ca_k], dt[z_k], dt[cam_y_k], dt[cam_a_k]), axis=1)
X_train, X_test, y_train, y_test = train_test_split(training_data, dt[o_class_k], test_size=.3, random_state=42)
features_cols = [0, 1, 2, 3, 4, 5]  # Features column idx to take into account
X_train = X_train[:, features_cols]

poly = PolynomialFeatures(2, include_bias=True)  # Increase features polynomial order
X_train = poly.fit_transform(X_train)

clf = LogisticRegression(solver='newton-cg', C=3, multi_class='auto', n_jobs=-1, max_iter=100)  # Init classifier
logger.info('Starting the classifier training')
clf.fit(X_train, y_train)

# Dump objects of classifier and polynomial transformer to files
with open(sys.argv[3] + '_clf.pcl', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(sys.argv[3] + '_poly.pcl', 'wb') as handle:
    pickle.dump(poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
