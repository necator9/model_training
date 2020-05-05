#!/usr/bin/env python3.7

# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Train a logistic regression classifier based on previously generated data (target objects + noises)

import sys
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

import config as sp


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


# Read the source data and filter it
target_df = pd.read_csv(sys.argv[1])
noises_df = pd.read_csv(sys.argv[2])
b_rec_k = ('x_px', 'y_px', 'w_px', 'h_px')
target_df = clean_by_margin(target_df, b_rec_k)
dt = pd.concat([noises_df, target_df])
logger.info('Data shape: {}'.format(dt.shape))
logger.info('Cases: angles {}, heights {}'.format(dt[sp.cam_a_k].unique(), dt[sp.cam_y_k].unique()))

# Prepare data for training
# All meaningful features
training_data = np.stack((dt[sp.w_k], dt[sp.h_k], dt[sp.ca_k], dt[sp.z_k], dt[sp.cam_y_k], dt[sp.cam_a_k]), axis=1)
features_cols = [0, 1, 2, 3, 4, 5]  # Features column idx to take into account
X_train = training_data[:, features_cols]
y_train = dt[sp.o_class_k]

poly = PolynomialFeatures(2, include_bias=True)  # Increase features polynomial order
X_train = poly.fit_transform(X_train)

# Init classifier
clf = LogisticRegression(solver='newton-cg', C=3, multi_class='auto', n_jobs=-1, max_iter=100, verbose=1)
logger.info('Starting the classifier training')
clf.fit(X_train, y_train)

# Dump objects of classifier and polynomial transformer to files
with open(sys.argv[3] + '_clf.pcl', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(sys.argv[3] + '_poly.pcl', 'wb') as handle:
    pickle.dump(poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
