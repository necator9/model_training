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
import lib_transform_data as tdata


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


def read_dataframe(target_df_path, noises_df_path):
    """
    Read the source training data from files and filter it
    :param target_df_path: path to csv file containing objects' features
    :param noises_df_path: path to csv file containing noises' features
    :return: filtered and merged dataframe
    """
    target_df = pd.read_csv(target_df_path)
    noises_df = pd.read_csv(noises_df_path)
    target_df = tdata.clean_by_margin(target_df)
    full_dataframe = pd.concat([noises_df, target_df])

    logger.info('Input data shape: {}'.format(dt.shape))
    logger.info('Cases: angles {}, heights {}'.format(dt[sp.cam_a_k].unique(), dt[sp.cam_y_k].unique()))

    return full_dataframe


def prepare_data_for_training(full_dataframe, features_cols=(0, 1, 2, 3, 4, 5)):
    """
    Prepare data for model fitting: select important features from dataframe and merge them into numpy array
    :param full_dataframe: dataframe describing target and noises classes
    :param features_cols: features indices to take into account
    :return: features, labels
    """
    # All meaningful features
    training_data = np.stack((full_dataframe[sp.w_k], full_dataframe[sp.h_k], full_dataframe[sp.ca_k],
                              full_dataframe[sp.z_k], full_dataframe[sp.cam_y_k], full_dataframe[sp.cam_a_k]), axis=1)
    x_tr = training_data[:, features_cols]
    y_tr = full_dataframe[sp.o_class_k]

    poly_scale = PolynomialFeatures(2, include_bias=True)  # Increase features polynomial order
    x_tr = poly_scale.fit_transform(x_tr)

    return x_tr, y_tr, poly_scale


def train_cassifier(x_tr, y_tr):
    """
    Fit the classifier model using given features and labels
    :param x_tr: features
    :param y_tr: labels
    :return: instance of the trained model
    """
    # Init classifier
    log_reg = LogisticRegression(solver='newton-cg', C=3, multi_class='auto', n_jobs=-1, max_iter=100, verbose=1)
    logger.info('Starting the classifier training')
    log_reg.fit(x_tr, y_tr)

    return log_reg


if __name__ == '__main__':
    dt = read_dataframe(sys.argv[1], sys.argv[2])
    X_train, y_train, poly = prepare_data_for_training(dt)
    clf = train_cassifier(X_train, y_train)

    # Dump objects of classifier and polynomial transformer to files
    with open(sys.argv[3] + '_clf.pcl', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sys.argv[3] + '_poly.pcl', 'wb') as handle:
        pickle.dump(poly, handle, protocol=pickle.HIGHEST_PROTOCOL)
