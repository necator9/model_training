#!/usr/bin/env python3.7

# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Train a logistic regression classifier based on previously generated data (target objects + noises)

import sys
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

import config as cf
from libs import lib_transform_data as tdata

# Set up logging,
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(__name__ + '.log')
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

    logger.info('Input data shape: {}'.format(full_dataframe.shape))
    logger.info('Cases: angles {}, heights {}'.format(full_dataframe[cf.cam_a_k].unique(),
                                                      full_dataframe[cf.cam_y_k].unique()))

    return full_dataframe


def prepare_data_for_training(full_dataframe, features_cols):
    """
    Prepare data for model fitting: select important features from dataframe and merge them into numpy array
    :param full_dataframe: dataframe describing target and noises classes
    :param features_cols: features indices to take into account
    :return: features, labels
    """
    # All meaningful features
    x_tr = np.stack([full_dataframe[key] for key in features_cols], axis=1)
    y_tr = full_dataframe[cf.o_class_k]

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
    """
    Train a single classifier for all given camera angle/height scenarios
    """
    dt = read_dataframe(sys.argv[1], sys.argv[2])
    # Name of columns are used for training
    feature_vector = [cf.w_k, cf.h_k, cf.ca_k, cf.z_k, cf.cam_y_k, cf.cam_a_k]
    X_train, y_train, poly = prepare_data_for_training(dt, feature_vector)
    clf = train_cassifier(X_train, y_train)

    # Dump objects of classifier and polynomial transformer to files
    tdata.dump_object(sys.argv[3] + '_clf.pcl', clf)
    tdata.dump_object(sys.argv[3] + '_poly.pcl', poly)
