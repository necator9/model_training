#!/usr/bin/env python3

# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Train a logistic regression classifier based on previously generated data (target objects + noises)


import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import map as cf
from libs import lib_transform_data as tdata
from libs import lib_logging as log

logger = log.spawn_logger(f'{__file__}.log')


def read_dataframe(target_df_path, noises_df_path):
    """
    Read the source training data from files and filter it
    :param target_df_path: path to csv file containing objects' features
    :param noises_df_path: path to csv file containing noises' features
    :return: filtered and merged dataframe
    """
    target_df = pd.read_csv(target_df_path)
    noises_df = pd.read_csv(noises_df_path)
    full_dataframe = pd.concat([noises_df, target_df])
    logger.info(f'Input data shape: {full_dataframe.shape}')
    logger.info(f'Cases: angles {full_dataframe[cf.cam_a_k].unique()}, heights { full_dataframe[cf.cam_y_k].unique()}')

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


def train_classifier(x_tr, y_tr):
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


def estimate_clf(clf, X_test, y):
    # Evaluate accuracy
    def gen_report(y, y_pred):
        report = 'Precision P=TP/TP+FP\nRecall R=TP/TP+FN\nF1 score F1=2*(P*R)/(P+R)\n'
        report += f'Accuracy {accuracy_score(y, y_pred)}\n'
        report += f'{classification_report(y, y_pred)}\n'
        report += f'{confusion_matrix(y, y_pred)}\n'

        return report

    # Classify the dataset
    y_pred = clf.predict(X_test)
    # y_prob = clf.predict_proba(X_test)
    return gen_report(y, y_pred)


if __name__ == '__main__':
    """
    Train a single classifier for all given camera angle/height scenarios
    """
    dt = read_dataframe(sys.argv[1], sys.argv[2])
    # Name of columns are used for training
    feature_vector = [cf.w_est_k, cf.h_est_k, cf.ca_est_k, cf.z_est_k, cf.cam_y_k, cf.cam_a_k]
    X_train, y_train, poly = prepare_data_for_training(dt, feature_vector)
    clf = train_classifier(X_train, y_train)

    # Dump objects of classifier and polynomial transformer to files
    tdata.dump_object(sys.argv[3] + '_clf.pcl', clf)
    tdata.dump_object(sys.argv[3] + '_poly.pcl', poly)
