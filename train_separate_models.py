# Created by Ivan Matveev at 06.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Train multiple classifiers' models, each for particular camera height and angle scenario

import sys
import itertools
import logging

import train_model as tm
import lib_transform_data as tdata
import config as cf
from joblib import Parallel, delayed, cpu_count

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


def select_slice(dataframe, keys_vals):
    """
    Filter input dataframe by given keys and values
    :param dataframe: Not filtered dataframe
    :param keys_vals: dictionary of columns' names corresponding to target values
    :return: filtered dataframe
    """
    for key, val in keys_vals.items():
        dataframe = dataframe[dataframe[key] == val]

    if dataframe.shape[0] < 100:
        logger.warning("Amount of rows in dataframe is not sufficient. "
                       "Scene: {}\nSkipping the scene".format([(key, val) for key, val in keys_vals.items()]))
        return None

    return dataframe


def train_single_clf(height, angle, filtered_df):
    """
    Worker function to train classifiers in parallel
    :param height: considering camera height
    :param angle: considering camera angle
    :param filtered_df: dataframe corresponding filtered by height and width
    :return: height, angle to be used as keys and classifier
    """
    # Camera angle and height are not taken into account since they are dictionary keys for particular classifier
    feature_vector = [cf.w_k, cf.h_k, cf.ca_k, cf.z_k]  # Name of columns are used for training
    x_train, y_train, poly = tm.prepare_data_for_training(filtered_df, feature_vector)
    clf = tm.train_cassifier(x_train, y_train)

    return height, angle, clf


def build_dictionary(it_params):
    """
    Represent result of training as nested dictionary
    :param it_params: each row is a one training case of height and angle: [[height_1, angle_1, clf_obj_1], ...]
    :return: dictionary of a following form: {height_1: {angle_1: clf_obj_1, ...}, ...}
    """
    data = dict()
    poly = None
    for height, angle, clf in it_params:
        temp_dict = {angle: clf}
        try:
            data[height].update(temp_dict)
        except KeyError:
            data[height] = temp_dict   # Init dictionary for a selected height if not exist
    data['poly'] = poly  # Add single poly object since they are all the same

    return data


dt = tm.read_dataframe(sys.argv[1], sys.argv[2])

angles = dt[cf.cam_a_k].unique()
heights = dt[cf.cam_y_k].unique()
h_a_it = itertools.product(heights, angles)

# Prepared training data for training in parallel: split it by height and angles cases
iterate = [[height, angle, select_slice(dt, {cf.cam_y_k: height, cf.cam_a_k: angle})] for height, angle in h_a_it]
# Drop cases with insufficient data filling, which are marked as None
iterate = [[height, angle, df] for height, angle, df in iterate if df is None]
# Run jobs in parallel
result = Parallel(n_jobs=cpu_count())(delayed(train_single_clf)(height, angle, dataframe)
                                      for height, angle, dataframe in iterate)

result_dict = build_dictionary(result)
# Dump dictionary with multiple training cases
tdata.dump_object(sys.argv[3] + 'separate_clf_dict.pcl', result_dict)
