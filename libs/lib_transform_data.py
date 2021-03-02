# Created by Ivan Matveev at 05.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Functions for data manipulations: filtration, writing to file

import pickle
import pandas as pd

import map as cf  # Columns names are used from the config file


def write_to_csv(header_, data_, out_file):
    """
    Round values and write to csv file. Function can be called multiple times in cycle appending data
    :param header_: Boolean status of writing columns' names (only first time is true)
    :param data_: dataframe to write
    :param out_file: path with name to csv file
    :return: header status. Once header have been written, the status changed to False
    """
    x_px_k, y_px_k, w_px_k, h_px_k = cf.b_rec_k
    df = pd.DataFrame(data_, columns=[cf.cam_a_k, cf.cam_y_k, cf.z_k, 'z', 'x_est', 'x', cf.w_k, 'ww', cf.h_k, 'hh',
                                      cf.ca_k, cf.o_name_k, cf.o_class_k, 'ry', x_px_k, y_px_k, w_px_k, h_px_k,
                                      'c_ar_px', 'thr', 'dd'])
    df = df.round({cf.z_k: 2, 'x_est': 2, cf.ca_k: 3, cf.w_k: 2, cf.h_k: 2, 'x': 2, cf.cam_y_k: 2, 'z': 2,
                  cf.cam_a_k: 1, 'ry': 1, 'ww': 2, 'hh': 2, 'dd': 2, cf.o_class_k: 0})
    with open(out_file, 'a') as f:
        df.to_csv(f, header=header_, index=False)

    return False


def dump_object(path, obj):
    """
    Dump object by serialization to file
    :param path: path to file
    :param obj: object to save
    """
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=4)


def clean_by_margin(df_data_or, margin=1, img_res=(1280, 720)):
    """
    # Remove objects which have intersections with frame borders
    :param df_data_or: Input dataframe to filter
    :param margin: Offset from horizontal and vertical frame borders
    :param img_res: Working image resolution
    :return: filtered dataframe
    """
    x_px_k, y_px_k, w_px_k, h_px_k = cf.b_rec_k
    df_data_p = df_data_or[(df_data_or[x_px_k] > margin) &
                           (df_data_or[x_px_k] + df_data_or[w_px_k] < img_res[0] - margin) &
                           (df_data_or[y_px_k] > margin) &
                           (df_data_or[y_px_k] + df_data_or[h_px_k] < img_res[1] - margin)]
    return df_data_p


def is_crossing_margin(f_margins, basic_params):
    margin_filter_mask = ((basic_params[:, 1] > f_margins['left']) &  # Built filtering mask
                          (basic_params[:, 5] < f_margins['right']) &
                          (basic_params[:, 2] > f_margins['up']) &
                          (basic_params[:, 6] < f_margins['bottom']))

    crossing = True if False in margin_filter_mask else False
    return crossing
