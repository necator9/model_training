#!/usr/bin/env python3.7

import os
import sys
import multiprocessing
import queue
import signal
import logging

import numpy as np
import itertools

from libs import lib_feature_extractor as fe, lib_transform_data as tdata, lib_transform_2d as t2d, \
    lib_transform_3d as t3d
import config as sp

# Set up logging,
logger = logging.getLogger(__name__)
logger.setLevel(sp.loglevel)
file_handler = logging.FileHandler('generator.log')
ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s %(processName)s - %(message)s')
file_handler.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(file_handler)


def build_dims_mask(in_dim):
    """
    Fill by zeros not used dimensions
    :param in_dim:
    :return:
    """
    in_dim = np.asarray(in_dim)
    dim_mask = np.zeros((3, 3))
    dim_mask[in_dim[:, 0].astype(int), :] = in_dim[:, 1:]

    return dim_mask


def gen_dims_ranges(dim_mask):
    ranges_lst = [np.linspace(row[0], row[1], row[2].astype(int)) for row in dim_mask]
    ranges_lst = [np.zeros(1) if rg.size == 0 else rg for rg in ranges_lst]

    return ranges_lst


def parse_3d_obj_file(path):
    """
    Convert vertices to np.array([N, 4]) in homogeneous form.
    Faces are converted to np.array([M, 3]), therefore faces must be preliminary triangulated
    :param path: path to wavefront.obj file
    :return: np.array(vertices), np.array(faces)
    """
    def parse_string(string):
        spl = [el.split('//') for el in string.split()]
        res = [el[0] for i, el in enumerate(spl) if i != 0]
        return res

    with open(path, "r") as fi:
        lines = fi.readlines()

    vertices = np.array([parse_string(ln) for ln in lines if ln.startswith("v")], dtype='float')
    faces = [parse_string(ln) for ln in lines if ln.startswith("f")]
    faces = np.asarray([[int(el) for el in ln] for ln in faces]) - 1

    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # Bring to homogeneous form

    return vertices, faces


# Ignore keyboard interrupt in forks, let parent process to handle this gently
def init_child_process():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# Build an iterator object based on given parameters
def init_iterator(sc, work_obj):
    dim_mask = build_dims_mask(work_obj['dim']['val'])
    dim_ranges = gen_dims_ranges(dim_mask)
    rotate_y_rg = np.linspace(*work_obj['rotate_y'])
    cam_angles = np.arange(*sc['cam_angle'])
    x_range = np.arange(*sc['x_range'])
    y_range = np.arange(*sc['y_range'])
    z_range = np.arange(*sc['z_range'])
    thr_range = np.arange(*sc['thr_range'])

    it = itertools.product(cam_angles, dim_ranges[0], dim_ranges[1], dim_ranges[2], rotate_y_rg, x_range, y_range,
                           z_range, thr_range)

    total_iter = np.prod([rg.size for rg in (cam_angles, dim_ranges[0], dim_ranges[1], dim_ranges[2], rotate_y_rg,
                                             x_range, y_range, z_range, thr_range)])

    return it, total_iter


def get_status(i, total_iter):
    return '{:3.2f} %, {} / {}'.format(i / total_iter * 100, i, total_iter)


# Worker function
def generate_features(o_key, sc_key, save_q, stop_event):
    multiprocessing.current_process().name = o_key  # Set process name
    # Generate iterator from parameters in config
    work_obj = sp.obj_info[o_key]
    sc = sp.scene_info[sc_key]
    try:
        it, total_iter = init_iterator(sc, work_obj)
    except Exception:
        print(o_key, sc_key, sc['cam_angle'])
    logger.info("Total iterations: {}".format(total_iter))

    is_prop = work_obj['dim']['prop']
    ry_init = work_obj['ry_init']
    img_res = sc['img_res']
    sens_dim = sc['sens_dim']
    f_l = sc['f_l']
    o_class = work_obj['o_class']
    cx_cy = sc['cxcy']

    intrinsic = (np.asarray(img_res), f_l, np.asarray(sens_dim), cx_cy)
    vertices, faces = parse_3d_obj_file(os.path.join(sp.obj_dir_path, o_key))
    rw_system = t3d.Handler3D(vertices, operations=['s', 'ry', 't', 'rx'], k=intrinsic)

    data_to_save = list()
    entries_counter = 0
    prev_rx, prev_y = None, None
    i = 0
    for cam_a, ww, hh, dd, ry, x, y, z, thr in it:
        rw_system.transform((is_prop, np.asarray([ww, hh, dd])), np.deg2rad(ry_init + ry), np.asarray([x, y, z]),
                            np.deg2rad(cam_a))  # Transform object in 3D space and project to image plane

        mask = t2d.generate_image_plane(rw_system.img_points, faces, thr, img_res)

        c_ar, b_rect = t2d.find_basic_params(mask)

        if any(c_ar):
            # From multiple contours select one with maximal contour area
            max_idx = np.argmax(c_ar)
            c_ar, b_rect = np.expand_dims(c_ar[max_idx], axis=0), np.expand_dims(b_rect[max_idx], axis=0)
            # Update instance of feature extractor only when influencing parameters are changed
            if prev_rx != cam_a or prev_y != y:
                pc = fe.FeatureExtractor(cam_a, y, img_res, sens_dim, f_l, cx_cy)
                prev_rx, prev_y = cam_a, y
            # Extract features from contours and bounding rectangles
            z_est, x_est, width_est, height_est, rw_ca_est = pc.extract_features(c_ar, b_rect)

            # Get actual object shape from feature extractor instance
            shape_mtx = [mtx for mtx in rw_system.mtx_seq if isinstance(mtx, t3d.ScaleMtx)]
            ww, hh, dd, = shape_mtx[0].shape_cursor
            x_px, y_px, w_px, h_px = b_rect[0]
            # Form entry of generated parameters to save
            data_to_save.append([cam_a, y, z_est[0], z, x_est[0], x, width_est[0], ww, height_est[0], hh, rw_ca_est[0],
                                 o_key, o_class, ry, x_px, y_px, w_px, h_px, c_ar[0], thr, dd])
            entries_counter += 1

            if entries_counter % 10000 == 0:
                save_q.put(data_to_save)
                data_to_save = list()
                logger.info(get_status(i, total_iter))

        if stop_event.is_set():
            break

        if logger.getEffectiveLevel() == logging.DEBUG:
            t2d.plt_2d_projections(rw_system.transformed_vertices)
            t2d.plot_image_plane(mask, img_res)

        i += 1

    if len(data_to_save) > 0:
        save_q.put(data_to_save)

    logger.info('Finished {}'.format(get_status(i, total_iter)))


def saver_thr(out_f, q, stop_event):
    init_child_process()
    global header
    i = int()
    while True:
        try:
            data_to_save = q.get(timeout=1)
            header = tdata.write_to_csv(header, data_to_save, out_f)
            i += len(data_to_save)
        except queue.Empty:
            pass

        if stop_event.is_set() and q.empty():
            logger.info('{} entries has been written'.format(i))
            break


def del_if_exists(out_f):
    if os.path.exists(out_f):
        os.remove(out_f)
        logger.warning('Previous output file removed: {} '.format(out_f))
    else:
        logger.info(' Output file: {}'.format(out_f))


header = True
if __name__ == '__main__':
    m = multiprocessing.Manager()
    stop_event_gen = m.Event()
    stop_event_saver = m.Event()
    data_save_q = m.Queue()

    out_file = sys.argv[1] if len(sys.argv) == 2 else 'default_filename.csv'
    del_if_exists(out_file)
    saver = multiprocessing.Process(target=saver_thr, args=(out_file, data_save_q, stop_event_saver), name='saver')
    o_keys = sp.obj_info.keys()
    scene_key = sp.scene_info.keys()

    if logger.getEffectiveLevel() == logging.DEBUG:
        o_keys = [key for key in o_keys if not key.startswith('_') and key.startswith('test')]
        scene_key = [key for key in scene_key if not key.startswith('_') and key.startswith('test')]
    else:
        o_keys = [key for key in o_keys if not key.startswith('_') and not key.startswith('test')]
        scene_key = [key for key in scene_key if not key.startswith('_') and not key.startswith('test')]

    iterable_args = itertools.product(o_keys, scene_key, [data_save_q], [stop_event_gen])
    pool = multiprocessing.Pool(processes=len(o_keys), initializer=init_child_process)

    try:
        saver.start()
        result = pool.starmap(generate_features, iterable_args)

    except KeyboardInterrupt:
        stop_event_gen.set()

    finally:
        pool.close()
        pool.join()
        stop_event_saver.set()
        saver.join()
