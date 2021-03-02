#!/usr/bin/env python3

# Created by Ivan Matveev at 01.03.20
# E-mail: ivan.matveev@hs-anhalt.de

# Method for synthetic features generation using 3D models and projective transformation

import os
import multiprocessing
import queue
import signal
import yaml
import argparse
import numpy as np
import itertools

from libs import lib_feature_extractor as fe, lib_transform_data as tdata, lib_transform_2d as t2d, \
    lib_transform_3d as t3d, lib_logging as log

logger = log.spawn_logger(f'{__file__}.log', formatter='%(asctime)s %(processName)s - %(message)s')


# Ignore keyboard interrupt in forks, let parent process to handle this gently
def init_child_process():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# Build an iterator based on given parameters
def init_iterator(conf, o_key):
    cam = conf['camera']
    obj = conf['obj'][o_key]
    obj_global = conf['obj_global']

    dim_ranges = init_scale_ranges([[1, *obj['scale']['range'], obj['scale']['n_points']]])
    rotate_y_rg = np.linspace(*obj['rotate_y']['range'], obj['rotate_y']['n_points'])
    cam_angles = np.linspace(*cam['angle']['range'], cam['angle']['n_points'])
    x_range = np.linspace(*obj_global['x_range']['range'], obj_global['x_range']['n_points'])
    y_range = np.linspace(*obj_global['y_range']['range'], obj_global['y_range']['n_points'])
    z_range = np.linspace(*obj_global['z_range']['range'], obj_global['z_range']['n_points'])
    thr_range = np.linspace(*obj_global['thr']['range'], obj_global['thr']['n_points'])

    it = itertools.product(cam_angles, dim_ranges[0], dim_ranges[1], dim_ranges[2], rotate_y_rg, x_range, y_range,
                           z_range, thr_range)

    total_iter = np.prod([rg.size for rg in (cam_angles, dim_ranges[0], dim_ranges[1], dim_ranges[2], rotate_y_rg,
                                             x_range, y_range, z_range, thr_range)])

    return it, total_iter


def init_scale_ranges(scale_range):
    """
    Generate range of values to which object is scaled
    :param scale_range: interval's border values
    :return: scale range for 3 dimensions
    """
    def build_dims_mask(in_dim):
        """
        Fill by zeros not used dimensions
        """
        in_dim = np.asarray(in_dim)
        mask = np.zeros((3, 3))
        mask[in_dim[:, 0].astype(int), :] = in_dim[:, 1:]

        return mask

    # Pass scale_range as [[dimension - id, lower - border, higher - border, amount - of - points - between]],
    # where dimension-id lies in range [0, 1, 2] that correspond to [X, Y, Z] (object width, height, depth)
    # To generate a mask for non-uniform scaling pass multiple lists, e.g. [[1, 1.4, 1.95, 10], [2, 0.4, 1.2, 3]] and
    # set not proportional argument while scaling
    dim_mask = build_dims_mask(scale_range)

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

    with open(path, 'r') as fi:
        lines = fi.readlines()

    vertices = np.array([parse_string(ln) for ln in lines if ln.startswith('v')], dtype='float')
    faces = [parse_string(ln) for ln in lines if ln.startswith('f')]
    faces = np.asarray([[int(el) for el in ln] for ln in faces]) - 1

    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # Bring to homogeneous form

    return vertices, faces


def get_status(i, total_iter):
    return f'{i / total_iter * 100:3.2f} %, {i} / {total_iter}'


def generate_features(o_key, conf, save_q, stop_event):
    """
    Main routine to generate features
    :param o_key: object to process by current worker
    :param conf: parsed config
    :param save_q: q where generated features are pushed
    :param stop_event: stop event
    """
    multiprocessing.current_process().name = o_key  # Set process name
    it, total_iter = init_iterator(conf, o_key)  # Generate iterator from parameters in config
    logger.info(f'Total iterations: {total_iter}')

    work_obj = conf['obj'][o_key]
    ry_init = work_obj['ry_init']
    o_class = work_obj['class']

    camera_params = yaml.safe_load(open(conf['camera']['params']))
    img_res = camera_params['optimized_res']
    f_l = camera_params['focal_length']
    intrinsic_raw = np.asarray(camera_params['optimized_matrix'])
    intrinsic = np.hstack((intrinsic_raw, np.zeros((3, 1))))
    intrinsic_args = (intrinsic, img_res)

    vertices, faces = parse_3d_obj_file(work_obj['file'])
    rw_system = t3d.Handler3D(vertices, operations=['s', 'ry', 't', 'rx'], k=intrinsic_args)

    mar = 1
    f_margins = {'left': mar, 'right': img_res[0] - mar, 'up': mar, 'bottom': img_res[1] - mar}

    data_to_save = list()
    prev_rx, prev_y = None, None
    i = 0
    entries_counter = 0

    for i, (cam_a, ww, hh, dd, ry, x, y, z, thr) in enumerate(it):
        rw_system.transform((True, np.asarray([ww, hh, dd])), np.deg2rad(ry_init + ry), np.asarray([x, y, z]),
                            np.deg2rad(cam_a))  # Transform object in 3D space and project to image plane

        mask = t2d.generate_image_plane(rw_system.img_points, faces, thr, img_res)
        basic_params = t2d.find_basic_params(mask)  # Extract basic parameters (ca, x, y, w, h) from the image
        if basic_params.size == 0:
            continue
        basic_params = t2d.calc_second_point(basic_params)  # Calculate opposite rectangle point and add it to array

        if tdata.is_crossing_margin(f_margins, basic_params):  # Filter objects intersecting frame border
            continue

        # Select row with maximal contour area and add dimension
        basic_params = basic_params[np.argmax(basic_params[:, 0])].reshape(1, basic_params.shape[1])

        # Update instance of feature extractor only when influencing parameters are changed
        if prev_rx != cam_a or prev_y != y:
            pc = fe.FeatureExtractor(cam_a, y, img_res, intrinsic_raw, f_l)
            prev_rx, prev_y = cam_a, y

        # Extract features from contours and bounding rectangles
        z_est, x_est, width_est, height_est, rw_ca_est = pc.extract_features(basic_params)

        # Get actual object shape from feature extractor instance
        shape_mtx = [mtx for mtx in rw_system.mtx_seq if isinstance(mtx, t3d.ScaleMtx)]
        ww, hh, dd, = shape_mtx[0].shape_cursor
        x_px, y_px, w_px, h_px = basic_params[0, 1:5].tolist()
        # Form entry of generated parameters to save
        data_to_save.append([cam_a, y, z_est[0], z, x_est[0], x, width_est[0], ww, height_est[0], hh, rw_ca_est[0],
                             o_key, o_class, ry, x_px, y_px, w_px, h_px, basic_params[0, 0], thr, dd])

        entries_counter += 1
        if entries_counter % 10000 == 0:
            save_q.put(data_to_save)
            data_to_save = list()
            logger.info(get_status(i + 1, total_iter))

        if stop_event.is_set():
            break

        if args.show:
            t2d.plt_2d_projections(rw_system.transformed_vertices)
            t2d.plot_image_plane(mask, img_res)

    if len(data_to_save) > 0:
        save_q.put(data_to_save)

    logger.info(f'Finished {get_status(i + 1, total_iter)}')


def saver_thr(out_f, q, stop_event):
    init_child_process()
    i = int()
    header = True
    while True:
        try:
            data_to_save = q.get(timeout=1)
            header = tdata.write_to_csv(header, data_to_save, out_f)
            i += len(data_to_save)
        except queue.Empty:
            pass

        if stop_event.is_set() and q.empty():
            logger.info(f'{i} entries has been written')
            break


def del_if_exists(out_f):
    if os.path.exists(out_f):
        os.remove(out_f)
        logger.warning(f'Previous output file removed: {out_f}')
    else:
        logger.info(f'Output file: {out_f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate features of 3D objects')
    parser.add_argument('config', action='store', help='path to the configuration file')
    parser.add_argument('--csv', action='store', help='path to the output csv file (default:features.csv )',
                        default='features.csv')
    parser.add_argument('--show', action='store_true', help='show the generated images')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    m = multiprocessing.Manager()
    stop_event_gen = m.Event()
    stop_event_saver = m.Event()
    data_save_q = m.Queue()

    del_if_exists(args.csv)
    saver = multiprocessing.Process(target=saver_thr, args=(args.csv, data_save_q, stop_event_saver), name='saver')

    o_keys = config['obj'].keys()

    iterable_args = itertools.product(o_keys, [config], [data_save_q], [stop_event_gen])
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
