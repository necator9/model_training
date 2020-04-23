from pinhole_camera_model import PinholeCameraModel
import os
import numpy as np

import spatial_parameters as sp
import generation_functions as gf
import numpy as np
import matplotlib.pyplot as plt

import itertools

import timeit

# out_file = path.join('{}.csv'.format(argv[1])) if len(argv) == 2 else 'default_filename.csv'
#gf.plt_2d_projections(obj_3d.vertices)

o_key = 'walking-man.obj'
work_obj = sp.obj_info[o_key]
scene = sp.scene_info['scene_a']

dim_mask = gf.build_dims_mask(work_obj['dim']['val'])
dim_ranges = gf.gen_dims_ranges(dim_mask)
rotate_y_rg = np.linspace(*work_obj['rotate_y'])

cam_angles = np.arange(*scene['cam_angle'])
x_range = np.arange(*scene['x_range'])
y_range = np.arange(*scene['y_range'])
z_range = np.arange(*scene['z_range'])
thr_range = np.arange(*scene['thr_range'])

it = itertools.product(cam_angles, dim_ranges[0], dim_ranges[1], dim_ranges[2], rotate_y_rg, x_range, y_range, z_range,
                       thr_range)


# obj_3d = gf.Handler3D(*gf.parse_3d_obj_file(os.path.join(sp.obj_dir_path, o_key)),
#                       yr_init=np.deg2rad(work_obj['yr_init']))
# obj_3d.transform_3d(r_y=np.deg2rad(0), coords=np.asarray([0, 0, 0]), scale=np.asarray([2, 1, 1]))
# gf.plt_2d_projections(obj_3d.vertices)
vertices, faces = gf.parse_3d_obj_file(os.path.join(sp.obj_dir_path, o_key))
intrinsic = (np.asarray(scene['img_res']), scene['f_l'], np.asarray(scene['sens_dim']))


single = False
single = True
if single:
    ww, hh, dd = 3, 2, 0
    cam_a = -10
    y_rotate = 0
    x, y, z = [0, -1, 15]

    rw_system = gf.Handler3DNew(vertices, operations=['s', 'ry', 't', 'rx'], k=intrinsic)
    rw_system.transform((False, np.asarray([ww, hh, dd])), np.deg2rad(y_rotate), np.asarray([x, y, z]),
                        np.deg2rad(cam_a))

    gf.plt_2d_projections(rw_system.transformed_vertices)
    mask = gf.plot_mask(rw_system.img_points, faces, 7, scene['img_res'])

    plt.imshow(mask, cmap='gray')
    plt.xlim(0, scene['img_res'][0]), plt.ylim(scene['img_res'][1], 0)
    plt.show()

    c_ar, b_rect = gf.find_basic_params(mask)

    if any(c_ar):
        # ONLY FOR SYNTHETIC GENERATOR DELETE fix to avoid insignificant objects
        max_idx = np.argmax(c_ar)
        c_ar, b_rect = np.expand_dims(c_ar[max_idx], axis=0), np.expand_dims(b_rect[max_idx], axis=0)
        print(c_ar, b_rect)
        #######################

        pc = gf.FeatureExtractor(cam_a, y, np.asarray(scene['img_res']), np.asarray(scene['sens_dim']), scene['f_l'])



        pc.extract_features(c_ar, b_rect)




else:
    rw_system = gf.Handler3DNew(vertices, operations=['s', 'ry', 't', 'rx'], k=intrinsic)

    start_time = timeit.default_timer()
    for i, (cam_a, ww, hh, dd, y_rotate, x, y, z, thr) in enumerate(it):
        rw_system.transform((True, np.asarray([ww, hh, dd])), y_rotate, np.asarray([x, y, z]), cam_a)
        mask = gf.plot_mask(rw_system.img_points, faces, 7, scene['img_res'])
        # scale_f = gf.find_scale_f(work_obj['dim']['prop'], rw_system.shape, np.asarray((ww, hh, dd)))
        # rw_system.transform_3d(r_x=np.deg2rad(cam_a), r_y=np.deg2rad(y_rotate), coords=np.asarray((x, y, z)),
        #                        scale=scale_f)
        #
        # rw_system.transform_3d(r_x=np.deg2rad(cam_a), r_y=np.deg2rad(y_rotate), coords=np.asarray((x, y, z)), scale=scale_f)
        # rw_system.project_to_image_plan(scene['img_res'], scene['f_l'], scene['sens_dim'])
        # mask = gf.plot_mask(rw_system.img_points, rw_system.faces, 7, scene['img_res'])
        #print(i)

       # params = find_obj_params5(rot_obj_3d.vertices, rot_obj_3d.faces, y, pinhole_cam, thr, work_scene['img_res'])

        # print(obj_3d.img_points)
        # break
        # print(i)

        if i == 1000:
            # print(rw_system.shape, rw_system.measure_act_shape())
            break

    elapsed = timeit.default_timer() - start_time

    with open('time.txt', 'a') as fd:
        fd.write(str(elapsed) + '\n')
        print(elapsed)
    # dim_ranges_it = gf.gen_dims_ranges_it(dim_mask)



     #   PINHOLE_CAM = PinholeCameraModel(rw_angle=cam_a, f_l=scene['f_l'], w_ccd=scene['sens_dim'][0],
     #                                    h_ccd=scene['sens_dim'][1], img_res=scene['img_res'])

     #   d = [PINHOLE_CAM.pixels_to_distance(y, y_ao + h_ao) for (x, y_ao, w, h_ao) in b_rect]
     #   w_ao_rw = [PINHOLE_CAM.get_width(y, dd, br) for dd, br in zip(d, b_rect)]
     #   h_ao_rw = [PINHOLE_CAM.get_height(y, dd, br) for dd, br in zip(d, b_rect)]
     #   print(d, w_ao_rw, h_ao_rw, 'result old')







