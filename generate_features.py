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

single = False
single = True
if single:
    scale_f = (1, 1, 1)
    cam_a = -30
    y_rotate = 20

    x, y, z = [0, -6, 7]
    # obj_3d.transform_3d(r_x=np.deg2rad(cam_a), r_y=np.deg2rad(y_rotate), coords=np.asarray((x, y, z)), scale=scale_f)
    # print(obj_3d.vertices)
    # gf.plt_2d_projections(obj_3d.vertices)

    # obj_3d.project_to_image_plan(scene['img_res'], scene['f_l'], scene['sens_dim'])
    #
    # mask = gf.plot_mask(obj_3d.img_points, obj_3d.faces, 1, scene['img_res'])
    #
    # plt.imshow(mask, cmap='gray')
    # plt.xlim(0, scene['img_res'][0]), plt.ylim(scene['img_res'][1], 0)
    # plt.show()
    #
    # c_areas, b_rects = gf.find_basic_params(mask)
    # print(c_areas, b_rects)
    #
    # new_cam = gf.PinholeCam(cam_a, y, scene['img_res'], scene['sens_dim'], scene['f_l'])
    # print('new', new_cam.pixels_to_distance(b_rects[0][1] + b_rects[0][3]))

    vertices, faces = gf.parse_3d_obj_file(os.path.join(sp.obj_dir_path, o_key))
    # Add principal point to vertices array
    vertices = np.vstack([vertices, gf.find_principal_point(vertices)])

    tracker = gf.OffsetTracker(vertices)
    scale_f = tracker.scale(True, np.asarray([0, 4, 0]))
    translate = tracker.translate(np.asarray([0, 0, 10]))

    # obj_3d = gf.Handler3DNew(vertices, operations=['ry'])
    # obj_3d.transform(np.deg2rad(work_obj['yr_init']))
    obj_3d = gf.Handler3DNew(vertices, operations=['ry', 's', 't'])
    obj_3d.transform(np.deg2rad(60), scale_f, np.array([10, 0, 10]))
    gf.plt_2d_projections(obj_3d.vertices)

    # rot_obj = gf.Handler3DNew(obj_3d.vertices, obj_3d.faces, operations=['s'])
    # tr = gf.OffsetTracker(rot_obj.vertices)
    # rot_obj.transform(tr.scale(True, (0, 2, 0)))
    # gf.plt_2d_projections(rot_obj.vertices)


    # obj_3d = gf.Handler3DNew(obj_3d.vertices, faces, operations=['s'])

    # t = tr.translate((0, 0, 0))
    # print(tr.find_principal_point())
    # s1 = tr.scale(True, (0, 4, 0))
    # print(s1)
    # t1 = tr.translate((10, 0, 0))
    # t2 = tr.translate((0, 0, 10))

    # print(tr.find_principal_point())
    #
    # t2 = tr.translate((0, 0, 10))
    # gf.plt_2d_projections(tr.vertices)

    # obj_3d.transform(s1)
    # gf.plt_2d_projections(obj_3d.vertices)


    # obj_3d = gf.Handler3DNew(*gf.parse_3d_obj_file(os.path.join(sp.obj_dir_path, o_key)),
    #                          operations=['ry', 's', 't', 'rx'])
    #
    # obj_3d.transform(np.deg2rad(y_rotate), np.asarray((x, y, z)), np.deg2rad(cam_a), np.asarray(scale_f))
    #                  (np.asarray(scene['img_res']), scene['f_l'], np.asarray(scene['sens_dim'])))


    # obj_3d.transform_3d(r_x=np.deg2rad(cam_a), r_y=np.deg2rad(y_rotate), coords=np.asarray((x, y, z)), scale=scale_f)
    # print(obj_3d.vertices)

else:
    start_time = timeit.default_timer()
    for i, (cam_a, ww, hh, dd, y_rotate, x, y, z, thr) in enumerate(it):

        scale_f = gf.find_scale_f(work_obj['dim']['prop'], obj_3d.shape, np.asarray((ww, hh, dd)))
        obj_3d.transform_3d(r_x=np.deg2rad(cam_a), r_y=np.deg2rad(y_rotate), coords=np.asarray((x, y, z)),
                                scale=scale_f)

        obj_3d.transform_3d(r_x=np.deg2rad(cam_a), r_y=np.deg2rad(y_rotate), coords=np.asarray((x, y, z)), scale=scale_f)
        obj_3d.project_to_image_plan(scene['img_res'], scene['f_l'], scene['sens_dim'])
        mask = gf.plot_mask(obj_3d.img_points, obj_3d.faces, 7, scene['img_res'])
        #print(i)

       # params = find_obj_params5(rot_obj_3d.vertices, rot_obj_3d.faces, y, pinhole_cam, thr, work_scene['img_res'])

        # print(obj_3d.img_points)
        # break

        if i == 1000:
            print(obj_3d.shape, obj_3d.measure_act_shape())
            break

    elapsed = timeit.default_timer() - start_time

    with open('time.txt', 'a') as fd:
        fd.write(str(elapsed) + '\n')
        print(elapsed)
    # dim_ranges_it = gf.gen_dims_ranges_it(dim_mask)











