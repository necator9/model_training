
"""
key 'dim': [[dimension-id, lower-border, higher-border, amount-of-points-between]],
where dimension-id lies in range [0, 1, 2] that correspond to [X, Y, Z] (object width, height, depth)
"""
obj_dir_path = "/home/ivan/Nextcloud/PhD_thesis/3dsmax/low_poly_obj/"

obj_info = {'walking-man.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]},
                                'flip_y': True, 'rotate_y': (0, 90, 5), 'o_class': 1, 'yr_init': 180}}


scene_info = {'scene_a': {'cam_angle': (0, -70, -3), 'x_range': (-8, 8, 2), 'y_range': (-2, -7, -0.2),
                          'z_range': (1, 30, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                          'thr_range': (1, 26, 12)}}





