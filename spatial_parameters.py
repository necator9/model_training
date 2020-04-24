# If logging level is DEBUG (10), the dict keys are starting with "test*" are be processed.
# The rendered images will be shown on each iteration.
# If logging level is higher, e.g. INFO (20), the keys excluding "test*" are processed.
# In this case, no images will be rendered and the data stored in *.csv file.
# The keys starting with "_" are ignored in both scenarios

loglevel = 10

"""
key 'dim': [[dimension-id, lower-border, higher-border, amount-of-points-between]],
where dimension-id lies in range [0, 1, 2] that correspond to [X, Y, Z] (object width, height, depth)
"""
obj_dir_path = '/home/ivan/Nextcloud/PhD_thesis/3dsmax/low_poly_obj/'

st_ry = (0, 90, 5)
st_flip = 180


obj_info = {'walking-man.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                                'ry_init': st_flip},
            'woman-1.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                            'ry_init': st_flip},
            'running-boy.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 2, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                                'ry_init': st_flip},
            'cyclist-1.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2.1, 9]]}, 'rotate_y': st_ry, 'o_class': 3,
                              'ry_init': 0},
            'pair-1.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2.1, 9]]}, 'rotate_y': st_ry, 'o_class': 2,
                           'ry_init': st_flip},
            'pair-2.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2.1, 9]]}, 'rotate_y': st_ry, 'o_class': 2,
                           'ry_init': st_flip},
            'car-3.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2.1, 9]]}, 'rotate_y': st_ry, 'o_class': 4,
                          'ry_init': st_flip},
            'test-obj.obj': {'dim': {'prop': True, 'val': [[1, 2, 2, 1]]}, 'rotate_y': (20, 20, 1), 'o_class': 1,
                             'ry_init': 180}}


scene_info = {'_scene_a': {'cam_angle': (0, -70, -3), 'x_range': (-8, 8, 2), 'y_range': (-2, -7, -0.2),
                           'z_range': (1, 30, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                           'thr_range': (1, 26, 12)},
              'test_1': {'cam_angle': (-20, -21, -1), 'x_range': (-2, -1, 1), 'y_range': (-5, -6, -1),
                         'z_range': (8, 9, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                         'thr_range': (1, 2, 1)},
              'for_plot': {'cam_angle': (-22, -23, -1), 'x_range': (-8, 10, 2), 'y_range': (-3.1, -3.2, -0.1),
                           'z_range': (1, 31, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                           'thr_range': (1, 26, 12)}}





