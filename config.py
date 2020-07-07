#!/usr/bin/env python3.7

# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Configuration file containing parameters for script of synthetic data generation


# If logging level is DEBUG (10), the dict keys are starting with "test*" are processed.
# The rendered images will be shown on each iteration.
# If logging level is higher, e.g. INFO (20), the keys excluding "test*" are processed.
# In this case, no images will be rendered and the data stored in *.csv file.
# The keys starting with "_" are ignored in both scenarios. Used to keep possibly useful parameters that
# were used previously.

loglevel = 20
obj_dir_path = 'obj/'  # Directory containing preliminary prepared objects in wavefront.obj format

st_ry = (0, 90, 5)  # Standard range of object rotation angles about y axis (movement direction imitation)
st_flip = 180   # Initial offset of r_y (some objects are initially rotated by back to a camera)

# Object transformation parameters
# main key '*.obj': corresponds to name of .obj file
# key 'dim':
#  - key 'val': [[dimension-id, lower-border, higher-border, amount-of-points-between]],
# where dimension-id lies in range [0, 1, 2] that correspond to [X, Y, Z] (object width, height, depth)
#  - key 'prop' - scale object proportionally in all dimensions
# key 'rotate_y': see description above for st_ry variable
# key 'ry_init': see description above  st_flip variable

obj_info = {'walking-man.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                                'ry_init': st_flip},
            'woman-1.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                            'ry_init': st_flip},
            'running-boy.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                                'ry_init': st_flip},
            'standing-man.obj': {'dim': {'prop': True, 'val': [[1, 1.4, 1.95, 10]]}, 'rotate_y': st_ry, 'o_class': 1,
                                 'ry_init': st_flip},
            'cyclist-1.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2, 9]]}, 'rotate_y': st_ry, 'o_class': 2,
                              'ry_init': 0},
            '_pair-1.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2.1, 9]]}, 'rotate_y': st_ry, 'o_class': 2,
                            'ry_init': st_flip},
            '_pair-2.obj': {'dim': {'prop': True, 'val': [[1, 1.65, 2.1, 9]]}, 'rotate_y': st_ry, 'o_class': 2,
                            'ry_init': st_flip},
            '_car-3.obj': {'dim': {'prop': True, 'val': [[1, 1.5, 2, 9]]}, 'rotate_y': st_ry, 'o_class': 3,
                          'ry_init': st_flip},
            'car-3.obj': {'dim': {'prop': True, 'val': [[1, 1.5, 3, 9]]}, 'rotate_y': st_ry, 'o_class': 3,
                          'ry_init': st_flip},
            'car-2.obj': {'dim': {'prop': True, 'val': [[1, 1.5, 3, 9]]}, 'rotate_y': (90, 90, 1), 'o_class': 3,
                          'ry_init': st_flip},
            '_test-obj.obj': {'dim': {'prop': True, 'val': [[1, 1.8, 1.8, 1]]}, 'rotate_y': (0, 0, 1), 'o_class': 1,
                             'ry_init': st_flip},
            '_test-obj-2.obj': {'dim': {'prop': True, 'val': [[1, 1.71, 1.71, 1]]}, 'rotate_y': (90, 90, 1), 'o_class': 1,
                              'ry_init': st_flip},
            'test-obj-3.obj': {'dim': {'prop': True, 'val': [[1, 3, 3, 1]]}, 'rotate_y': (45, 45, 1),
                               'o_class': 1,
                               'ry_init': st_flip}}




# Parameters of scene to be generated.
# Parameters which are given by range are passed to np.arrange function
# main key '*': just a name
# key 'cam_angle': camera incline towards to ground surface: 0 deg. - the camera is parallel to the ground surface;
# -90 deg. - camera points perpendicularly down
# key 'x_range': object coordinates in meters along x axis (left, right relatively camera)
# key 'y_range': ground surface coordinates in meters relatively to a camera origin (e.g. -5 is 5m of camera height)
# key 'z_range': distance to an object from camera in meters
# key 'img_res': resulting image resolution (width, height) in pixels
# key 'f_l': a camera focal length in mm
# key 'sens_dim': physical camera's sensor dimensions in mm (width, height)
# key 'thr_range': size of a kernel for morphological dilate on the resulting mask to imitate real images motion blur
scene_info = {'_scene_a': {'cam_angle': (0, -70, -3), 'x_range': (-8, 8, 2), 'y_range': (-2, -7, -0.2),
                           'z_range': (1, 30, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                           'thr_range': (1, 26, 12), 'cxcy': (640, 360)},
              '_test_1': {'cam_angle': (-22, -23, -1), 'x_range': (-2, -1, 1), 'y_range': (-3.2, -3.3, -0.1),
                          'z_range': (13, 14, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                          'thr_range': (1, 2, 1), 'cxcy': (640, 360)},
              'test_2': {'cam_angle': (-39, -40, -1), 'x_range': (0, 1, 2), 'y_range': (-3.325, -3.4, -0.2),
                         'z_range': (3.4, 3.5, 0.2), 'img_res': (1024, 768), 'f_l': 2.2, 'sens_dim': (4.8502388678445065,
                                                                                                3.6501095778269583),
                         'thr_range': (1, 2, 1), 'cxcy': (517.5116402, 365.84214009)},

              'lamp_pole_1': {'cam_angle': (-39, -40, -1), 'x_range': (-8, 9, 1), 'y_range': (-3.325, -3.4, -0.2),
                         'z_range': (1, 15, 1), 'img_res': (1024, 768), 'f_l': 2.2, 'sens_dim': (4.8502388678445065,
                                                                                                3.6501095778269583),
                         'thr_range': (1, 26, 12), 'cxcy': (517.5116402, 365.84214009)},
              '_for_plot': {'cam_angle': (-22, -23, -1), 'x_range': (-8, 10, 2), 'y_range': (-3.1, -3.2, -0.1),
                            'z_range': (1, 31, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                            'thr_range': (1, 26, 12), 'cxcy': (640, 360)},
              '_rw_scenes': {'cam_angle': (-10, -33, -3), 'x_range': (-8, 10, 2), 'y_range': (-3, -5.5, -0.5),
                             'z_range': (1, 31, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                             'thr_range': (1, 26, 12), 'cxcy': (640, 360)},
              '_all_scenes': {'cam_angle': (-0, -95, -5), 'x_range': (-8, 10, 2), 'y_range': (-3, -10.5, -0.5),
                             'z_range': (1, 31, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                             'thr_range': (1, 26, 12), 'cxcy': (640, 360)},
              '_all_scenes_detailed': {'cam_angle': (-0, -92, -2), 'x_range': (-8, 10, 2), 'y_range': (-3, -10.2, -0.2),
                             'z_range': (1, 31, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                             'thr_range': (1, 26, 12), 'cxcy': (640, 360)},
              '_sel_sc': {'cam_angle': (-13, -16, -3), 'x_range': (-8, 10, 2), 'y_range': (-3, -3.5, -0.5),
                          'z_range': (1, 31, 1), 'img_res': (1280, 720), 'f_l': 3.6, 'sens_dim': (3.4509, 1.9373),
                          'thr_range': (1, 26, 12), 'cxcy': (640, 360)}}

# Used for noise generation and model training later on
# IMPORTANT! Select a particular scene
processing_scene = scene_info['lamp_pole_1']

# Mapping the most important columns' names in csv file
cam_a_k = 'cam_a'         # Camera angle relative to the ground surface in range [0, -90] deg.
cam_y_k = 'y'             # Ground surface offset (negative camera height) relative to camera origin in range [-3, -n] m
w_k = 'width_est'         # Feature - estimated object width
h_k = 'height_est'        # Feature - estimated object height
ca_k = 'rw_ca_est'        # Feature - estimated object contour area
z_k = 'z_est'             # Feature - estimated object distance from a camera
o_class_k = 'o_class'     # Object class as an integer, where 0 is a noise class
o_name_k = 'o_name'       # Object name as a string
b_rec_k = ('x_px', 'y_px', 'w_px', 'h_px')  # Parameters of a bounding rectangle
