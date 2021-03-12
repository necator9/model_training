# Mapping the most column names in csv file
# Used for feature generation and training

cam_a_k = 'cam_a'         # Camera incline relative to a ground surface, deg
cam_y_k = 'y'             # Ground surface offset (negative camera height) relative to a camera origin, m
z_est_k = 'z_est'         # Distance to the closest object point (for a camera) estimated by feature extractor, m
z_k = 'z'                 # Real object distance the closest object point (for a camera), m
x_est_k = 'x_est'         # Central object x coordinate estimated by feature extractor, m
x_k = 'x'                 # Real central object x coordinate, m
w_est_k = 'width_est'     # Object width estimated by feature extractor, m
ww_k = 'ww'               # Real object width, m
h_est_k = 'height_est'    # Object height estimated by feature extractor, m
hh_k = 'hh'               # Real object height, m
ca_est_k = 'rw_ca_est'    # Object contour area estimated by feature extractor, m2 
o_name_k = 'o_name'       # Unique name of an object
o_class_k = 'o_class'     # Object class as an integer, where 0 is a noise class
ry_k = 'ry'               # Initial offset of r_y (some objects are initially rotated by back to a camera)
b_rec_k = ('x_px', 'y_px', 'w_px', 'h_px')  # Parameters of a bounding rectangle:
# 0 - left upper x coordinate of an object bounding rectangle in image plane, px
# 1 - left upper y coordinate of an object bounding rectangle in image plane, px
# 2 - width of an object bounding rectangle in image plane, px
# 3 - height of an object bounding rectangle in image plane, px
c_ar_px_k = 'c_ar_px'     # Object contour area in image plane, px
thr_k = 'thr'             # Size of the used kernel for morphological dilate on the resulting mask to imitate motion blur
dd_k = 'dd'               # Real object depth, m

# Features used for training
feature_vector = [w_est_k, h_est_k, z_est_k]  # Name of columns are used for training  cf.ca_k,
