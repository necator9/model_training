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
