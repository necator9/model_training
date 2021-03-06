# Created by Ivan Matveev at 05.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Class for object features extraction

import numpy as np

from libs import lib_transform_3d as t3d


class FeatureExtractor(object):
    """
    Extract object features from given bounding rectangles and contour areas
    :param r_x: # Camera rotation angle about x axis in radians
    :param cam_h: # Ground y coord relative to camera (cam. is origin) in meters
    :param img_res: # Image resolution (width, height) in px
    :param f_l: # Focal length in mm
    """
    def __init__(self, r_x, cam_h, img_res, intrinsic, f_l):
        self.r_x = np.deg2rad(r_x, dtype=np.float32)  # Camera rotation angle about x axis in radians
        self.cam_h = np.asarray(cam_h, dtype=np.float32)  # Ground y coord relative to camera (cam. is origin) in meters
        self.img_res = np.asarray(img_res, dtype=np.int16)  # Image resolution (width, height) in px
        self.f_l = f_l  # Focal length in mm
        # Camera sensor dimensions (width, height) in mm
        self.sens_dim = self.f_l * self.img_res / intrinsic.diagonal()[:2]
        self.px_h_mm = self.sens_dim[1] / (self.f_l * self.img_res[1])  # Scaling between pixels in millimeters
        self.cx_cy = intrinsic[:2, 2]

        # Transformation matrices for 3D reconstruction
        self.rev_intrinsic_mtx = np.linalg.inv(intrinsic)
        rot_x_mtx = t3d.RotationMtx('rx', None).build(self.r_x)
        self.rev_rot_x_mtx = np.linalg.inv(rot_x_mtx)

    def extract_features(self, basic_params):  # [[c_ar, x, y, w, h, p2x, p2y]]
        ca_px, b_rect = basic_params[:, 0], basic_params[:, 1:]
        # * Transform bounding rectangles to required shape
        # Important! Reverse the y coordinates of bound.rect. along y axis before transformations (self.img_res[1] - y)
        px_y_bottom_top = self.img_res[1] - b_rect[:, [5, 1]]
        # Distances from vertices to img center (horizon) along y axis, in px
        y_bottom_top_to_hor = self.cx_cy[1] - px_y_bottom_top
        # Convert to mm and find angle between object pixel and central image pixel along y axis
        np.arctan(np.multiply(y_bottom_top_to_hor, self.px_h_mm, out=y_bottom_top_to_hor), out=y_bottom_top_to_hor)

        # * Find object distance in real world
        rw_distance = self.estimate_distance(y_bottom_top_to_hor[:, 0])  # Passed arg is angles to bottom vertices
        # * Find object height in real world
        rw_height = self.estimate_height(rw_distance, y_bottom_top_to_hor)

        # * Transform bounding rectangles to required shape
        # Build a single array from left and right rects' coords to compute within single vectorized transformation
        px_x_l_r = np.hstack((b_rect[:, 0], b_rect[:, 4]))  # Left and right bottom coords
        # so the [:shape/2] belongs to left and [shape/2:] to the right bound. rect. coordinates
        x_lr_yb_hom = np.stack((px_x_l_r,
                                np.tile(px_y_bottom_top[:, 0], 2),
                                np.ones(2 * px_y_bottom_top.shape[0])), axis=1)

        # * Find object coordinates in real world
        left_bottom, right_bottom = self.estimate_3d_coordinates(x_lr_yb_hom, rw_distance)
        # * Find object width in real world
        rw_width = np.absolute(left_bottom[:, 0] - right_bottom[:, 0])

        # * Find contour area in real world
        rw_rect_a = rw_width * rw_height
        px_rect_a = b_rect[:, 2] * b_rect[:, 3]
        rw_ca = ca_px * rw_rect_a / px_rect_a

        return rw_distance, left_bottom[:, 0] - rw_width / 2, rw_width, rw_height, rw_ca

    # Estimate distance to the bottom pixel of a bounding rectangle. Based on assumption that object is aligned with the
    # ground surface. Calculation uses angle between vertex and optical center along vertical axis
    def estimate_distance(self, ang_y_bot_to_hor):
        deg = ang_y_bot_to_hor - self.r_x
        distance = abs(self.cam_h) / np.where(deg >= 0, np.tan(deg), np.inf)

        return distance

    # Estimate coordinates of vertices in real world
    def estimate_3d_coordinates(self, x_lr_yb_hom, distance):
        # Z cam is a scaling factor which is needed for 3D reconstruction
        z_cam_coords = self.cam_h * np.sin(self.r_x) + distance * np.cos(self.r_x)
        z_cam_coords = np.expand_dims(np.tile(z_cam_coords, 2), axis=0).T
        cam_xlr_yb_h = x_lr_yb_hom * z_cam_coords

        # Transform from image plan to camera coordinate system
        camera_coords = self.rev_intrinsic_mtx @ cam_xlr_yb_h.T
        camera_coords = np.vstack((camera_coords, np.ones((1, camera_coords.shape[1]))))  # To homogeneous form

        # Transform from to camera to real world coordinate system
        rw_coords = self.rev_rot_x_mtx @ camera_coords

        left_bottom, right_bottom = np.split(rw_coords.T, 2, axis=0)  # Split into left/right vertices
        # left_bottom = [[ X,  Y,  Z,  1],  #  - The structure of a returning vertices array, where each row is
        #                [.., .., .., ..]...]    different rectangle. The right_bottom has the same format
        return left_bottom, right_bottom

    # Estimate height of object in real world
    def estimate_height(self, distance, ang_y_bot_top_to_hor):
        angle_between_pixels = np.absolute(ang_y_bot_top_to_hor[:, 0] - ang_y_bot_top_to_hor[:, 1])
        gamma = np.arctan(distance * 1. / abs(self.cam_h))
        beta = np.pi - angle_between_pixels - gamma
        height = np.hypot(abs(self.cam_h), distance) * np.sin(angle_between_pixels) / np.sin(beta)

        return height
