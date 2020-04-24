import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def find_principal_point(vertices):
    return np.array([[(vertices[:, 0].min() + vertices[:, 0].max()) / 2,
                     vertices[:, 1].min(), vertices[:, 2].min(), 1]])


def build_dims_mask(in_dim):
    in_dim = np.asarray(in_dim)
    dim_mask = np.zeros((3, 3))
    dim_mask[in_dim[:, 0].astype(int), :] = in_dim[:, 1:]

    return dim_mask


def gen_dims_ranges_it(dim_mask):
    ranges_lst = [np.linspace(row[0], row[1], row[2].astype(int)) for row in dim_mask]
    ranges_lst = [np.zeros(1) if rg.size == 0 else rg for rg in ranges_lst]
    dims_it = itertools.product(ranges_lst[0], ranges_lst[1], ranges_lst[2])

    return dims_it


def gen_dims_ranges(dim_mask):
    ranges_lst = [np.linspace(row[0], row[1], row[2].astype(int)) for row in dim_mask]
    ranges_lst = [np.zeros(1) if rg.size == 0 else rg for rg in ranges_lst]

    return ranges_lst


def find_scale_f(prop, shape, req_dims):
    scale_f = req_dims / shape
    if prop:
        scale_f[:] = scale_f.max()
    else:
        scale_f[scale_f == 0] = scale_f[scale_f != 0].min()

    return scale_f


def parse_string(string):
    spl = [el.split('//') for el in string.split()]
    res = [el[0] for i, el in enumerate(spl) if i != 0]

    return res


def parse_3d_obj_file(path):
    step = 39.3701
    with open(path, "r") as fi:
        lines = fi.readlines()

    vertices = np.array([parse_string(ln) for ln in lines if ln.startswith("v")], dtype='float') / step
    faces = [parse_string(ln) for ln in lines if ln.startswith("f")]
    faces = np.asarray([[int(el) for el in ln] for ln in faces]) - 1

    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    return vertices, faces


class TranslateMtx(object):
    def __init__(self, key, vertices):
        self.mtx = np.identity(4)
        self.vertices = vertices

    def build(self, args):
        coords = args
        t = np.asarray(coords) - self.find_principal_point()[:-1]
        self.mtx[:-1, -1] = t
        return self.mtx

    def find_principal_point(self):
        return np.array([(self.vertices[:, 0].min() + self.vertices[:, 0].max()) / 2,
                         self.vertices[:, 1].min(), self.vertices[:, 2].min(), 1])


class ScaleMtx(object):
    def __init__(self, key, vertices):
        self.vertices = vertices
        self.mtx = np.identity(4)
        self.shape = self.measure_act_shape()

    def build(self, args):
        self.shape = self.measure_act_shape()
        prop, req_dims = args
        scale_f = np.asarray(req_dims) / self.shape
        if prop:
            scale_f[:] = scale_f.max()
        else:
            scale_f[scale_f == 0] = 1

        np.fill_diagonal(self.mtx, np.append(scale_f, 1))
        return self.mtx

    def measure_act_shape(self):
        return self.vertices[:, :-1].max(axis=0) - self.vertices[:, :-1].min(axis=0)


class IntrinsicMtx(object):
    def __init__(self, args, vertices, img_points):
        self.img_res, self.f_l, self.sens_dim = args
        self.mtx = np.eye(3, 4)
        np.fill_diagonal(self.mtx, self.f_l * self.img_res / self.sens_dim)
        self.mtx[:, 2] = np.append(self.img_res / 2, 1)  # Append 1 to replace old value in mtx after fill_diagonal

        self.img_points = img_points
        self.vertices = vertices

    def project_to_image(self):
        temp = self.vertices @ self.mtx.T
        self.img_points[:] = np.asarray([temp[:, 0] / temp[:, 2],
                                         temp[:, 1] / temp[:, 2]]).T
        self.img_points[:, 1] = self.img_res[1] - self.img_points[:, 1]  # Reverse along y axis


class RotationMtx(object):
    def __init__(self, key, vertices):
        self.mtx = np.identity(4)
        self.rot_map = {'rx': self.fill_rx_mtx, 'ry': self.fill_ry_mtx, 'rz': self.fill_rz_mtx}
        self.fill_function = self.rot_map[key]
        self.prev_angle = float()

    def build(self, angle):
        #ang = np.deg2rad(angle)
        self.fill_function(np.sin(angle), np.cos(angle))
        return self.mtx

    def fill_rx_mtx(self, a_sin, a_cos):
        self.mtx[1][1] = a_cos
        self.mtx[1][2] = -a_sin
        self.mtx[2][1] = a_sin
        self.mtx[2][2] = a_cos

    def fill_ry_mtx(self, a_sin, a_cos):
        self.mtx[0][0] = a_cos
        self.mtx[0][2] = a_sin
        self.mtx[2][0] = -a_sin
        self.mtx[2][2] = a_cos

    def fill_rz_mtx(self, a_sin, a_cos):
        self.mtx[0][0] = a_cos
        self.mtx[0][1] = -a_sin
        self.mtx[1][0] = a_sin
        self.mtx[1][1] = a_cos


class Handler3DNew(object):
    def __init__(self, vert, operations, k=None):
        self.vertices = vert
        self.transformed_vertices = np.copy(self.vertices)
        self.img_points = np.zeros((self.vertices.shape[0], 2), dtype='int32')
        self.operations = operations
        self.mtx_seq = {'rx': RotationMtx, 'ry': RotationMtx, 'rz': RotationMtx, 's': ScaleMtx, 't': TranslateMtx}

        self.mtx_seq = [self.mtx_seq[op](op, self.transformed_vertices) for op in self.operations]  # Create instances of matrices

        self.k_obj = IntrinsicMtx(k, self.transformed_vertices, self.img_points) if k is not None else None

    def transform(self,  *args):
        self.transformed_vertices[:] = np.copy(self.vertices)
        for mtx, arg in zip(self.mtx_seq, args):
            self.transformed_vertices[:] = self.transformed_vertices @ mtx.build(arg).T

        if self.k_obj is not None:
            self.k_obj.project_to_image()


def plt_2d_projections(vertices):
    fig, ax_lst = plt.subplots(2, 2, squeeze=True, figsize=(8.0, 13.0))
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ax_to_plot = [[[x, y], [z, y]],
                  [[x, z]]]
    fig.delaxes(ax_lst[1][1])

    color = 'k'
    s = 0.1
    marg = 0.1

    for (m, n), subplot in np.ndenumerate(ax_lst):
        if m == 1 and n == 1:
            break
        subplot.scatter(*ax_to_plot[m][n], color=color, s=s)
        subplot.set_aspect('equal', adjustable='box')
        min_x = min(ax_to_plot[m][n][0])
        max_x = max(ax_to_plot[m][n][0])
        subplot.set_xlim(min_x - marg, max_x + marg)
        subplot.set_ylim(min(ax_to_plot[m][n][1]) - marg, max(ax_to_plot[m][n][1]) + marg)
        subplot.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_image_plane(mask, img_res):
    plt.imshow(mask, cmap='gray')
    plt.xlim(0, img_res[0]), plt.ylim(img_res[1], 0)
    plt.show()


def generate_image_plane(img_points, faces, k_size, img_res):
    mask = np.zeros((img_res[1], img_res[0]), np.uint8)
    poly = np.take(img_points, faces, axis=0)
    for p in poly:
        cv2.fillConvexPoly(mask, p, color=255)
    # cv2.fillPoly(mask, pts=poly, color=255) # Works faster but not as desired
    cv2.dilate(mask, cv2.getStructuringElement(0, (int(k_size), int(k_size))), dst=mask)

    return mask


def find_basic_params(mask):
    cnts, _ = cv2.findContours(mask, mode=0, method=1)
    c_areas = [cv2.contourArea(cnt) for cnt in cnts]
    b_rects = [cv2.boundingRect(b_r) for b_r in cnts]

    return np.asarray(c_areas), np.asarray(b_rects)


class FeatureExtractor(object):
    def __init__(self, r_x, cam_h, img_res, sens_dim, f_l):
        self.r_x = np.deg2rad(r_x, dtype=np.float32)  # Camera rotation angle about x axis in radians
        self.cam_h = np.asarray(cam_h, dtype=np.float32)  # Ground y coord relative to camera (cam. is origin) in meters
        self.img_res = np.asarray(img_res, dtype=np.int16)  # Image resolution (width, height) in px
        self.sens_dim = np.asarray(sens_dim, dtype=np.float32)  # Camera sensor dimensions (width, height) in mm
        self.px_h_mm = self.sens_dim[1] / self.img_res[1]  # Height of a pixel in mm
        self.f_l = np.asarray(f_l, dtype=np.float32)  # Focal length in mm

        # Transformation matrices for 3D reconstruction
        intrinsic_mtx = IntrinsicMtx((self.img_res, self.f_l, self.sens_dim), None, None).mtx
        self.rev_intrinsic_mtx = np.linalg.inv(intrinsic_mtx[:, :-1])  # Last column is not needed in reverse transf.
        rot_x_mtx = RotationMtx('rx', None).build(self.r_x)
        self.rev_rot_x_mtx = np.linalg.inv(rot_x_mtx)

    def extract_features(self, ca_px, b_rect):
        # * Transform bounding rectangles to required shape
        # Important! Reverse the y coordinates of bound.rect. along y axis before transformations (self.img_res[1] - y)
        px_y_bottom_top = self.img_res[1] - np.stack((b_rect[:, 1] + b_rect[:, 3], b_rect[:, 1]), axis=1)
        # Distances from vertices to img center (horizon) along y axis, in px
        y_bottom_top_to_hor = self.img_res[1] / 2. - px_y_bottom_top
        np.multiply(y_bottom_top_to_hor, self.px_h_mm, out=y_bottom_top_to_hor)  # Convert to mm
        # Find angle between object pixel and central image pixel along y axis
        np.arctan(np.divide(y_bottom_top_to_hor, self.f_l, out=y_bottom_top_to_hor), out=y_bottom_top_to_hor)

        # * Find object distance in real world
        rw_distance = self.estimate_distance(y_bottom_top_to_hor[:, 0])  # Passed arg is angles to bottom vertices
        # * Find object height in real world
        rw_height = self.estimate_height(rw_distance, y_bottom_top_to_hor)

        # * Transform bounding rectangles to required shape
        # Build a single array from left and right rects' coords to compute within single vectorized transformation
        px_x_l_r = np.hstack((b_rect[:, 0], b_rect[:, 0] + b_rect[:, 2]))  # Left and right bottom coords
        # so the [:shape/2] belongs to left and [shape/2:] to the right bound. rect. coordinates
        x_lr_yb_hom = np.stack((px_x_l_r,
                                np.repeat(px_y_bottom_top[:, 0], 2),
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
        deg = abs(self.r_x) + ang_y_bot_to_hor
        distance = abs(self.cam_h) / np.where(deg >= 0, np.tan(deg), np.inf)

        return distance

    # Estimate coordinates of vertices in real world
    def estimate_3d_coordinates(self, x_lr_yb_hom, distance):
        # Z cam is a scaling factor which is needed for 3D reconstruction
        z_cam_coords = self.cam_h * np.sin(self.r_x) + distance * np.cos(self.r_x)
        z_cam_coords = np.expand_dims(np.repeat(z_cam_coords, 2), axis=0).T
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


def write_to_csv(header_, data_, out_file):
    df = pd.DataFrame(data_, columns=['cam_a', 'y', 'z_est', 'z', 'x_est', 'x', 'width_est', 'ww', 'height_est', 'hh',
                                      'rw_ca_est', 'o_name', 'o_class', 'ry', 'x_px', 'y_px', 'w_px', 'h_px', 'c_ar_px',
                                      'thr', 'dd'])
    df = df.round({"z_est": 2, "x_est": 2, "rw_ca_est": 3, "width_est": 2, "height_est": 2, 'x': 2, 'y': 2, 'z': 2,
                  'cam_a': 1, 'ry': 1, 'ww': 2, 'hh': 2, 'dd': 2})

    with open(out_file, 'a') as f:
        df.to_csv(f, header=header_, index=False)

    return False
