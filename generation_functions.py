import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
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


def center_3d_obj(vert):
    centers = [np.median([vert[:, 0].max(), vert[:, 0].min()]), vert[:, 1].min(),
               vert[:, 2].min()]  # xc, y_lowest, zc
    vert[:, 0] = vert[:, 0] - centers[0]
    vert[:, 1] = vert[:, 1] - centers[1]
    vert[:, 2] = vert[:, 2] - centers[2]

    return vert


def parse_3d_obj_file(path):
    step = 39.3701
    path = os.path.join('obj', path)
    with open(path, "r") as fi:
        lines = fi.readlines()

    vertices = np.array([parse_string(ln) for ln in lines if ln.startswith("v")], dtype='float') / step
    faces = [parse_string(ln) for ln in lines if ln.startswith("f")]
    faces = np.asarray([[int(el) for el in ln] for ln in faces]) - 1

    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    vertices = center_3d_obj(vertices)

    return vertices, faces


# # Defines interface to transformation matrices
# class TransformMtx(object):
#     def __init__(self, key):
#         self.mtx = None
#
#     def build(self, *args):
#         # Fills self.mtx accordingly and return mtx.T
#         pass


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
        ang = np.deg2rad(angle)
        self.fill_function(np.sin(ang), np.cos(ang))
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


def plot_mask(img_points, faces, k_size, img_res):
    mask = np.zeros((img_res[1], img_res[0]), np.uint8)
    poly = np.take(img_points, faces, axis=0)
    for p in poly:
        cv2.fillConvexPoly(mask, p, color=255)
    # cv2.fillPoly(mask, pts=poly, color=255) # Works faster but not as desired
    cv2.dilate(mask, cv2.getStructuringElement(0, (int(k_size), int(k_size))), dst=mask)

    return mask


def find_obj_params5(img_points, faces, height, pinhole_cam, k_size, img_res):

    mask = np.zeros((img_res[1], img_res[0]), np.uint8)

    for face in faces:
        poly = np.array([img_points[i - 1] for i in face])
        mask = cv2.fillPoly(mask, pts=[poly], color=255)

    mask = cv2.blur(mask, (int(k_size), int(k_size)))

    _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_a_pxs = [cv2.contourArea(cnt) for cnt in cnts]

    if any(c_a_pxs):
        c_a_px_i = c_a_pxs.index(max(c_a_pxs))
        c_a_px = c_a_pxs[c_a_px_i]

        b_r = x, y, w, h = cv2.boundingRect(cnts[c_a_px_i])
        d = pinhole_cam.pixels_to_distance(height, y + h)

        h_rw = pinhole_cam.get_height(height, d, b_r)
        w_rw = pinhole_cam.get_width(height, d, b_r)

        rect_area_rw = w_rw * h_rw
        rect_area_px = w * h
        extent = float(c_a_px) / rect_area_px
        c_a_rw = c_a_px * rect_area_rw / rect_area_px

        return [d, c_a_rw, w_rw, h_rw, extent, x, y, w, h, c_a_px]


def find_basic_params(mask):
    cnts, _ = cv2.findContours(mask, mode=0, method=1)
    c_areas = [cv2.contourArea(cnt) for cnt in cnts]
    cnts = [cnts[c_areas.index(max(c_areas))]]  # DELETE fix to avoid insignificant objects
    b_rects = [cv2.boundingRect(b_r) for b_r in cnts]

    return np.asarray(c_areas), np.asarray(b_rects)


class PinholeCam(object):
    def __init__(self, r_x, cam_h, img_res, sens_dim, f_l):
        self.r_x = np.deg2rad(r_x)  #  negative
        self.cam_h = cam_h # Should be negative
        self.img_res = img_res
        self.sens_dim = sens_dim
        self.f_l = f_l

        self.pixels_to_distance_v = np.vectorize(self.pixels_to_distance)
        self.distance = None

    def pixels_to_distance(self, n):
        #n = self.img_res[1] - n
        pxlmm = self.sens_dim[1] / self.img_res[1]  # print 'pxlmm ', pxlmm
        h_px = self.img_res[1] / 2 - n
        h_mm = h_px * pxlmm  # print 'hmm ', h_mm
        bo = np.arctan(h_mm / self.f_l)  # print 'bo ', np.rad2deg(bo)
        deg = abs(self.r_x) + bo
        tan = np.tan(deg) if deg >= 0 else -1.
        d = (abs(self.cam_h) / tan)
        return d

    def get_intrinsic_matrix(self):
        w_img, h_img = self.img_res
        w_ccd, h_ccd = self.sens_dim

        fx = self.f_l * w_img / float(w_ccd)
        fy = self.f_l * h_img / float(h_ccd)

        px = w_img / 2.0
        py = h_img / 2.0

        k = np.array([[fx, 0, px, 0],
                      [0, fy, py, 0],
                      [0, 0, 1, 0]])

        return k

    def get_extrinsic_matrix(self):
        ang = self.r_x
        # R_x(angle)
        r = np.array([[1, 0, 0, 0],
                      [0, np.cos(ang), -np.sin(ang), 0],
                      [0, np.sin(ang), np.cos(ang), 0],
                      [0, 0, 0, 1]])

        return r

    def extract_features(self, b_rect):
        # Reverse the lowest coordinate of bound.rect. along y axis before transformation
        lowest_y = self.img_res[1] - (b_rect[:, 1] + b_rect[:, 3])
        self.distance = self.pixels_to_distance_v(lowest_y)
        print(self.distance)

        self.find_3d_x(b_rect, lowest_y)

    def find_3d_x(self, b_rect, lowest_y):
        # Build a single array from left and right rects' coords to compute within single vectorized transformation
        xlr = np.hstack((b_rect[:, 0], b_rect[:, 0] + b_rect[:, 2]))  # Left and right bottom coords of bounding rect,
        # so the [:shape/2] belongs to left and [shape/2:] to the right bound. rect. coordinates
        xlr_yb_hom = np.vstack((xlr, np.repeat(lowest_y, 2), np.ones(2 * lowest_y.size))).T  #

        z_cam_coords = self.cam_h * np.sin(self.r_x) + self.distance * np.cos(self.r_x)
        z_cam_coords = np.expand_dims(z_cam_coords, axis=0).T
        cam_xlr_yb_h = xlr_yb_hom * np.repeat(z_cam_coords, 2, axis=0)

        k_inv = np.linalg.inv(self.get_intrinsic_matrix()[:, :-1])

        cam_xlr_yb_h = cam_xlr_yb_h.T
        camera_coords = k_inv @ cam_xlr_yb_h
        #camera_coords[1] = -camera_coords[1]  # Some magic

        camera_coords = np.vstack((camera_coords, np.ones((1, camera_coords.shape[1]))))
        print(camera_coords, camera_coords.shape)

        e_inv = np.linalg.inv(self.get_extrinsic_matrix())

        rw_coords = e_inv @ camera_coords
        print(rw_coords.T)



    # def get_3d_point(self, img_coords, y, z):
    #     z_cam_coords = (y * np.sin(np.radians(self.rw_angle))) + (z * np.cos(np.radians(self.rw_angle)))
    #
    #     img_coords_prime = z_cam_coords * img_coords
    #
    #     camera_coords = self.k_inv.dot(img_coords_prime.T)
    #     camera_coords[1] = -camera_coords[1]  # Some magic
    #
    #     camera_coords_prime = np.array([np.append(camera_coords, np.array(1))])
    #     rw_coords = self.e_inv.dot(camera_coords_prime.T)
    #
    #     return rw_coords.T[0]


    def calc_width(self, z_rw, b_rect, lowest_y):
        y_rw = self.cam_h

        br_left_down_2d = np.asarray([b_rect[:, 0], lowest_y, np.ones(lowest_y.shape)]).T
        br_right_down_2d = np.asarray([b_rect[:, 0] + b_rect[:, 2], lowest_y, np.ones(lowest_y.shape)]).T

        self.get_3d_point(br_left_down_2d)

        # br_left_down_2d = np.array([[x, y + h, 1]])
        # br_right_down_2d = np.array([x + w, y + h, 1])
        # br_down_2d = [br_left_down_2d, br_right_down_2d]
        # br_down_3d = [self.get_3d_point(br_vertex_2d, y_rw, z_rw) for br_vertex_2d in br_down_2d]
        #
        # br_left_down_3d = br_down_3d[0][0]
        # br_right_down_3d = br_down_3d[1][0]
        # br_x_down_list = [br_left_down_3d, br_right_down_3d]
        #
        # br_w_3d = abs(max(br_x_down_list) - min(br_x_down_list))

    def get_width(self, y_rw, z_rw, b_rect):
        x, y, w, h = b_rect

        br_left_down_2d = np.array([[x, y + h, 1]])
        br_right_down_2d = np.array([x + w, y + h, 1])
        br_down_2d = [br_left_down_2d, br_right_down_2d]
        br_down_3d = [self.get_3d_point(br_vertex_2d, y_rw, z_rw) for br_vertex_2d in br_down_2d]

        br_left_down_3d = br_down_3d[0][0]
        br_right_down_3d = br_down_3d[1][0]
        br_x_down_list = [br_left_down_3d, br_right_down_3d]

        br_w_3d = abs(max(br_x_down_list) - min(br_x_down_list))

        return br_w_3d