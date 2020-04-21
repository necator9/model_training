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


# Defines interface to transformation matrices
class TransformMtx(object):
    def __init__(self, key):
        self.mtx = None
        self.args = None

    def build(self, *args):
        # Fills self.mtx accordingly and return mtx.T
        pass

    def calculate_offset(self, *args):
        self.args = args


class TranslateMtx(TransformMtx):
    def __init__(self, key):
        super(TranslateMtx, self).__init__(key)
        self.mtx = np.identity(4)

    def build(self, t):
        self.mtx[:-1, -1] = t
        return self.mtx



class ScaleMtx(TransformMtx):
    def __init__(self, key):
        super(ScaleMtx, self).__init__(key)
        self.mtx = np.identity(4)

    def build(self, scale):
        np.fill_diagonal(self.mtx, np.append(scale, 1))
        return self.mtx


class IntrinsicMtx(TransformMtx):
    def __init__(self, key):
        super(IntrinsicMtx, self).__init__(key)
        self.mtx = np.eye(3, 4)

    def build(self, args):
        img_res, f_l, sens_dim = args
        print(img_res, f_l, sens_dim)
        np.fill_diagonal(self.mtx, f_l * img_res / sens_dim)
        self.mtx[:, 2] = np.append(img_res / 2, 1)  # Append 1 to replace old value in mtx after fill_diagonal
        return self.mtx


class RotationMtx(TransformMtx):
    def __init__(self, key):
        super(RotationMtx, self).__init__(key)
        self.mtx = np.identity(4)
        self.rot_map = {'rx': self.fill_rx_mtx, 'ry': self.fill_ry_mtx, 'rz': self.fill_rz_mtx}
        self.fill_function = self.rot_map[key]

    def build(self, angle):
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
    def __init__(self, vert, operations):
        self.vertices = vert
        self.map = {'rx': RotationMtx, 'ry': RotationMtx, 'rz': RotationMtx, 's': ScaleMtx,
                    't': TranslateMtx, 'k': IntrinsicMtx}

        self.mtx_seq = self._construct_sequence(operations)
        self.img_points = None

    def _construct_sequence(self, operations):
        return [self.map[op](op) for op in operations]  # Create instances of matrices

    def transform(self, *args):
        mtx_list = [mtx.build(arg).T for mtx, arg in zip(self.mtx_seq, args)]
        if IntrinsicMtx in self.mtx_seq:
            intr_idx = self.mtx_seq.index(IntrinsicMtx)
            intrinsic_mtx = mtx_list.pop(intr_idx)
            self.vertices = np.linalg.multi_dot([self.vertices] + mtx_list)
            self.img_points = self.vertices @ intrinsic_mtx
            self.img_points = np.asarray([self.img_points[:, 0] / self.img_points[:, 2],
                                          self.img_points[:, 1] / self.img_points[:, 2]], dtype='int32').T
            # args[intr_idx][0][1] is the image height
            self.img_points[:, 1] = args[intr_idx][0][1] - self.img_points[:, 1]  # Reverse along y axis
        else:
            print(mtx_list)
            # for mtx in mtx_list:
            #     self.vertices = self.vertices @ mtx
            self.vertices = np.linalg.multi_dot([self.vertices] + mtx_list)


class OffsetTracker(object):
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices)
        self.principal_point = self.find_principal_point()
        self.shape = self.measure_act_shape()

    def find_principal_point(self):
        return np.array([(self.vertices[:, 0].min() + self.vertices[:, 0].max()) / 2,
                         self.vertices[:, 1].min(), self.vertices[:, 2].min()])

    def measure_act_shape(self):
        return self.vertices[:, :-1].max(axis=0) - self.vertices[:, :-1].min(axis=0)

    def translate(self, coords):
        t = np.asarray(coords) - self.principal_point
        self.principal_point += t
        return t

    def scale(self, prop, req_dims):
        scale_f = np.asarray(req_dims) / self.shape
        if prop:
            scale_f[:] = scale_f.max()
        else:
            scale_f[scale_f == 0] = 1

        self.shape *= scale_f
        self.principal_point *= scale_f
        return scale_f






class Handler3D(object):
    def __init__(self, vert, faces, yr_init=0):
        if vert.shape[1] == 3:
            self.vertices = np.hstack((vert, np.ones((vert.shape[0], 1))))
        else:
            self.vertices = vert

        self.faces = faces
        self.shape = self.measure_act_shape()
        self._xr_ang = 0
        self._yr_ang = 0
        self._zr_ang = 0

        self.img_points = None

        self.reset_angle(yr_init)
        self.principal_point = self.find_principal_point()

    def measure_act_shape(self):
        return self.vertices[:, :-1].max(axis=0) - self.vertices[:, :-1].min(axis=0)

    def find_principal_point(self):
        return np.array([(self.vertices[:, 0].min() + self.vertices[:, 0].max()) / 2,
                         self.vertices[:, 1].min(), self.vertices[:, 2].min()])

    def transform_3d(self, r_x=None, r_y=None, r_z=None, coords=np.array([0, 0, 0]), scale=np.array([1, 1, 1])):
        to_multiply = [self.vertices]

        coords_passed = coords is not self.transform_3d.__defaults__[3]
        scale_passed = scale is not self.transform_3d.__defaults__[4]

        if coords_passed and scale_passed:
            self.shape *= scale
            self.principal_point *= scale
            t = coords - self.principal_point
            rotation = scale_and_translate_mtx(t, scale)
            self.principal_point += t
            to_multiply.append(rotation.T)

        if coords_passed and not scale_passed:
            t = coords - self.principal_point
            rotation = scale_and_translate_mtx(t, scale)
            self.principal_point += t
            to_multiply.append(rotation.T)

        if scale_passed and not coords_passed:
            self.shape *= scale
            principal_point_old = self.principal_point
            self.principal_point *= scale
            t = principal_point_old - self.principal_point
            rotation = scale_and_translate_mtx(t, scale)
            self.principal_point += t
            to_multiply.append(rotation.T)

        if r_x is not None:
            rotation = rotation_mtx_wrapper(r_x - self._xr_ang, fill_rx_mtx)
            self._xr_ang += r_x
            to_multiply.append(rotation.T)

        if r_y is not None:
            rotation = rotation_mtx_wrapper(r_y - self._yr_ang, fill_ry_mtx)
            self._yr_ang += r_y
            to_multiply.append(rotation.T)

        if r_z is not None:
            rotation = rotation_mtx_wrapper(r_z - self._zr_ang, fill_rz_mtx)
            self._zr_ang += r_z
            to_multiply.append(rotation.T)

        if len(to_multiply) > 1:
            self.vertices = np.linalg.multi_dot(to_multiply)

    def project_to_image_plan(self, img_res, f_l, sens_dim):
        intrinsic_mtx = fill_intrinsic_mtx(img_res, f_l, sens_dim)
        self.img_points = self.vertices @ intrinsic_mtx.T
        self.img_points = np.asarray([self.img_points[:, 0] / self.img_points[:, 2],
                                      self.img_points[:, 1] / self.img_points[:, 2]], dtype='int32').T
        self.img_points[:, 1] = img_res[1] - self.img_points[:, 1]  # Reverse along y axis

    def reset_angle(self, _yr_ang):
        rotation = rotation_mtx_wrapper(_yr_ang, fill_ry_mtx)
        self.vertices = self.vertices @ rotation.T


def rotation_mtx_wrapper(angle, fill_function):
    a_sin = np.sin(angle)
    a_cos = np.cos(angle)
    rotation = np.identity(4)

    return fill_function(a_sin, a_cos, rotation)


def fill_rx_mtx(a_sin, a_cos, rotation):
    rotation[1][1] = a_cos
    rotation[1][2] = -a_sin
    rotation[2][1] = a_sin
    rotation[2][2] = a_cos

    return rotation


def fill_ry_mtx(a_sin, a_cos, rotation):
    rotation[0][0] = a_cos
    rotation[0][2] = a_sin
    rotation[2][0] = -a_sin
    rotation[2][2] = a_cos

    return rotation


def fill_rz_mtx(a_sin, a_cos, rotation):
    rotation[0][0] = a_cos
    rotation[0][1] = -a_sin
    rotation[1][0] = a_sin
    rotation[1][1] = a_cos

    return rotation


def scale_and_translate_mtx(t, scale):
    rotation = np.identity(4)
    rotation[0][0] = scale[0]
    rotation[1][1] = scale[1]
    rotation[2][2] = scale[2]

    rotation[:-1, -1] = t

    return rotation


def fill_intrinsic_mtx(img_res, f_l, sens_dim):
    w_img, h_img = img_res
    w_ccd, h_ccd = sens_dim

    fx = f_l * w_img / float(w_ccd)
    fy = f_l * h_img / float(h_ccd)

    px = w_img / 2.0
    py = h_img / 2.0

    k = np.array([[fx, 0, px, 0],
                  [0, fy, py, 0],
                  [0, 0,  1,  0]])

    return k


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

    return c_areas, b_rects


class PinholeCam(object):
    def __init__(self, r_x, cam_h, img_res, sens_dim, f_l):
        self.r_x = r_x   # Now in degrees
        self.cam_h = cam_h
        self.img_res = img_res
        self.sens_dim = sens_dim
        self.f_l = f_l

    def pixels_to_distance(self, n):
        n = self.img_res[1] - n
        pxlmm = self.sens_dim[1] / self.img_res[1]  # print 'pxlmm ', pxlmm
        h_px = self.img_res[1] / 2 - n
        h_mm = h_px * pxlmm  # print 'hmm ', h_mm
        bo = np.arctan(h_mm / self.f_l)  # print 'bo ', np.rad2deg(bo)
        deg = np.deg2rad(abs(self.r_x)) + bo
        tan = np.tan(deg) if deg >= 0 else -1.
        d = (abs(self.cam_h) / tan)

        return d