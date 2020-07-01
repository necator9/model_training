# Created by Ivan Matveev at 05.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Functions and classes to manipulate 3d objects in space (including projection to image plane) are defined in this file

import numpy as np


class TranslateMtx(object):
    def __init__(self, key, vertices):
        """
        Transformation matrix class defining object's coordinates in 3d space
        :param key: present for interface compatibility, not used
        :param vertices: np.array([N, 4)] of the object vertices coordinates in homogeneous form
        """
        self.mtx = np.identity(4)
        self.vertices = vertices

    def build(self, coords):
        """
        Build a 4 x 4 translation matrix
        :param coords: target object X, Y, Z coordinates (not offset), e.g. tuple(x, y, z)
        :return: filled translation matrix which is ready for linear multiplication
        """
        t = np.asarray(coords) - self.find_principal_point()[:-1]  # Find offset between previous and target coordinates
        self.mtx[:-1, -1] = t  # Fill the matrix
        return self.mtx

    def find_principal_point(self):
        """
        The object principal point is defined as middle coordinate along X axis, the lowest coordinate along Y axis and
        the lowest (frontal) coordinate along Z axis
        :return: Object principal point coordinates in homogeneous form
        """
        return np.array([(self.vertices[:, 0].min() + self.vertices[:, 0].max()) / 2,
                         self.vertices[:, 1].min(), self.vertices[:, 2].min(), 1])


class ScaleMtx(object):
    def __init__(self, key, vertices):
        self.vertices = vertices
        self.mtx = np.identity(4)
        self.shape = self.measure_act_shape()
        self.shape_cursor = None

    def build(self, args):
        self.shape = self.measure_act_shape()
        prop, req_dims = args
        scale_f = np.asarray(req_dims) / self.shape
        if prop:
            scale_f[:] = scale_f.max()
        else:
            scale_f[scale_f == 0] = 1

        self.shape_cursor = self.shape * scale_f

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


class Handler3D(object):
    def __init__(self, vert, operations, k=None):
        """
        Manage 3D and projective transformations of 3D object
        :param vert: np.array([N, 4)] of the object vertices coordinates in homogeneous form
        :param operations: list of str(keys) for the self.mtx_seq defining sequence of transformations
        :param k: intrinsic camera parameters as list [np.array([img_width, img_height]), float(focal_length),
        np.array([sensor width, sensor height])]
        """
        self.vertices = vert  # Input vertices
        self.transformed_vertices = np.copy(self.vertices)  # Resulting vertices
        self.img_points = np.zeros((self.vertices.shape[0], 2), dtype='int32')  # Resulting image plane (if intrinsic
        # matrix is given)
        self.operations = operations  # Sequence of transformations, operations can be repeated multiple times
        # Mapping the extrinsic transformation matrices to keys
        self.mtx_seq = {'rx': RotationMtx, 'ry': RotationMtx, 'rz': RotationMtx, 's': ScaleMtx, 't': TranslateMtx}
        # Create instances of the transformation matrices
        self.mtx_seq = [self.mtx_seq[op](op, self.transformed_vertices) for op in self.operations]
        # Create instance of intrinsic projective matrix
        self.k_obj = IntrinsicMtx(k, self.transformed_vertices, self.img_points) if k is not None else None

    def transform(self,  *args):
        """
        Multiply input vertices and transformation matrices that result in results 3D transformation (array of
        self.transformed_vertices) and object image plane coordinates
        :param args: list of arguments for transformations, where each item corresponds to passed to constructor
         operation. See allowed arguments in particular matrix class method.
        """
        # Do not modify original vertices
        self.transformed_vertices[:] = np.copy(self.vertices)
        # Perform linear matrix multiplication
        for mtx, arg in zip(self.mtx_seq, args):
            self.transformed_vertices[:] = self.transformed_vertices @ mtx.build(arg).T
        # Project vertices to image plane if intrinsic matrix is given
        if self.k_obj is not None:
            self.k_obj.project_to_image()
