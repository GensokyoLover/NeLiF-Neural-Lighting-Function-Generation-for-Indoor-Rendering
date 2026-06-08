import pyrr.matrix44 as m44
from pyrr import Quaternion
import numpy as np
from collections import namedtuple

TRS = namedtuple('TRS', ['scale', 'rotation', 'translation', 'rotation_angle', 'rotation_axis'])

def multiply(mat0, mat1):
    return m44.multiply(mat0, mat1)

def create_identity(size=4, dtype=np.float32):
    return np.eye(size, size).astype(dtype)

def create_from_translation(translation, dtype=np.float32):
    return m44.create_from_translation(translation).astype(dtype)

def create_from_axis_rotation(axis, radians, dtype=np.float32):
    return m44.create_from_axis_rotation(axis, radians).astype(dtype)

def create_from_scale(scale, dtype=np.float32):
    return m44.create_from_scale(scale).astype(dtype)

def create_perspective(fovy, aspect, near, far, dtype=np.float32):
    return m44.create_perspective_projection(fovy, aspect, near, far, dtype=dtype)

def create_view_matrix(origin, lookat, up, dtype=np.float32):
    return m44.create_look_at(origin, lookat, up, dtype=dtype)

def inverse_matrix(mat):
    return m44.inverse(mat)

def decompose_TRS(mat, dtype=np.float32):
    scale, rotation, translation = m44.decompose(mat)
    rotation = rotation.astype(dtype)
    rotation = Quaternion(rotation)
    return TRS(scale.astype(dtype), rotation, translation.astype(dtype), float(rotation.angle), np.array(rotation.axis.xyz))

