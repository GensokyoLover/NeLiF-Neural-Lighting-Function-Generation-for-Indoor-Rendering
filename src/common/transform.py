import numpy as np

from . import Serializable, LazyEditable
from logger import GlobalLogger
from nmath import matrix
from nmath import vector 


class Transform(Serializable, LazyEditable):
    def __init__(self, transform):
        self._transform = transform
        self._inv_transform = None
        self.set_dirty()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._update_transform(transform)

    def _update_transform(self, transform):
        self._transform = transform
        self.set_dirty()

    @property
    def inverse(self):
        if self.is_dirty():
            self._inv_transform = matrix.inverse_matrix(self.transform)
            self.clear_dirty()
        return self._inv_transform
    
    def serialize(self, **kwargs):
        return self._transform
    

class TRSTransform(Transform):
    def __init__(self, transform=None, translation=None, scale=None, rotation_angle=None, rotation_axis=None):
        assert (transform is None) or all(x is None for x in [translation, scale, rotation_angle, rotation_axis]), \
            'Please use one of "transform" and "TRS" to define a TRSTransform, not both'

        super().__init__(transform)
        self._translation = translation if translation is not None else np.zeros(3, np.float32)
        self._scale = scale if scale is not None else np.ones(3, np.float32)
        self._rotation_angle = rotation_angle if rotation_angle is not None else 0.0
        self._rotation_axis = rotation_axis if rotation_axis is not None else np.zeros(3, np.float32)
        self._rotation_mat = matrix.create_identity() # make sure _rotation matrix exists

        if self._transform is not None:
            self._sync_TRS()
        else:
            self._update_all()
        
    def _sync_TRS(self):
        trs = matrix.decompose_TRS(self._transform)
        self._translation = trs.translation
        self._scale = trs.scale
        self._rotation_angle = trs.rotation_angle
        self._rotation_axis = trs.rotation_axis
        self._update_translation(self._translation)
        self._update_scale(self._scale)
        self._update_rotation(self._rotation_angle, self._rotation_axis)

    def _update_all(self):
        self._update_translation(self._translation)
        self._update_scale(self._scale)
        self._update_rotation(self._rotation_angle, self._rotation_axis)
        self._update_transform_matrix()

    def _update_transform_matrix(self):
        self._transform = matrix.multiply(matrix.multiply(self._rotation_mat, self._scale_mat), self._translation_mat)
        self.set_dirty()

    def _update_transform(self, transform):
        super()._update_transform(transform)
        self._sync_TRS()
   
    def _update_translation(self, translation):
        self._translation = translation
        self._translation_mat = matrix.create_from_translation(self._translation)

    def _update_rotation(self, rotation_angle, rotation_axis):
        self._rotation_angle = rotation_angle
        self._rotation_axis = rotation_axis
        if vector.length(self._rotation_axis) > 1e-3:
            self._rotation_mat = matrix.create_from_axis_rotation(rotation_axis, rotation_angle)
    
    def _update_scale(self, scale):
        self._scale = scale
        self._scale_mat = matrix.create_from_scale(self._scale) 

    @property
    def rotation_axis(self):
        return self._rotation_axis

    @rotation_axis.setter
    def rotation_axis(self, rotation_axis):
        self._update_rotation(self._rotation_angle, rotation_axis)
        self._update_transform_matrix()

    @property
    def rotation_angle(self):
        return np.degrees(self._rotation_angle)

    @rotation_angle.setter
    def rotation_angle(self, rotation_angle):
        self._update_rotation(np.radians(rotation_angle, self._rotation_axis))
        self._update_transform_matrix()

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        self._update_translation(translation)
        self._update_transform_matrix()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._update_scale(scale)
        self._update_transform_matrix() 
