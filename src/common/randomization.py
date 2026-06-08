import abc
from dataclasses import dataclass
from functools import cached_property

import numpy
import numpy as np
import math
import torch
import open3d as o3d

@dataclass(frozen=True)
class BaseRandomConfigs:
    randomizable: bool = False
    avoid_collision: bool = False

    @cached_property
    def num_parameters(self):
        # Check valid variables & Stats its dimension
        num_params = 0
        if self.randomizable:
            for v in self.__dict__.values():
                if isinstance(v, np.ndarray):
                    range_bound = v[1] - v[0]
                    zero_dims = np.isclose(range_bound, 0)
                    num_params += len(range_bound) - sum(zero_dims)
        return num_params

class Randomizable(metaclass=abc.ABCMeta):
    def random(self, random_samples=None):
        assert isinstance(self._random_configs, BaseRandomConfigs)
        if random_samples:
            assert len(random_samples) == self.num_parameters
        else:
            random_samples = np.random.rand(self.num_parameters).astype(np.float32)
        if self._random_configs.randomizable:
            # randomize all variables declared in random_configs
            current_sample_idx = 0
            for k, v in self._random_configs.__dict__.items():
                if isinstance(v, np.ndarray):
                    assert len(v) == 2
                    randomized_v = v[0].copy()
                    range_bound = v[1] - v[0]
                    zero_dims = np.isclose(range_bound, 0)
                    for idx, skip in enumerate(zero_dims):
                        if not skip:
                            randomized_v[idx] = random_samples[current_sample_idx] * (v[1][idx] - v[0][idx]) + v[0][idx]
                            current_sample_idx += 1
                    if len(randomized_v) == 1:
                        randomized_v = float(randomized_v)
                    try:
                        setattr(self, k, randomized_v)
                    except Exception as e:
                        print('[Error] Cannot set ({})\'s attribute ({}) to ({}).'.format(self.name, k, randomized_v))
                        raise e
            self._random() # post random
    
    # def _interp(self, random_samples: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray):
    #     range_bounds = max_bounds - min_bounds
    #     assert np.all(range_bounds > 0)
    #     assert np.all(random_samples >= 0 & random_samples <= 1)
    #     return random_samples * range_bounds + min_bounds
    
    @property
    def randomizable(self):
        return self._random_configs.randomizable
    
    @property
    def num_parameters(self):
        return self._random_configs.num_parameters
  
    # @abc.math.abstractmethod
    # def as_dict(self): # serialize
    #     raise NotImplementedError
    
    # @abc.math.abstractmethod
    # def _get_bounds(self):
    #     raise NotImplementedError
    
    # @abc.math.abstractmethod
    # def get_vars(self): # deprecated
    #     raise NotImplementedError

    @abc.math.abstractmethod
    def _random(self):
        raise NotImplementedError


def WrapEqualAreaSquare(uv):
    if (uv[0] < 0):
        uv[0] = -uv[0]
        uv[1] = 1 - uv[1]
    elif (uv[0] > 1):
        uv[0] = 2 - uv[0]
        uv[1] = 1 - uv[1]
    if (uv[1] < 0) :
        uv[0] = 1 - uv[0]
        uv[1] = -uv[1]
    elif (uv[1] > 1):
        uv[0] = 1 - uv[0]
        uv[1] = 2 - uv[1]
    return uv



def EqualAreaSquareToSphere(p):
    u = 2 * p[0] - 1
    v = 2 * p[1] - 1
    up = math.fabs(u)
    vp = math.abs(v)

    signedDistance = 1 - (up + vp)
    d = math.fabs(signedDistance)
    r = 1 - d
    if r == 0:
        phi = 1 * math.pi / 4
    else:
        phi =( (vp - up) / r + 1) * math.pi / 4
    z = math.fabs(1 - r*r)
    if (signedDistance < 0):
        z = -z;
    cosPhi = math.fabs(math.cos(phi))
    if (u<0):
        cosPhi = - cosPhi
    sinPhi = math.fabs(math.sin(phi))
    if (v<0):
        sinPhi = - sinPhi
    return (cosPhi * r * math.sqrt(2 - r * r), sinPhi * r * math.sqrt(2 - r * r),z)


def spherical_texture(radius,width,height):
    spherical_texture = numpy.zeros((width,height,3))
    for i in range(width):
        for j in range(height):
            uv = [i,j]
            uv = WrapEqualAreaSquare(uv)
            uv = list(uv)
            dir = EqualAreaSquareToSphere(uv)
            dir = np.array(list(dir))
            spherical_texture[i,j] = dir
    return spherical_texture

def view_point_cloud(position):
    if len(list(position.shape)) == 3:
        position = position[0]
    if position.shape[0] == 3:
        position_new_view = position.permute(1,0)
    else:
        position_new_view = position
    if isinstance(position_new_view,torch.Tensor):
        position_new_view = position_new_view.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    # 将 NumPy 数组中的点赋值给点云
    pcd.points = o3d.utility.Vector3dVector(position_new_view)
    # 保存点云到文件
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    # 如果需要可视化点云
    o3d.visualization.draw_geometries([pcd])