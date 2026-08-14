import random

import numpy
import torch
from .shadow_network import *
from .partition_pyramid import *
from networks.loss_functions import Loss

from einops import rearrange
from .unet import *

import gzip
import numpy as np
from torchvision.models.vision_transformer import VisionTransformer, ConvStemConfig, _log_api_usage_once, \
    Conv2dNormActivation, OrderedDict, Encoder, EncoderBlock, MLPBlock
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
import pickle
import torch
import torch.nn as nn
import time 

def fc_layer(in_features, out_features):
    net = nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True)
    )
    return net
def get_lightformer_input(localLightData, localData,inverse=True):
    z_f = localLightData['shadow']['pixel_emitter_distance'] 
    z = localLightData['shadow']['occluder_emitter_distance'] 

    position_mask = localData["position_mask"]
    z_f[position_mask[..., 0]] = 3
    z[position_mask[..., 0]] = -3
    normal = localData["gbuffer"]["normal"]

    toLight = localData["toLight"][..., :3]
    toView = localData["gbuffer"]["view_dir"]
    c_c = torch.sum(localData["gbuffer"]['normal'] * localData["gbuffer"]['view_dir'], axis=-1)[
        ..., None]  # diffuse # TODO: Use this feature?
    depth = localData["gbuffer"]["depth"]
    #depth =normalize_depth_with_mask(depth,position_mask[...,0:1])
    localData["lightformer_input0"] = z_f - z
    localData["lightformer_input1"] = z / z_f
    localData["lightformer_input2"] = depth[:1,...]
    localData["lightformer_input3"] =  c_c[:1,...]
    localData["normal_dot_view"] = c_c
    shadow_tanh = localData["tanh_shadow"]
    localData["hard_shadow"] = torch.where(z_f - z > (0.0009 + localData["tanh_shadow"] * 0.003), 0, 1)
    if inverse:
        localData["hard_shadow"] = torch.where(z_f - z > (0.0009 + localData["tanh_shadow"] * 0.003), 1, 0)
    depth = 1 - 1 / (1 + depth)
    depth[position_mask[...,0]] = -1
    return torch.cat([z_f - z, z / z_f, depth, c_c[:,...],localData["hard_shadow"]], dim=-1)
class Timer:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_ts = time.perf_counter()
    
    def end(self):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event)  # ms
        else:
            return (time.perf_counter() - self.start_ts) * 1000.0 

def wrap_sample(img, uv, wrap_u=False, wrap_v=False):
    """
    img: (B, C, H, W)
    uv: (B, N, 2)
    wrap_u: 是否对 u wrap (表示 φ)
    wrap_v: 是否对 v wrap
    """

    B, C, H, W = img.shape
    u = uv[..., 0] * W - 0.5
    v = uv[..., 1] * H - 0.5

    # bilinear 4 neighbors
    u0 = torch.floor(u)
    v0 = torch.floor(v)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = u - u0
    fv = v - v0

    # wrap or clamp per dimension
    def idx(val, size, do_wrap):
        if do_wrap:
            return (val.long() % size)
        else:
            return val.long().clamp(0, size - 1)

    u0i = idx(u0, W, wrap_u)
    u1i = idx(u1, W, wrap_u)
    v0i = idx(v0, H, wrap_v)
    v1i = idx(v1, H, wrap_v)

    # gather
    def gather(uu, vv):
        return img[..., vv, uu]  # (B,N,C)

    c00 = gather(u0i, v0i)
    c10 = gather(u1i, v0i)
    c01 = gather(u0i, v1i)
    c11 = gather(u1i, v1i)

    # bilinear weight
    c0 = c00 * (1 - fu)[...,None] + c10 * fu[...,None]
    c1 = c01 * (1 - fu)[...,None] + c11 * fu[...,None]
    out = c0 * (1 - fv)[...,None] + c1 * fv[...,None]
    return out

def octahedral_project(xyz):
    """
    xyz: (...,3)，必须是已归一化的方向向量
    return: uv in [-1,1]
    """
    x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
    abs_sum = x.abs() + y.abs() + z.abs() + 1e-8

    # 初步投影
    u = x / abs_sum
    v = y / abs_sum

    # 下半球折叠
    mask = (z < 0)
    u2 = (1 - v.abs()) * u.sign()
    v2 = (1 - u.abs()) * v.sign()

    u = torch.where(mask, u2, u)
    v = torch.where(mask, v2, v)

    # 归一化到 [0,1]
    uv = torch.stack([u, v], dim=-1)
    return (uv + 1) * 0.5
def normalize(x):
    return x / torch.sqrt((x * x).sum(dim=-1)).unsqueeze(dim=-1)


def find_closest_intersection(positions, directions, radius=1):
    # 取批量射线位置和方向

    # 二次方程的系数 A, B, C，计算每条射线的系数
    A = torch.sum(directions ** 2, dim=-1)  # B, N
    B_coeff = 2 * torch.sum(positions * directions, dim=-1)  # B, N
    C = torch.sum(positions ** 2, dim=-1) - radius ** 2  # B, N

    # 计算判别式
    discriminant = B_coeff ** 2 - 4 * A * C  # B, N

    # 如果判别式小于0，表示没有交点，返回inf
    no_intersection = discriminant < 0

    # 计算t值：求解二次方程
    t1 = (-B_coeff + torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * A)  # B, N
    t2 = (-B_coeff - torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * A)  # B, N

    t_values = torch.stack([t1, t2], dim=-1)  # B, N, 2
    t_min = torch.min(t_values, dim=-1).values  # B, N
    # 对于没有交点的射线，返回inf
    t_min[no_intersection] = float('inf')
    # 计算交点
    intersections = positions + t_min.unsqueeze(-1) * directions  # B, N, 2
    return intersections


class NeuralCubemapSampler(nn.Module):
    """
    faces: [B, 6, R, R, C]  # R=face resolution, usually 4
    direction: [B, H, W, 3] (unit vector)
    return: [B, H, W, C]
    """
    def __init__(self, face_res=4):
        super().__init__()
        self.res = face_res

        # normals
        self.N = torch.tensor([
            [ 1, 0, 0],   # +X
            [-1, 0, 0],   # -X
            [ 0, 1, 0],   # +Y
            [ 0,-1, 0],   # -Y
            [ 0, 0, 1],   # +Z
            [ 0, 0,-1],   # -Z
        ], dtype=torch.float32)

        # U axis (right)
        self.V = torch.tensor([
            [0, 1, 0],    # +X
            [0, 1, 0],    # -X
            [0, 0,-1],    # +Y
            [0, 0, 1],    # -Y
            [0, 1, 0],    # +Z
            [0, 1, 0],    # -Z
        ], dtype=torch.float32)

        # V axis (up)
        self.U =- torch.tensor([
            [ 0, 0, 1],   # +X
            [ 0, 0,-1],   # -X
            [-1, 0, 0],   # +Y
            [-1, 0, 0],   # -Y
            [-1, 0, 0],   # +Z
            [ 1, 0, 0],   # -Z
        ], dtype=torch.float32)


    # ----------------------------------------------------
    # 方向 → face, u, v
    # ----------------------------------------------------
    def direction_to_face_uv(self, d):
        """
        d: [B,H,W,3]
        return:
            face: [B,H,W]
            u,v ∈ [-1,1]
        """

        dx, dy, dz = d[...,0], d[...,1], d[...,2]

        # 面选择 (点积最大)
        scores = torch.stack([dx, -dx, dy, -dy, dz, -dz], dim=-1)
        face = scores.argmax(dim=-1)            # [B,H,W]

        # 准备 N,U,V（按 face 索引取）
        N = self.N.to(d.device)[face]           # [B,H,W,3]
        U = self.U.to(d.device)[face]
        V = self.V.to(d.device)[face]

        # 投影到 face 平面
        w = (d * N).sum(-1, keepdim=True)       # d·N
        proj = d / (w + 1e-8)                   # 射到平面

        u = (proj * U).sum(-1)                  # [-1,1]
        v = (proj * V).sum(-1)                  # [-1,1]

        return face, u, v


    # ----------------------------------------------------
    # face,u,v → nearest texel index
    # ----------------------------------------------------
    def uv_to_texel(self, face, u, v):
        """
        u,v ∈ [-1,1], convert to pixel
        """
        res = self.res

        px = (u + 1) * 0.5 * (res - 1)  # float in [0,res-1]
        py = (v + 1) * 0.5 * (res - 1)

        xi = torch.clamp(px.round().long(), 0, res - 1)
        yi = torch.clamp(py.round().long(), 0, res - 1)

        return face, xi, yi


    # ----------------------------------------------------
    # bilinear 权重
    # ----------------------------------------------------
    def bilinear_weights(self, u, v):
        res = self.res

        px = (u + 1) * 0.5 * (res - 1)
        py = (v + 1) * 0.5 * (res - 1)

        x0 = torch.floor(px).long()
        y0 = torch.floor(py).long()
        x1 = x0 + 1
        y1 = y0 + 1

        wx = px - x0.float()
        wy = py - y0.float()

        return x0, x1, y0, y1, wx, wy


    # ----------------------------------------------------
    # 主函数：neural cubemap sample
    # ----------------------------------------------------
    def forward(self, faces, direction):
        """
        faces: [B,6,R,R,C]
        direction: [B,H,W,3]
        """

        B, H, W, _ = direction.shape
        device = direction.device
        res = self.res
        C = faces.shape[-1]

        # -------- Step 1: 中心方向映射 --------
        face_c, ua, va = self.direction_to_face_uv(direction)
        u_c =ua
        v_c = va
        x0, x1, y0, y1, wx, wy = self.bilinear_weights(u_c, v_c)

        # 对于 0/1 坐标取 texel center 的 u,v
        def tex_uv(ix, iy):
            # [0,res-1] → [-1,1]
            u = ix.float() / (res - 1) * 2 - 1
            v = iy.float() / (res - 1) * 2 - 1
            return u, v

        # 四个点的局部 uv
        u00, v00 = tex_uv(x0, y0)
        u10, v10 = tex_uv(x1, y0)
        u01, v01 = tex_uv(x0, y1)
        u11, v11 = tex_uv(x1, y1)

        # -------- Step 2: 四邻点构造方向 --------
        def uv_to_dir(face, u, v):
            # 根据 face 的 N, U, V 求方向
            N = self.N.to(device)[face]  # [B,H,W,3]
            U = self.U.to(device)[face]
            V = self.V.to(device)[face]

            u = u.unsqueeze(-1)
            v = v.unsqueeze(-1)

            p = N + u * U + v * V         # 未归一化
            d = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
            return d

        d00 = uv_to_dir(face_c, u00, v00)
        d10 = uv_to_dir(face_c, u10, v10)
        d01 = uv_to_dir(face_c, u01, v01)
        d11 = uv_to_dir(face_c, u11, v11)

        # -------- Step 3: 四个方向重新找真正落在哪个 face --------
        def dir_sample(d):
            face, u, v = self.direction_to_face_uv(d)
            face, xi, yi = self.uv_to_texel(face, u, v)
            return face, xi, yi

        f00, x00, y00 = dir_sample(d00)
        f10, x10, y10 = dir_sample(d10)
        f01, x01, y01 = dir_sample(d01)
        f11, x11, y11 = dir_sample(d11)

        # -------- Step 4: gather texels --------
        def gather(fi, xi, yi):
            # faces: [B,6,res,res,C]
            out = faces[torch.arange(B)[:,None,None], fi, yi, xi]
            return out  # [B,H,W,C]

        c00 = gather(f00, x00, y00)
        c10 = gather(f10, x10, y10)
        c01 = gather(f01, x01, y01)
        c11 = gather(f11, x11, y11)

        # -------- Step 5: bilinear combine --------
        wx2 = wx.unsqueeze(-1)
        wy2 = wy.unsqueeze(-1)
        w00 = (1-wx2)*(1-wy2)
        w10 = wx2*(1-wy2)
        w01 = (1-wx2)*wy2
        w11 = wx2*wy2

        out = w00*c00 + w10*c10 + w01*c01 + w11*c11
        return out,d00

def rotate(a, b, c, v):
    return torch.cat([torch.sum((a * v), dim=-1).unsqueeze(-1),
                      torch.sum((b * v), dim=-1).unsqueeze(-1),
                      torch.sum((c * v), dim=-1).unsqueeze(-1)], dim=-1)


def compute_angle(u, v):
    # 计算点积
    dot_product = torch.sum(u * v, dim=-1)

    # 计算余弦值
    cos_theta = dot_product / (torch.norm(u, dim=-1) * torch.norm(v, dim=-1))

    # 通过反余弦计算夹角（单位：弧度）
    angle = torch.acos(torch.clamp(cos_theta, min=-1.0, max=1.0)).unsqueeze(-1)

    return angle


def get_lod_tracing(lposition, specular_ray, normal):
    lenth = torch.norm(lposition, keepdim=True, dim=-1)
    gg = torch.asin(1 / lenth)
    light_range = torch.cat([torch.pi / 2 - gg, torch.pi / 2 + gg], dim=-1)
    z = normalize(-lposition)
    normalize_ray = normalize(specular_ray)
    up = normalize(torch.cross(z, specular_ray, dim=-1))
    right = normalize(torch.cross(up, z, dim=-1))
    rotation_matrix = torch.cat([right.unsqueeze(-2), z.unsqueeze(-2), up.unsqueeze(-2)], dim=-2)
    local_ray = (rotation_matrix @ specular_ray.unsqueeze(-1)).squeeze(-1)
    relative_angular = torch.atan(local_ray[..., 1:2] / (local_ray[..., 0:1]))
    lod_angular = torch.Tensor([0 / 180, 15 / 180, 45 / 180, 90 / 180]) * torch.pi
    final_cast_ray = []
    local_lposition = (rotation_matrix @ lposition.unsqueeze(-1)).squeeze(-1)[..., :2]
    local_normal = (rotation_matrix @ normal.unsqueeze(-1)).squeeze(-1)
    result_mask = []
    # result_hit = []
    result_uv = []
    spherical_uv = []
    angular_range_list = []
    space_range_list = []
    local_point_left_list = []
    standard_dir_list = []
    local_point_right_list = []
    shuaijian_list = []
    # pyexr.write("../testData/relative_angular.exr",relative_angular[0].cpu().numpy())
    extend_angular = [15 / 180 * torch.pi, 30 / 180 * torch.pi, 45 / 180 * torch.pi, 0.01 / 180 * torch.pi]
    for i in range(4):

        ray_range = torch.cat([relative_angular - lod_angular[i], relative_angular + lod_angular[i]], dim=-1)
        left_min = torch.max(ray_range[..., :1], light_range[..., :1])
        right_max = torch.min(ray_range[..., 1:2], light_range[..., 1:2])
        left_cha = light_range[..., :1] - ray_range[..., 1:2]
        left_cha = torch.where(left_cha < 0, 0, left_cha)
        shuaijian = 1 - left_cha / extend_angular[i]
        shuaijian = torch.where(shuaijian < 0, 0, shuaijian)

        final_range = torch.cat([left_min, right_max], dim=-1)

        ray_mask = left_min <= (right_max + 1e-5)

        ray_left = torch.cat([torch.cos(left_min), torch.sin(left_min)], dim=-1)
        ray_right = torch.cat([torch.cos(right_max), torch.sin(right_max)], dim=-1)
        ray_mid = torch.cat([torch.cos((left_min + right_max) / 2), torch.sin((left_min + right_max) / 2)], dim=-1)
        point_left = find_closest_intersection(local_lposition, ray_left)
        left_angular = torch.atan(point_left[..., 1:2] / (point_left[..., 0:1] + 1e-5))
        left_angular = torch.where(left_min == light_range[..., :1], -gg, left_angular)
        point_left = torch.cos(left_angular) * right + torch.sin(left_angular) * z

        local_point_left = torch.cat([torch.cos(left_angular), torch.sin(left_angular)], dim=-1) * 1
        local_to_left = local_point_left - local_lposition
        point_left = point_left * 1
        to_left = normalize(point_left - lposition)
        point_right = find_closest_intersection(local_lposition, ray_right)

        right_angular = torch.atan(point_right[..., 1:2] / (point_right[..., 0:1] + 1e-5))
        right_angular = torch.where(right_max == light_range[..., 1:2], -torch.pi + gg, right_angular)
        right_angular = torch.where(right_angular > 0, right_angular - torch.pi, right_angular)
        point_right = torch.cos(right_angular) * right + torch.sin(right_angular) * z
        local_point_right = torch.cat([torch.cos(right_angular), torch.sin(right_angular)], dim=-1) * 1
        local_to_right = local_point_right - local_lposition

        point_right = point_right * 1
        to_right = normalize(point_right - lposition)
        angular_range = torch.abs(right_angular - left_angular)

        point_mid = find_closest_intersection(local_lposition, ray_mid)
        new_lenth = torch.sqrt(lenth * lenth - 1 * 1)
        ##pyexr.write("../testData/lenth{}.exr".format(i),new_lenth[0].cpu().numpy())
        standard_left_point_x = new_lenth * torch.cos(light_range[..., :1])
        standard_left_point_y = new_lenth * torch.sin(light_range[..., :1]) - lenth
        standard_point = standard_left_point_x * right + standard_left_point_y * z

        ##pyexr.write("../testData/standard_point{}.exr".format(i), standard_point[0].cpu().numpy())
        true_point = point_mid[..., :1] * right + point_mid[..., 1:2] * z
        true_point = torch.where(~ray_mask.repeat(1, 1, 1, 3), standard_point, true_point)
        to_standard_dir = normalize(true_point - lposition)
        standard_dir_list.append(to_standard_dir)
        normalize_pt = normalize(true_point)
        uv = EqualAreaSphereToSquare(normalize(normalize_pt))
        reflect_forward, reflect_up, reflect_right = get_forward_up_right_tensor(-normalize_pt)
        reflect_up = -reflect_up
        uvw = rotate(reflect_right, reflect_up, reflect_forward, to_standard_dir)
        standard_dir = normalize(uvw)
        another_uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
        angular_range_list.append(angular_range)
        space_range = compute_angle(to_left, to_right)
        space_range_list.append(space_range)
        result_uv.append(uv)
        spherical_uv.append(another_uv)
        local_point_left_list.append(local_to_left)
        local_point_right_list.append(local_to_right)
        if i == 3:
            ray_mask = (~torch.isnan(uv)[..., :1]) & (left_min <= (right_max + 1e-5))
        else:
            ray_mask = (~torch.isnan(uv)[..., :1])
        ray_mask = (~torch.isnan(another_uv)[..., :1]) & ray_mask
        shuaijian[~ray_mask] = 0
        # pyexr.write("../testData/shuaijian{}.exr".format(i), (shuaijian[0]).cpu().numpy())
        # pyexr.write("../testData/raymask{}.exr".format(i), ray_mask[0].cpu().numpy())
        # pyexr.write("../testData/uv{}.exr".format(i), uv[0].cpu().numpy())
        # pyexr.write("../testData/another_uv{}.exr".format(i), another_uv[0].cpu().numpy())
        result_mask.append(ray_mask)

        shuaijian_list.append(shuaijian)
    result_mask = torch.cat(result_mask, dim=0)
    # result_hit = []
    result_uv = torch.cat(result_uv, dim=0)
    spherical_uv = torch.cat(spherical_uv, dim=0)
    angular_range_list = torch.cat(angular_range_list, dim=0)
    space_range_list = torch.cat(space_range_list, dim=0)
    local_point_left_list = torch.cat(local_point_left_list, dim=0)
    shuaijian_list = torch.cat(shuaijian_list, dim=0)
    local_point_right_list = torch.cat(local_point_right_list, dim=0)
    standard_dir_list = torch.cat(standard_dir_list, dim=0)
    data = {}
    data["angular_uv"] = result_uv
    data["space_uv"] = spherical_uv
    data["angular_range"] = angular_range_list
    data["space_range"] = space_range_list
    data["local_point_left"] = local_point_left_list
    data["local_point_right"] = local_point_right_list
    data["true_point"] = local_point_right_list
    data["local_normal"] = local_normal
    data["local_ray"] = local_ray
    data["ray_mask"] = result_mask
    data["shuaijian"] = shuaijian_list
    data["standard_dir"] = standard_dir_list
    return data


def get_lod_tracing_arbitrary(lposition, specular_ray, normal, roughness, table, light_angular_size, light_space_size):
    data = {}
    _, W1, H1, _ = roughness.shape
    lenth = torch.norm(lposition, keepdim=True, dim=-1)
    gg = torch.asin(1 / lenth)
    light_range = torch.cat([torch.pi / 2 - gg, torch.pi / 2 + gg], dim=-1)
    z = normalize(-lposition)
    normalize_ray = normalize(specular_ray)
    up = normalize(torch.cross(z, specular_ray, dim=-1))
    right = normalize(torch.cross(up, z, dim=-1))
    rotation_matrix = torch.cat([right.unsqueeze(-2), z.unsqueeze(-2), up.unsqueeze(-2)], dim=-2)
    local_ray = (rotation_matrix @ specular_ray.unsqueeze(-1)).squeeze(-1)
    relative_angular = torch.atan(local_ray[..., 1:2] / (local_ray[..., 0:1]))
    data["x"] = torch.sin(relative_angular)
    data["y"] = (roughness * 2 - 1)
    data["z"] = ((lenth - 0.4) / 6) * 2 - 1
    roughness = (roughness * 1000).long()
    roughness = torch.clamp(roughness, min=0, max=999)
    # table = torch.Tensor(table).cuda()
    lod_angular = table[roughness.reshape(-1)].reshape(1, W1, H1, 1) * 2  # halfvec to reflectdir double 2

    ####print_image_exr(lod_angular,"lod_angular")
    ####print_image_exr(roughness,"roughness")
    final_cast_ray = []
    local_lposition = (rotation_matrix @ lposition.unsqueeze(-1)).squeeze(-1)[..., :2]
    local_normal = (rotation_matrix @ normal.unsqueeze(-1)).squeeze(-1)
    result_mask = []
    # result_hit = []
    result_uv = []
    spherical_uv = []
    angular_range_list = []
    space_range_list = []
    local_point_left_list = []
    standard_dir_list = []
    local_point_right_list = []
    ###print_image_exr(lod_angular,"lod_angular")
    ray_range = torch.cat([relative_angular - lod_angular, relative_angular + lod_angular], dim=-1)
    ###print_image_exr(ray_range[...,:1], "ray_range0")
    ###print_image_exr(ray_range[...,1:2], "ray_range1")
    clamp_ray_range = torch.clamp(ray_range, min=0, max=torch.pi)
    ray_mid_after_clamp = (clamp_ray_range[..., :1] + clamp_ray_range[..., 1:2]) / 2

    left_min = torch.max(ray_range[..., :1], light_range[..., :1])
    right_max = torch.min(ray_range[..., 1:2], light_range[..., 1:2])
    # compute the scale when cone is nearly outside the 4d representation
    scale = (lod_angular * 2 / (right_max - left_min + 1e-4)) * (lod_angular * 2 / (right_max - left_min + 1e-4))

    scale2, cone2, cone_bin = cone_union_percentage_radians(torch.clamp(gg * 1.3, min=0, max=torch.pi // 2),
                                                            torch.clamp(lod_angular * 1.3, min=0, max=torch.pi // 2),
                                                            torch.abs(ray_mid_after_clamp - torch.pi / 2))
    scale2 = torch.clamp(scale2, min=1)
    # ###print_image_exr(scale2,"scale2")
    # ###print_image_exr(cone2,"cone2")
    # ###print_image_exr(cone_bin,"cone_bin")
    # ###print_image_exr(gg,"gg")
    # ###print_image_exr(lod_angular,"lod_angular")
    # ###print_image_exr(torch.abs(ray_mid_after_clamp-torch.pi/2),"phi")

    final_range = torch.cat([left_min, right_max], dim=-1)

    ray_mask = left_min <= (right_max + 1e-6)

    ray_left = torch.cat([torch.cos(left_min), torch.sin(left_min)], dim=-1)
    ray_right = torch.cat([torch.cos(right_max), torch.sin(right_max)], dim=-1)
    ###print_image_exr(ray_mid_after_clamp, "ray_mid_after_clamp")
    ###print_image_exr((left_min + right_max) / 2, "ray_mid")

    ray_mid = torch.cat([torch.cos((left_min + right_max) / 2), torch.sin((left_min + right_max) / 2)], dim=-1)

    point_left = find_closest_intersection(local_lposition, ray_left)
    ####print_image_exr(ray_range[...,:1],"ray_range0")
    ####print_image_exr(ray_range[...,1:2],"ray_range1")
    ####print_image_exr(light_range[...,:1],"light_range0")
    ####print_image_exr(light_range[...,1:2],"light_range1")
    ####print_image_exr(left_min,"left_min")
    ####print_image_exr(right_max,"right_max")
    ####print_image_exr(relative_angular,"relative_angular")
    left_angular = torch.atan(point_left[..., 1:2] / (point_left[..., 0:1] + 1e-5))
    left_angular = torch.where(left_min == light_range[..., :1], -gg, left_angular)

    point_left = torch.cos(left_angular) * right + torch.sin(left_angular) * z
    ####print_image_exr(point_left, "point_left")
    local_point_left = torch.cat([torch.cos(left_angular), torch.sin(left_angular)], dim=-1) * 1
    local_to_left = local_point_left - local_lposition
    point_left = point_left * 1
    to_left = normalize(point_left - lposition)
    point_right = find_closest_intersection(local_lposition, ray_right)

    right_angular = torch.atan(point_right[..., 1:2] / (point_right[..., 0:1] + 1e-5))
    right_angular = torch.where(right_angular > 0, right_angular - torch.pi, right_angular)
    right_angular = torch.where(right_max == light_range[..., 1:2], - torch.pi + gg, right_angular)
    point_right = torch.cos(right_angular) * right + torch.sin(right_angular) * z
    angular_lenth = torch.abs(right_angular - left_angular) / torch.pi / 2
    space_lenth = torch.abs(right_max - left_min) / torch.pi
    ####print_image_exr(point_right,"point_right")
    local_point_right = torch.cat([torch.cos(right_angular), torch.sin(right_angular)], dim=-1) * 1
    local_to_right = local_point_right - local_lposition

    point_right = point_right * 1

    to_right = normalize(point_right - lposition)
    angular_range = torch.abs(right_angular - left_angular)

    point_mid = find_closest_intersection(local_lposition, ray_mid)
    ####print_image_exr(point_mid,"point_mid")
    new_lenth = torch.sqrt(lenth * lenth - 1 * 1)
    ##pyexr.write("../testData/lenth{}.exr".format(i),new_lenth[0].cpu().numpy())
    standard_left_point_x = new_lenth * torch.cos(light_range[..., :1])
    standard_left_point_y = new_lenth * torch.sin(light_range[..., :1]) - lenth
    standard_point = standard_left_point_x * right + standard_left_point_y * z

    ##pyexr.write("../testData/standard_point{}.exr".format(i), standard_point[0].cpu().numpy())
    true_point = point_mid[..., :1] * right + point_mid[..., 1:2] * z
    to_standard_dir = normalize(true_point - lposition)
    to_left_dir = normalize(point_left - lposition)
    to_right_dir = normalize(point_right - lposition)

    standard_dir_list.append(to_standard_dir)
    normalize_pt = normalize(true_point)
    uv = EqualAreaSphereToSquare(normalize(normalize_pt))
    left_uv = EqualAreaSphereToSquare(normalize(point_left))
    right_uv = EqualAreaSphereToSquare(normalize(point_right))
    reflect_forward1, reflect_up1, reflect_right1 = get_forward_up_right_tensor(-normalize_pt)
    reflect_up1 = -reflect_up1

    reflect_right = torch.cat([reflect_right1[..., :1], reflect_up1[..., :1], reflect_forward1[..., :1]], dim=-1)
    reflect_up = torch.cat([reflect_right1[..., 1:2], reflect_up1[..., 1:2], reflect_forward1[..., 1:2]], dim=-1)
    reflect_forward = torch.cat([reflect_right1[..., 2:3], reflect_up1[..., 2:3], reflect_forward1[..., 2:3]], dim=-1)

    # uvw = rotate(reflect_right, reflect_up, reflect_forward, to_standard_dir)
    # uvw_left = rotate(reflect_right, reflect_up, reflect_forward, to_left_dir)
    # uvw_right = rotate(reflect_right, reflect_up, reflect_forward, to_right_dir)

    uvw = rotate(reflect_right1, reflect_up1, reflect_forward1, to_standard_dir)
    uvw_left = rotate(reflect_right1, reflect_up1, reflect_forward1, to_left_dir)
    uvw_right = rotate(reflect_right1, reflect_up1, reflect_forward1, to_right_dir)
    standard_dir = normalize(uvw)
    left_dir = normalize(uvw_left)
    right_dir = normalize(uvw_right)
    another_uv = concentric_mapping_hemisphere_3D_to_2D(standard_dir)
    left_space_uv = concentric_mapping_hemisphere_3D_to_2D(left_dir)
    right_space_uv = concentric_mapping_hemisphere_3D_to_2D(right_dir)
    angular_range_list.append(angular_range)
    space_range = compute_angle(to_left, to_right)
    space_range_list.append(space_range)
    result_uv.append(uv)
    spherical_uv.append(another_uv)
    local_point_left_list.append(local_to_left)
    local_point_right_list.append(local_to_right)

    ray_mask = (~torch.isnan(uv)[..., :1]) & ray_mask
    ray_mask = (~torch.isnan(another_uv)[..., :1]) & ray_mask

    result_mask.append(ray_mask)

    result_mask = torch.cat(result_mask, dim=0)
    ####print_image_exr(ray_mask,"ray_mask")
    # ###print_image_exr(uv,"angular_uv")
    # ###print_image_exr(another_uv,"space_uv")
    # ###print_image_exr(ray_mask,"ray_mask")
    # ####print_image_exr(left_uv,"left_uv")
    ####print_image_exr(right_uv,"right_uv")
    # result_hit = []
    result_uv = torch.cat(result_uv, dim=0)
    spherical_uv = torch.cat(spherical_uv, dim=0)
    angular_range_list = torch.cat(angular_range_list, dim=0)
    space_range_list = torch.cat(space_range_list, dim=0)
    local_point_left_list = torch.cat(local_point_left_list, dim=0)
    local_point_right_list = torch.cat(local_point_right_list, dim=0)
    standard_dir_list = torch.cat(standard_dir_list, dim=0)

    data["angular_uv"] = result_uv
    data["angular_uv_lenth"] = angular_lenth
    data["right_angular_uv"] = right_uv

    data["space_uv"] = spherical_uv
    data["space_uv_lenth"] = space_lenth
    ###print_image_exr(data["space_uv_lenth"],"space_uv_lenth")
    ###print_image_exr(data["angular_uv_lenth"],"angular_uv_lenth")
    data["space_uv_left"] = torch.clamp((data["space_uv"] - data["space_uv_lenth"] + 1) / 2 * light_space_size // 1,
                                        min=0, max=light_space_size)
    data["space_uv_right"] = torch.clamp(
        (data["space_uv"] + data["space_uv_lenth"] + 1) / 2 * light_space_size // 1 + 1, min=0, max=light_space_size)
    data["angular_uv_left"] = torch.clamp(
        (data["angular_uv"] - data["angular_uv_lenth"] + 1) / 2 * light_angular_size // 1, min=0,
        max=light_angular_size)
    data["angular_uv_right"] = torch.clamp(
        (data["angular_uv"] + data["angular_uv_lenth"] + 1) / 2 * light_angular_size // 1 + 1, min=0,
        max=light_angular_size)

    data["space_uv_left"] = data["space_uv_left"][..., [1, 0]]
    data["space_uv_right"] = data["space_uv_right"][..., [1, 0]]
    data["angular_uv_left"] = data["angular_uv_left"][..., [1, 0]]
    data["angular_uv_right"] = data["angular_uv_right"][..., [1, 0]]

    data["space_uv_left1"] = (data["space_uv"] - data["space_uv_lenth"] + 1) / 2 * light_space_size
    data["space_uv_right1"] = (data["space_uv"] + data["space_uv_lenth"] + 1) / 2 * light_space_size + 1
    data["angular_uv_left1"] = (data["angular_uv"] - data["angular_uv_lenth"] + 1) / 2 * light_angular_size
    data["angular_uv_right1"] = (data["angular_uv"] + data["angular_uv_lenth"] + 1) / 2 * light_angular_size + 1
    ###print_image_exr(data["space_uv_left1"],"space_uv_left1")
    ###print_image_exr(data["space_uv_right1"],"space_uv_right1")
    ###print_image_exr(data["angular_uv_left1"],"angular_uv_left1")
    ###print_image_exr(data["angular_uv_right1"],"angular_uv_right1")

    data["angular_range"] = angular_range_list
    data["space_range"] = space_range_list
    data["local_point_left"] = local_point_left_list
    data["local_point_right"] = local_point_right_list
    data["true_point"] = local_point_right_list
    data["local_normal"] = local_normal
    data["local_ray"] = local_ray
    data["ray_mask"] = result_mask
    data["standard_dir"] = standard_dir_list
    data["cone_scale"] = scale
    data["cone_scale2"] = scale2
    return data
def get_forward_up_right_tensor(to_light):
    to_light = normalize(to_light)
    up = torch.zeros_like(to_light)
    up[..., :] = 0
    up[..., 2:3] = 1
    right = normalize(torch.cross(to_light, up, dim=-1))
    up = normalize(torch.cross(to_light, right, dim=-1))
    return to_light, up, right
def rotate(a, b, c, v):
    return torch.cat([torch.sum((a * v), dim=-1).unsqueeze(-1),
                      torch.sum((b * v), dim=-1).unsqueeze(-1),
                      torch.sum((c * v), dim=-1).unsqueeze(-1)], dim=-1)
def get_light_space_gbuffer(data,forward_indirect):
    lposition = data["local"]["gbuffer"]["lposition"]
    lenth = torch.norm(lposition, keepdim=True, dim=-1)
    z = normalize(-lposition)
    reflect_forward1, reflect_up1, reflect_right1 = get_forward_up_right_tensor(z)

    localGbuffer = {}
    localGbuffer["specular_ray"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                          data["local"]["gbuffer"]["specular_ray"])
    localGbuffer["normal"] = rotate(reflect_right1, reflect_up1, reflect_forward1, data["local"]["gbuffer"]["normal"])
    localGbuffer["view_dir"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                      data["local"]["gbuffer"]["view_dir"])
    localGbuffer["half_vec"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                      data["local"]["gbuffer"]["half_vec"])
    if forward_indirect:
        light_position = data["local"]["lights"]["shadow"]["light_position"]
        lenth = torch.norm(light_position, keepdim=True, dim=-1)
        z = normalize(-light_position)
        reflect_forward1, reflect_up1, reflect_right1 = get_forward_up_right_tensor(z)
        data["local"]["lights"]["shadow"]["light_lnormal"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                                                data["local"]["lights"]["shadow"]["light_normal"])
        data["local"]["lights"]["shadow"]["light_half_vec"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                                                data["local"]["lights"]["shadow"]["light_half_vec"])
        data["local"]["lights"]["shadow"]["light_specular_ray"] = rotate(reflect_right1, reflect_up1, reflect_forward1,
                                                                data["local"]["lights"]["shadow"]["light_specular_ray"])
    # if forward_volume:
    #     print("reflect_right1",reflect_right1.shape)
    #     print("view_dir",data["local"]["gbuffer"]["view_dir"].shape)
    #exit()
    return localGbuffer


# def oct_transformer2(intuv, size):
#     result_int = intuv
#     bool_a = (intuv < 0)
#     bool_b = (intuv >= size)
#     result_int[bool_a] = 0
#     result_int[bool_b] = size - 1
#     bool_uv = bool_a | bool_b
#     bool_v = bool_uv[..., 1].clone()
#     bool_uv[..., 1] = bool_uv[..., 0]
#     bool_uv[..., 0] = bool_v

#     result_int = torch.where(bool_uv, size - 1 - result_int, result_int)
#     result_uv = (result_int + 0.5) / size * 2 - 1
#     return result_uv

def oct_transformer2(intuv, size):
    # [修复 Bug] 1. 必须 clone，防止原地修改污染前面的计算图和原始坐标
    result_int = intuv.clone() 
    
    bool_a = (intuv < 0)
    bool_b = (intuv >= size)
    
    result_int[bool_a] = 0
    result_int[bool_b] = size - 1
    
    bool_uv = bool_a | bool_b
    
    # 巧妙的 U V 轴翻转
    bool_v = bool_uv[..., 1].clone()
    bool_uv[..., 1] = bool_uv[..., 0]
    bool_uv[..., 0] = bool_v

    # 折叠坐标
    result_int = torch.where(bool_uv, size - 1 - result_int, result_int)
    result_uv = (result_int + 0.5) / size * 2 - 1
    return result_uv

import torch
import torch.nn.functional as F







def oct_transform(uv, size):
    B = uv.shape[:-1]
    C = uv.shape[-1:]
    float_uv = (uv + 1) / 2 * size
    intuv = ((uv + 1) / 2 * size).long()
    cha = float_uv - intuv
    intuv_ll = torch.where(cha < 0.5, intuv - 1, intuv)
    cha = torch.where(cha < 0.5, cha + 0.5, cha - 0.5)
    biasu = torch.zeros_like(intuv).cuda()
    biasu[..., :1] = 1
    biasv = torch.zeros_like(intuv).cuda()
    biasv[..., 1:2] = 1
    intuv_rl = intuv_ll + biasu
    intuv_lr = intuv_ll + biasv
    intuv_rr = intuv_ll + biasu + biasv
    uv_ll_mid = oct_transformer2(intuv_ll, size)
    uv_rl_mid = oct_transformer2(intuv_rl, size)
    uv_lr_mid = oct_transformer2(intuv_lr, size)
    uv_rr_mid = oct_transformer2(intuv_rr, size)
    a = (1 - cha[..., 0:1]) * (1 - cha[..., 1:2])
    b = (cha[..., 0:1]) * (1 - cha[..., 1:2])
    c = (1 - cha[..., 0:1]) * (cha[..., 1:2])
    d = (cha[..., 0:1]) * (cha[..., 1:2])

    weight = torch.cat([a, b, c, d], dim=-1)
    return uv_ll_mid, uv_rl_mid, uv_lr_mid, uv_rr_mid, weight


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def active_func(act):
    if act == "sigmoid":
        return torch.sigmoid()


class DitBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            mlp_ratio=4.0,
            moe_config=None,
            soft_max=True
    ):
        super().__init__()
        self.num_heads = num_heads
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        # Attention block
        if norm_layer:
            self.ln_1 = norm_layer(hidden_dim)
            self.ln_2 = norm_layer(hidden_dim)
            print("layer norm yes")
        else:
            self.ln_1 = nn.Identity()
            self.ln_2 = nn.Identity()
            #exit()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        # MLP block
        if moe_config == None:
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            print("lets create moe")
            # self.mlp = PositionwiseFeedforwardLayer(hidden_dim, moe_config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    # def forward(self, x: torch.Tensor,c):
    #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
    #     prex = x
    #     sa = modulate(self.ln_1(x),shift_msa,scale_msa)
    #     sa,_ = self.self_attention(sa,sa,sa,need_weights=False)
    #     x = prex +  gate_msa.unsqueeze(dim=1) * sa
    #     x = x + gate_mlp.unsqueeze(dim=1) * self.mlp(modulate(self.ln_2(x),shift_mlp,scale_mlp))
    #     return x
    def forward(self, x: torch.Tensor, c=None):
        prex = x
        # #print(self.ln_1.type)
        sa = self.ln_1(x)
        # print(sa.shape)f
        sa, _ = self.self_attention(sa, sa, sa, need_weights=False, average_attn_weights=False)
        x = prex + sa
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = True,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cond_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
        )
        self.query_layer = nn.Linear(inner_dim, cond_dim)
        self.value_layer = nn.Linear(cond_dim, cond_dim)
        self.key_layer = nn.Linear(cond_dim, cond_dim)

    def forward(self, x, cond):

        light, weight = self.cross_attn(self.query_layer(x), self.key_layer(cond), self.value_layer(cond),
                                        need_weights=False)
        return light



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Batch_Norm(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.BN = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        x = rearrange(x, 'b n d -> b d n')
        x = self.BN(x)
        x = rearrange(x, 'b d n -> b n d')
        return x





class FourDTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            angular_size: int,
            space_size: int,
            angular_block_size: int,
            space_block_size: int,
            num_layers: int,
            num_heads: int,
            input_dim: int,
            hidden_dim: int,
            mlp_dim: int,
            output_dim: int,
            linear_output: bool,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.space_block_size = space_block_size
        self.angular_block_size = angular_block_size
        self.angular_size = angular_size
        self.space_size = space_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        
        input_dim = input_dim * self.space_block_size * self.space_block_size * self.angular_block_size * self.angular_block_size
        self.input_layer = nn.Linear(
            input_dim, hidden_dim
        )
        
        # 🔥 新增：计算切块后的 4D 网格尺寸
        # 假设 W1=H1=angular_size, W2=H2=space_size
        grid_angular = self.angular_size // self.angular_block_size
        grid_space = self.space_size // self.space_block_size
        
        # 🔥 新增：定义 4D 的可学习绝对位置编码
        # 形状为 (1, grid_angular, grid_angular, grid_space, grid_space, hidden_dim)
        # 初始化推荐使用标准差为 0.02 的正态分布（ViT 标准做法）
        self.pos_embed = nn.Parameter(
            torch.randn(1, grid_angular, grid_angular, grid_space, grid_space, hidden_dim) * 0.02
        )

        if linear_output:
            self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        else:
            self.output_layer = fc_layer(self.hidden_dim, output_dim)
            
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers * 2):
            layers[f"encoder_layer_{i}"] = DitBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                0.0,
                0.0,
                norm_layer,
                4.0
            )

        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        # Reshape and permute the input tensor
        B, W1, H1, W2, H2, C = input.shape
        
        input = input.reshape(B, W1 // self.angular_block_size, self.angular_block_size, H1 // self.angular_block_size,
                              self.angular_block_size, W2 // self.space_block_size, self.space_block_size,
                              H2 // self.space_block_size, self.space_block_size,
                              C).permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9).reshape(B, W1 // self.angular_block_size,
                                                                               H1 // self.angular_block_size,
                                                                               W2 // self.space_block_size,
                                                                               H2 // self.space_block_size, -1)

        # 线性投影到 hidden_dim
        input = self.input_layer(input)
        
        # 🔥 新增：注入位置信息
        # 此时 input 的 shape 是 (B, grid_W1, grid_H1, grid_W2, grid_H2, hidden_dim)
        # self.pos_embed 会利用 PyTorch 的 Broadcasting 机制自动应用到整个 Batch 上
        input = input + self.pos_embed

        cnt = 0
        B, W1, H1, W2, H2, C = input.shape
        for layer in self.layers:
            if cnt % 2 == 0:
                input = input.reshape(B * W1 * H1, W2 * H2, C)
            else:
                input = input.permute(0, 3, 4, 1, 2, 5).reshape(B * W2 * H2, W1 * H1, C)
                
            input = layer(input, None)
            
            if cnt % 2 == 0:
                input = input.reshape(B, W1, H1, W2, H2, C)
            else:
                input = input.reshape(B, W2, H2, W1, H1, C).permute(0, 3, 4, 1, 2, 5)
            
            cnt = cnt + 1
            
        input = self.output_layer(input)
        return input






class TriLinear(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.xy_linear = nn.Linear(input_dim, output_dim)
        self.xz_linear = nn.Linear(input_dim, output_dim)
        self.yz_linear = nn.Linear(input_dim, output_dim)
    def forward(self,x):
        xy = x[:,0:1,...]
        xz = x[:,1:2,...]
        yz = x[:,2:3,...]
        xy = self.xy_linear(xy)
        xz = self.xz_linear(xz)
        yz = self.yz_linear(yz)
        return torch.cat([xy,xz,yz],dim=1)

    



def get_forward_up_right_tensor(to_light):
    to_light = normalize(to_light)
    up = torch.zeros_like(to_light)
    up[..., :] = 0
    up[..., 2:3] = 1
    right = normalize(torch.cross(to_light, up, dim=-1))
    up = normalize(torch.cross(to_light, right, dim=-1))
    return to_light, up, right

def get_forward_up_right_tensor_for_indirect(to_light):
    up = torch.zeros_like(to_light)
    up[..., :] = 0
    up[..., 2:3] = 1
    right = normalize(torch.cross(to_light, up, dim=-1))
    up = normalize(torch.cross(to_light, right, dim=-1))
    return to_light, -up, -right


def rotate(a, b, c, v):
    return torch.cat([torch.sum((a * v), dim=-1).unsqueeze(-1),
                      torch.sum((b * v), dim=-1).unsqueeze(-1),
                      torch.sum((c * v), dim=-1).unsqueeze(-1)], dim=-1)




class DownsampleBy4_To6x1_Strong(nn.Module):
    """
    输入:  x ∈ [B, C, H, W]
    三次 stride=4 降采样: H,W → H/4,W/4 → H/16,W/16 → H/64,W/64
    最终得到 (6×1) 空间  
    并使用 ResBlock + SE + LayerNorm 稳定训练
    """
    def __init__(self, 
                 channels=(256, 256, 256), out_dim=64):
        super().__init__()
        c1, c2, c3 = channels

        # ★ 输入层从 64 → 64，更平滑
        self.stem = nn.Sequential(
            nn.Conv2d(7 + 64, 64, 1, bias=True),
        )

        # ★ 带 skip 的 ResDown block
        self.block1 = self._res_down(64, c1)
        self.block2 = self._res_down(c1, c2)
        self.block3 = self._res_down(c2, c3)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c3, c3 // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3 // 8, c3, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Linear(c3  * 6, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def _res_down(self, in_c, out_c):
        """Residual Downsample: Conv(4x4,stride=4) + skip"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride=4, padding=0),
            nn.GroupNorm(8, out_c),
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.stem(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        w = self.se(x3)
        x3 = x3 * w
        x3 = x3.reshape(x3.shape[0], -1)         
        return self.fc(x3)

import time
def start_timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()                     # ⭐ 必须记录 start
        return start, end
    else:
        return time.perf_counter(), None

def end_timer(start, end=None):
    if isinstance(start, torch.cuda.Event):
        end.record()                       # ⭐ 必须记录 end
        torch.cuda.synchronize()
        return start.elapsed_time(end)     # 返回毫秒
    else:
        return (time.perf_counter() - start) * 1000.0


class IndirectFarwardProxy(nn.Module):
    def __init__(self,indirect_feature):
        super().__init__()
        self.encoder = "cnn"
        self.indirect_feature = indirect_feature
        self.compress_layer = nn.Linear(self.indirect_feature,self.indirect_feature)
        self.indirect_encoder = DownsampleBy4_To6x1_Strong(out_dim = indirect_feature)
        self.indirect_decoder = nn.Sequential(
                fc_layer(10 + self.indirect_feature, self.indirect_feature * 1),
                fc_layer(self.indirect_feature  * 1, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, self.indirect_feature * 1),
                fc_layer(self.indirect_feature * 1, 2))

    def train_step(self,rsm_input,gbuffer_input,gt,mask):
        diffuse_decoder_input = self.compress_layer(rsm_input[...,:64])
        light_albedo,light_position,light_normal = rsm_input[...,64 : 64  +1], rsm_input[...,64  +1: 64  +4], rsm_input[...,64 +4: 64  +7]
        indirect_encoder_input = torch.cat([diffuse_decoder_input,light_albedo,light_position,light_normal],dim=-1)
        indirect_feature = self.indirect_encoder(indirect_encoder_input).unsqueeze(1).unsqueeze(1).repeat(1,512,512,1)
        print("indirect_feature",indirect_feature.shape)
        indirect_decoder_input = torch.cat([indirect_feature,gbuffer_input.repeat(3,1,1,1)],dim=-1)
        indirect_result = self.indirect_decoder(indirect_decoder_input)
        indirect_diffuse_result = indirect_result[...,:1].permute(3,1,2,0)
        indirect_specular_result = indirect_result[...,1:2].permute(3,1,2,0)
        
        indirect_diffuse_result = torch.where(mask[...,:3],gt[...,:3],indirect_diffuse_result)
        indirect_specular_result = torch.where(mask[...,3:6],gt[...,3:6],indirect_specular_result)
        return torch.cat([indirect_diffuse_result,indirect_specular_result],dim=-1)
import math
import torch 
import numpy





def xyz_to_uvd_pixel_center(position,plane_res=32):
    """
    将 3D 坐标转换为 (u, v, d) 采样坐标。
    
    坐标定义：
    1. UV: 标准球面展开，范围 [0, 1]
    2. Depth (d): 对数深度。
       R = 0.1          -> d = 0.5
       R = 0.1 * 1.14   -> d = 1.5
       R = 0.1 * 1.14^k -> d = k + 0.5
    
    Args:
        position (torch.Tensor): 形状 (B, W, H, 3)
    
    Returns:
        torch.Tensor: 形状 (B, W, H, 3)，最后一维是 (u, v, d)
    """
    # 1. 计算物理半径 R (Euclidean Norm)
    # keepdim=True 保持形状方便广播
    radius = torch.norm(position, dim=-1, keepdim=True)
    
    # 避免 R=0 导致 log 报错
    eps = 1e-8
    radius_safe = radius.clamp(min=eps)
    
    # ================= 深度 d 计算 (核心修改) =================
    # 公式推导：
    # 我们需要一个线性映射: log_base(R/0.1) -> d_index
    # 当 R=0.1 时，log项为0，我们需要结果为 0.5，所以直接 + 0.5

    r_center_0 = 0.1
    if plane_res == 32:
        growth_rate = 1.14
    elif plane_res == 64:
        growth_rate = 1.07
    elif plane_res == 128:
        growth_rate = 1.033
    elif plane_res == 16:
        growth_rate = 1.303
    else:
        print("no such res",plane_res)
        exit()


    
    ln_growth = math.log(growth_rate)
    ln_r0 = math.log(r_center_0)
    
    # d = log_{1.14}(R / 0.1) + 0.5
    #   = (ln(R) - ln(0.1)) / ln(1.14) + 0.5
    d_val = (torch.log(radius_safe) - ln_r0) / ln_growth + 0.5
    
    # ================= UV 计算 (保持不变) =================
    # 归一化方向向量
    unit_vec = position / radius_safe
    x = unit_vec[..., 0]
    y = unit_vec[..., 1]
    z = unit_vec[..., 2]
    
    abs_z = torch.abs(z).clamp(max=1.0)
    r_sphere = torch.sqrt(1.0 - abs_z)
    s = torch.sqrt(x**2 + y**2).clamp(min=eps)
    
    px = (x / s).clamp(-1.0, 1.0)
    py = (y / s).clamp(-1.0, 1.0)
    
    phi = torch.atan2(torch.abs(py), torch.abs(px))
    alpha = 4.0 * phi / math.pi
    diff = r_sphere * (alpha - 1.0)
    
    sum_abs = torch.where(z < 0.0, 2.0 - r_sphere, r_sphere)
    
    up = ((sum_abs - diff) * 0.5).clamp(0.0, 1.0)
    vp = ((sum_abs + diff) * 0.5).clamp(0.0, 1.0)
    
    sign_x = torch.where(x >= 0.0, 1.0, -1.0)
    sign_y = torch.where(y >= 0.0, 1.0, -1.0)
    
    uv_x = (sign_x * up + 1.0) * 0.5
    uv_y = (sign_y * vp + 1.0) * 0.5
    
    # ================= 组合输出 =================
    # 调整形状以拼接
    uv_x = uv_x.unsqueeze(-1)
    uv_y = uv_y.unsqueeze(-1)
    # d_val 已经是 (B, W, H, 1)
    
    return torch.cat([uv_y,uv_x, d_val], dim=-1)




class PlaneDecoder(nn.Module):
    # ... (你的原有代码) ...
    def __init__(self,  embedding_dim=1024, decoder_dim=64,plane_resolution=32):
        super().__init__()
        # ... (略: 请确保包含原有代码) ...
        self.embedding_size = embedding_dim
        self.decoder_dim = decoder_dim
        self.up_encoder = nn.ModuleList()
        self.plane_cnt =plane_resolution
        self.decoder_dim = decoder_dim
        self.radiance_linear_layer_list = nn.ModuleList()
        self.aux_linear_layer_list = nn.ModuleList()
   

        self.tri_plane_embed = nn.Parameter(0.01 * torch.randn(3, self.plane_cnt, self.plane_cnt, self.embedding_size),
                                        requires_grad=True)

        self.transformer_decoder = CrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                        num_heads=16, eps=1e-6)
        self.tri_output_layer = TriLinear(self.embedding_size,self.decoder_dim)

            
    def forward_triplane(self, photon_texture):
        B,_,_,_,_,C = photon_texture.shape
        query = self.tri_plane_embed.reshape(-1, self.embedding_size).unsqueeze(0).repeat(B, 1, 1)
        photon_texture = photon_texture.reshape(B, -1, C)
        if self.decoder_tri == False:
            triplane_feature = self.transformer_decoder(query, photon_texture)
        else:
            triplane_feature = self.tri_transformer_decoder(query, photon_texture)
        triplane_feature = triplane_feature.reshape(B, 3, self.plane_cnt , self.plane_cnt , self.embedding_size)
        if self.compress_tri== False:
            compressed_triplane_feature = self.output_layer(triplane_feature)
        else:
            compressed_triplane_feature = self.tri_output_layer(triplane_feature)
        return triplane_feature,compressed_triplane_feature
class TriplaneImageReconstructor(nn.Module):
    def __init__(self, plane_res=128, plane_dim=64,output_dim=3):
        super().__init__()
        self.plane_res = plane_res
        self.plane_dim = plane_dim
        self.planes = nn.Parameter(torch.randn(32,3, plane_dim, plane_res, plane_res) * 0.1)

    def forward_light(self,light_data):
        global_light_feature = self.image_encoder(light_data)
        if torch.any(global_light_feature.isnan()):
            print("global_light_feature nan")
            exit()
        return self.image_to_plane.forward_triplane(global_light_feature)
    
    def triplane_loss(self,plane_feature):
        h_diff = plane_feature[..., 1:, :] - plane_feature[..., :-1, :]
        w_diff = plane_feature[..., :, 1:] - plane_feature[..., :, :-1]
        loss_tv = torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))
        loss_l2 = torch.mean(plane_feature ** 2)
        loss = 0.05 * loss_tv + 0.0001 * loss_l2
        return loss
    def build_unique_triplanes(self,C, X, Y, Z,device):
        """
        构造三个 plane，使 triplane 重建后的 voxel 全唯一
        返回:
            xy: [C, X, Y]
            xz: [C, X, Z]
            yz: [C, Y, Z]
        """

        # 创建坐标网格
        xs = torch.arange(X, device=device).view(X, 1)
        ys = torch.arange(Y, device=device).view(1, Y)
        zs = torch.arange(Z, device=device).view(1, Z)

        # ⚠️ 位权设计（关键！！）
        base_y = Z + 1
        base_x = (Y + 1) * base_y

        # 三个 plane 编码
        xy = xs * base_x + ys * base_y          # [X, Y]
        xz = xs * base_x + zs                   # [X, Z]
        yz = ys * base_y + zs                   # [Y, Z]

        # 扩展到 channel
        xy = xy.unsqueeze(0).repeat(C, 1, 1).float()
        xz = xz.unsqueeze(0).repeat(C, 1, 1).float()
        yz = yz.unsqueeze(0).repeat(C, 1, 1).float()

        return xy, xz, yz
    def fetch_local_features(self, plane_feature, coords = None):
        """
        plane_feature: [B, 3, C, H, W] (假设 H, W 是全局分辨率 64)
        coords: [B, 6] -> sx, ex, sy, ey, sz, ez
        """
        batch_size = plane_feature.shape[0]
        local_features = []
        # print(plane_feature.shape)
        for i in range(batch_size):
            
            xy_plane = plane_feature[i, 0] # [C, X, Y]
            xz_plane = plane_feature[i, 1] # [C, X, Z]
            yz_plane = plane_feature[i, 2] # [C, Y, Z:
            if coords != None:
                sx, ex, sy, ey, sz, ez = coords[i]
                xy_crop = xy_plane[:, sx:ex, sy:ey] 
                
                # XZ: [sx:ex, sz:ez]
                xz_crop = xz_plane[:, sx:ex, sz:ez]
                
                # YZ: [sy:ey, sz:ez]
                yz_crop = yz_plane[:, sy:ey, sz:ez]
            else:
                xy_crop = xy_plane[:, :, :] 
                
                # XZ: [sx:ex, sz:ez]
                xz_crop = xz_plane[:, :, :]
                
                # YZ: [sy:ey, sz:ez]
                yz_crop = yz_plane[:, :, :]
            feat = (xy_crop.unsqueeze(-1) + xz_crop.unsqueeze(-2) + yz_crop.unsqueeze(-3))
            local_features.append(feat)
            
        return torch.stack(local_features) # [B, C, 32, 32, 32]

    def forward_decoder_only(self,feature):
        return self.decoder(feature)

    def forward_decoder(self, idx_list, targets, coords): # 增加 coords
        plane_feature = self.planes[idx_list]
        plane_loss = self.triplane_loss(plane_feature)
        feature = self.fetch_local_features(plane_feature, coords)
        B, C, X, Y, Z = feature.shape
        _, _, _, _, W, H, _ = targets.shape
        feature = feature.permute(0, 2, 3, 4, 1).contiguous()
        img = self.decoder(feature.reshape(-1, self.plane_dim))
        img = img.reshape(B, X, Y, Z, W, H, 1)
        visual_loss = nn.L1Loss()(img, targets)
        loss = visual_loss + plane_loss * 0.01
        return img, loss, visual_loss, plane_loss
    def sample_from_triplane(self,plane_features, coords,mode="nearest"):
        """
        从三平面中根据3D坐标采样特征
        
        参数:
        plane_features: [B, 3, C, H_p, W_p] 三个平面的特征图
        coords: [B, W, H, 3] 采样坐标，最后一维的 3 代表 (X, Y, Z)
                !!! 注意：为了使用 grid_sample，coords 的值域必须归一化到 [-1, 1] 之间 !!!
                
        返回:
        sampled_features: [B, C, W, H] 采样组合后的特征
        """
        B, W, H, _ = coords.shape
        xy_plane = plane_features[:, 0]
        xz_plane = plane_features[:, 1]
        yz_plane = plane_features[:, 2]
        # 形状均为 [B, W, H]
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        grid_xy = torch.stack([y, x], dim=-1)
        grid_xz = torch.stack([z, x], dim=-1)
        grid_yz = torch.stack([z, y], dim=-1)
        grid_xy = grid_xy.to(xy_plane.dtype)
        grid_xz = grid_xz.to(xy_plane.dtype)
        grid_yz = grid_yz.to(xy_plane.dtype)
        feat_xy = F.grid_sample(xy_plane, grid_xy, mode=mode, padding_mode='border', align_corners=False)
        feat_xz = F.grid_sample(xz_plane, grid_xz, mode=mode, padding_mode='border', align_corners=False)
        feat_yz = F.grid_sample(yz_plane, grid_yz, mode=mode, padding_mode='border', align_corners=False)
        sampled_features = feat_xy + feat_xz + feat_yz
        
        return sampled_features
    @torch.no_grad()
    def forward_decoder_no_grad(self, idx_list): # 增加 coords
        plane_feature = self.planes[idx_list]
        plane_loss = self.triplane_loss(plane_feature)
        feature = self.fetch_local_features(plane_feature, None)
        B, C, X, Y, Z = feature.shape
        feature = feature.permute(0, 2, 3, 4, 1).contiguous()
        img = self.decoder(feature.reshape(-1, self.plane_dim))
        img = img.reshape(B, X, Y, Z, 8, 8, 1)
        return img, feature
    # def forward_decoder(self, idx_list,targets):
    #     plane_feature = self.planes[idx_list]
    #     plane_loss = self.triplane_loss(plane_feature)
    #     xy = plane_feature[:, 0]
    #     xz = plane_feature[:, 1]
    #     yz = plane_feature[:, 2]
    #     feature = (xy[:, :, :, :, None] + xz[:, :, :, None, :] + yz[:, :, None, :, :])
    #     _,_,_,_,W,H,_ = targets.shape
    #     _,_,X,Y,Z = feature.shape
    #     feature = feature.permute(0,2,3,4,1).contiguous().reshape(-1,self.plane_dim)
    #     img = self.decoder(feature)
    #     img = img.reshape(len(idx_list),X,Y,Z,W,H,1)
    #     visual_loss = nn.L1Loss()(img, targets) # 修正: 在内部调用 criterion 避免全局变量依赖
    #     loss = visual_loss+ plane_loss * 0.01
    #     return img,loss,visual_loss,plane_loss

    def freeze_decoder_for_stage1(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.planes.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.image_to_plane.parameters():
            param.requires_grad = True
        print("Model status: Decoder FROZEN, Encoder TRAINABLE.")

    def forward_encoder(self, light_data, targets, coords, return_planes=True, indices=None): # 增加 coords
        global_light_feature = self.image_encoder(light_data)
        _, raw_planes = self.image_to_plane.forward_triplane(global_light_feature)
        
        generated_planes = raw_planes.permute(0, 1, 4, 2, 3).contiguous() 
        # generated_planes: [B, 3, C, 64, 64]
        
        plane_reg_loss = self.triplane_loss(generated_planes)
        
        # === 修改核心逻辑 ===
        # 使用 coords 切割生成的 planes
        feature = self.fetch_local_features(generated_planes, coords)
        # feature shape: [B, C, 32, 32, 32]
        
        B, C, X, Y, Z = feature.shape
        feature = feature.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, self.plane_dim)
        
        img = self.decoder(feature) 
        _,_,_,_,W,H,_ = targets.shape
        img = img.reshape(B, X, Y, Z, W, H, 1)
        
        with torch.no_grad():    
            target_planes_gt = self.planes[indices]
            
        loss_distill = nn.L1Loss()(generated_planes, target_planes_gt)
        visual_loss = nn.L1Loss()(img, targets)
        loss = visual_loss + 1.0 * loss_distill + 0.01 * plane_reg_loss
        
        if return_planes:
            return img, loss, visual_loss, loss_distill, plane_reg_loss, generated_planes
        else:
            return img, loss, visual_loss, loss_distill, plane_reg_loss

def sample_from_triplane(plane_features, coords,mode="nearest"):
        """
        从三平面中根据3D坐标采样特征
        
        参数:
        plane_features: [B, 3, C, H_p, W_p] 三个平面的特征图
        coords: [B, W, H, 3] 采样坐标，最后一维的 3 代表 (X, Y, Z)
                !!! 注意：为了使用 grid_sample，coords 的值域必须归一化到 [-1, 1] 之间 !!!
                
        返回:
        sampled_features: [B, C, W, H] 采样组合后的特征
        """
        B, W, H, _ = coords.shape
        xy_plane = plane_features[:, 0]
        xz_plane = plane_features[:, 1]
        yz_plane = plane_features[:, 2]
        # 形状均为 [B, W, H]
        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]
        grid_xy = torch.stack([y, x], dim=-1)
        grid_xz = torch.stack([z, x], dim=-1)
        grid_yz = torch.stack([z, y], dim=-1)
        grid_xy = grid_xy.to(xy_plane.dtype)
        grid_xz = grid_xz.to(xy_plane.dtype)
        grid_yz = grid_yz.to(xy_plane.dtype)
        feat_xy = F.grid_sample(xy_plane, grid_xy, mode=mode, padding_mode='border', align_corners=False)
        feat_xz = F.grid_sample(xz_plane, grid_xz, mode=mode, padding_mode='border', align_corners=False)
        feat_yz = F.grid_sample(yz_plane, grid_yz, mode=mode, padding_mode='border', align_corners=False)
        sampled_features = feat_xy + feat_xz + feat_yz
        return sampled_features
        
class PlanePostProcess(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, upsample=True):
        """
        Args:
            in_channels (int): 输入通道数 (C)
            hidden_channels (int): 隐藏层通道数
            upsample (bool): 是否进行上采样。
                             True -> 输出尺寸为 4H x 4W
                             False -> 输出尺寸为 H x W (原地卷积)
        """
        super(PlanePostProcess, self).__init__()
        self.enable_upsample = upsample
        
        # 1. 头部：特征映射到高维空间
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        if self.enable_upsample:
            self.up1 = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
            self.up2 = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        else:
            self.body = nn.Sequential(
                # 对应原 up1 的计算量
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,padding_mode='replicate'),
                nn.ReLU(inplace=True),
                
                # 对应原 up2 的计算量
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,padding_mode='replicate'),
                nn.ReLU(inplace=True)
            )
        
        # 4. 输出层：映射回原始通道数
        self.tail = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1,padding_mode='replicate')

    def forward(self, x):
        # x: [B, C, W, H]
        
        # 提取特征
        x = self.head(x) 
        
        # 根据初始化时的选项决定路径
        if self.enable_upsample:
            x = self.up1(x) # [B, Hidden, 2W, 2H]
            x = self.up2(x) # [B, Hidden, 4W, 4H]
        else:
            x = self.body(x) # [B, Hidden, W, H] 尺寸不变
        
        # 输出重构
        x = self.tail(x) 
        
        return x
    


class NelifDecoder(nn.Module):
    def __init__(self, configs, loss_config,need_direct,need_indirect,need_shadow,need_encoder=False):
        super().__init__()
        if need_direct:
            print("direct init")
        if need_indirect:
            print("indirect init")
        self.configs = configs
        self.angular_size = configs["light_angular_resolution"]
        self.space_size = configs["light_direction_resolution"]
        self.loss_config = loss_config
        self.channel_cut = True
        self.light_input_dim = 9
        if self.channel_cut:
            self.light_input_dim = 7
        self.decoder_light_feature = configs["plane"]["decoder_direct_dim"]
        self.plane_res = 64
        self.image_encoder = FourDTransformer(angular_size=8 , space_size=128 , angular_block_size=1, space_block_size=16,
                                            num_layers=3, num_heads=16, input_dim=7, hidden_dim=256, mlp_dim=1024,
                                            output_dim=1024, linear_output=True)
        self.image_to_plane = PlaneDecoder(1024,self.decoder_light_feature,self.plane_res)
        
        self.loss_func = Loss(loss_config) if loss_config else None
        decoder_light_feature = self.decoder_light_feature
        self.dir_data = pyexr.read(r"../datasets2/TogLightAll8x128/OutDir.exr")[...,:3].reshape(8,128,8,128,3).transpose(0,2,1,3,4)
        self.output_dim = 1
        self.conditional_layer = False

        if need_direct:
            self.diffuse_decoder = nn.Sequential(
                fc_layer(self.decoder_light_feature * 1 + 12, self.decoder_light_feature * 1),
                fc_layer(self.decoder_light_feature * 1, self.decoder_light_feature * 1),
                fc_layer(self.decoder_light_feature * 1, self.decoder_light_feature * 1),
                fc_layer(self.decoder_light_feature * 1, self.output_dim))
            self.specular_decoder = nn.Sequential(
                fc_layer(self.decoder_light_feature * 1 + 12, self.decoder_light_feature * 1),
                fc_layer(self.decoder_light_feature * 1, self.decoder_light_feature * 1),
                fc_layer(self.decoder_light_feature * 1, self.decoder_light_feature * 1),
                fc_layer(self.decoder_light_feature * 1, self.output_dim))

        if need_shadow:
            self.shadow_network = NelifShadowNetwork(16 + 5)
            self.direct_compress_layer = nn.Linear(self.decoder_light_feature, 16)
        self.indirect_feature =64
        self.indirect_proxy = IndirectFarwardProxy(self.indirect_feature)
        self.sampling_way = configs["sampling"]
    
        
        self.upsampler = PlanePostProcess(64,128,False)
            
    def forward_light(self,data,gt_plane):

        data["direction"] = self.dir_data
        lightData = data["global"]
        #print(data["global"].keys())
        lightData["radiance"] = lightData["radiance"].permute(0,-1,1,2,3,4).reshape(3,*lightData["radiance"].shape[1:-1],1)
        # print(lightData["radiance"].shape)
        # print(lightData["radiance"].reshape(3,-1,1).max(dim=1).values)
        light_input = torch.cat(
            [lightData["radiance"],lightData["position"].repeat(3,1,1,1,1,1),lightData["direction"].repeat(3,1,1,1,1,1)], dim=-1)
        with torch.no_grad():
            #print(light_input.shape)
            global_light_feature = self.image_encoder(light_input)
            #print(global_light_feature.shape)
            _, raw_planes = self.image_to_plane.forward_triplane(global_light_feature)
        
        generated_plane = raw_planes.permute(0, 1, 4, 2, 3).contiguous() 
        # if torch.any(light_input.isnan()):
        #     print("light input nan")
        #     exit()
        # if torch.any(global_light_feature.isnan()):
        #     print("global_light_feature nan")
        #     exit()
        return generated_plane

    

    def forward_direct(self,decoder_type,data,result,light_feature,light_space_gbuffer,shading,mask):
        gbuffer_input = torch.cat([light_space_gbuffer["normal"], light_space_gbuffer["half_vec"],
                 light_space_gbuffer["specular_ray"],
                 data["local"]["gbuffer"]["roughness"],
                 data["local"]["gbuffer"]["dot"],
                 data["local"]["gbuffer"]["lenth"]],dim=-1).repeat(3,1,1,1)
        specular_input = torch.cat(
            [light_feature, gbuffer_input], dim=-1)

        target_dtype = next(self.diffuse_decoder.parameters()).dtype

        # 2. 将输入强制对齐到这个类型
        specular_input = specular_input.to(target_dtype)
        if decoder_type=="diffuse":
            direct_shading = self.diffuse_decoder(specular_input)
        elif decoder_type=="specular":
            direct_shading = self.specular_decoder(specular_input)
        elif decoder_type=="direct":
            direct_shading = self.direct_decoder(specular_input)
        direct_shading = direct_shading.permute(3,1,2,0)
        print("direct_shading",direct_shading.shape)
        direct_shading = torch.where(mask,
                                     shading,direct_shading)
        return direct_shading

    def forward_shadow(self,data,result,light_feature,light_space_gbuffer):
        localData = data["local"]
        data["local"]["shadow_light_repr"] = self.direct_compress_layer(data["local"]["direct_light_reprs"])
        data["local"]["shadow_input"] = get_lightformer_input(data["local"]["lights"], data["local"],True).to(data["local"]["shadow_light_repr"].dtype)
        shadow_result = self.shadow_network.step(data)

        shadow_result = 1 - shadow_result
        if torch.any(torch.isnan(shadow_result)):
            print("nan")
            exit()
        shadow_result = torch.where(
            data["local"]["mask"] | data["local"]["shadow_mask"],
            localData["shadow"],
            shadow_result)
 
        return shadow_result
    

    def forward(self, data, need_diffuse, need_specular, need_shadow, need_indirect):
        timers = {}   # <---- 用于存储每个阶段的耗时(ms)
        
        light_space_gbuffer = buffer_process(data, False)
        for key in light_space_gbuffer:
            data["local"][key] = light_space_gbuffer[key]
    
        gt_plane = data["global"]["plane"]
        B = gt_plane.shape[0]
        gt_plane = gt_plane.reshape(B*3,*gt_plane.shape[2:])
        #sampled_plane = self.forward_light(data)
        sampled_plane = gt_plane

        B = data["global"]["plane"].shape[0]
        voxel_coord = xyz_to_uvd_pixel_center(data["local"]["gbuffer"]["lposition"],self.plane_res)
        voxel_coord[...,:2] = voxel_coord[...,:2] * 2 - 1
        voxel_coord[...,2:3] = voxel_coord[...,2:3] /self.plane_res * 2 - 1
        voxel_coord = voxel_coord.repeat(3,1,1,1)
        B,L,C,W,H = sampled_plane.shape
        sampled_plane = self.upsampler(sampled_plane.reshape(B*L,C,W,H))
        sampled_plane = sampled_plane.reshape(B,L,C,self.plane_res,self.plane_res)
        light_feature = sample_from_triplane_oct(sampled_plane,voxel_coord[...,[0,1,2]],mode='bilinear').permute(0,2,3,1)
        data["local"]["voxel_coord"] = ((voxel_coord* 0.5 + 0.5) * self.plane_res)
        data["local"]["feature_visualize"] = light_feature[...,:3]

        data["local"]["planeuv_visualize"] = sampled_plane[:,0,...].permute(0,2,3,1)[...,:3]
        data["local"]["planeud_visualize"] = sampled_plane[:,1,...].permute(0,2,3,1)[...,:3]
        data["local"]["planevd_visualize"] = sampled_plane[:,2,...].permute(0,2,3,1)[...,:3]
     

        # ====== 你的原代码保持不动 =======
        if torch.any(light_feature.isnan()):
            print("light_feature nan")
            exit()
        data["local"]["sampled_principle_value"] = light_feature[..., :3]
        data["local"]["direct_light_reprs"] = light_feature
        

        result = {}

        
        if need_indirect:
            #s, e = start_timer()
            indirect_data = data["local"]["lights"]["shadow"]
            invoxel_coord = xyz_to_uvd_pixel_center(indirect_data["light_position"],self.plane_res)
            invoxel_coord[...,:2] = invoxel_coord[...,:2] * 2 - 1
            invoxel_coord[...,2:3] = invoxel_coord[...,2:3] /self.plane_res * 2 - 1
            invoxel_coord = invoxel_coord.repeat(3,1,1,1)
            indirect_feature = sample_from_triplane_oct(sampled_plane,invoxel_coord[...,[0,1,2]],mode='bilinear').permute(0,2,3,1)
            print("indirect_feature",indirect_feature.shape)
      

            lp = indirect_data["light_position"]
            sp = data["local"]["gbuffer"]["lposition"]
                   
            print("light albedo shape",indirect_data["light_albedo"].shape)
            print(" indirect_feature shape",indirect_feature.shape)
            rsm_input = torch.cat([indirect_feature,indirect_data["light_albedo"].permute(3,1,2,0),lp.repeat(3,1,1,1),indirect_data["light_normal"].repeat(3,1,1,1)],dim=-1)
            data["local"]["rsm_input"] = rsm_input
            
            screen_input = torch.cat([data["local"]["gbuffer"]["normal"], 
                data["local"]["gbuffer"]["view_dir"],
                data["local"]["gbuffer"]["roughness"],
                sp],dim=-1)
            data["local"]["screen_input"] = screen_input
            indirect_result = self.indirect_proxy.train_step(
                rsm_input,screen_input,torch.cat([data["local"]["log1p_diffuse_indirect_shading"],
                        data["local"]["log1p_specular_indirect_shading"]], dim=-1),
                torch.cat([data["local"]["mask"] | data["local"]["albedo_mask"],
                        data["local"]["mask"] | data["local"]["specular_mask"]], dim=-1)
            )
    

            result["log1p_diffuse_indirect_shading"] = indirect_result[...,:3]
            result["log1p_specular_indirect_shading"] = indirect_result[...,3:6]
        if need_diffuse:
            #s, e = start_timer()
            pred_diffuse = self.forward_direct(
                "diffuse", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_diffuse_direct_shading"],
                data["local"]["mask"] | data["local"]["albedo_mask"]
            )
            result["log1p_diffuse_direct_shading"] = pred_diffuse

        if need_specular:
            #s, e = start_timer()
            pred_specular = self.forward_direct(
                "specular", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_specular_direct_shading"],
                data["local"]["mask"] | data["local"]["specular_mask"]
            )
            result["log1p_specular_direct_shading"] = pred_specular
            #print("log1p_specular_indirect_shading",result["log1p_specular_indirect_shading"].shape)
        if need_shadow:
            result["shadow"] = self.forward_shadow(
                data, result, light_feature, light_space_gbuffer
            )

            #print("shadow",result["shadow"].shape)

        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        else:
            loss_map = None
        #print(timers)
        
        #data["local"]["compressed_principle_triplane"] = compressed_principle_triplane
        return data, result, loss_map, timers


def buffer_process(data,forward_indirect):
    specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                        dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                            data["local"]["gbuffer"]["view_dir"]
    data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
    data["local"]["gbuffer"]["half_vec"] = normalize(
        (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
    data["local"]["gbuffer"]["dot"] = torch.sum(
        data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)
    
    light_space_gbuffer = get_light_space_gbuffer(data,forward_indirect)
    return light_space_gbuffer





def pad_equal_area_plane_correct(uv_plane):
    """
    为等面积投影的 UV 特征平面进行严格拓扑正确的 1 像素 Padding。
    uv_plane: [B, C, H, W]
    """
    B, C, H, W = uv_plane.shape
    padded = torch.zeros((B, C, H + 2, W + 2), dtype=uv_plane.dtype, device=uv_plane.device)
    
    # 填入中心原始数据
    padded[:, :, 1:-1, 1:-1] = uv_plane
    
    # 🌟 1. 顶部边缘填充：拿原图的 Top Row，进行水平翻转 (U -> -U)
    # uv_plane[:, :, 0, :] 的形状是 [B, C, W]，我们在最后一个维度 (-1) 上翻转
    padded[:, :, 0, 1:-1] = torch.flip(uv_plane[:, :, 0, :], dims=[-1])
    
    # 🌟 2. 底部边缘填充：拿原图的 Bottom Row，进行水平翻转 (U -> -U)
    padded[:, :, -1, 1:-1] = torch.flip(uv_plane[:, :, -1, :], dims=[-1])
    
    # 🌟 3. 左侧边缘填充：拿原图的 Left Column，进行垂直翻转 (V -> -V)
    # uv_plane[:, :, :, 0] 的形状是 [B, C, H]，我们在最后一个维度 (-1) 上翻转
    padded[:, :, 1:-1, 0] = torch.flip(uv_plane[:, :, :, 0], dims=[-1])
    
    # 🌟 4. 右侧边缘填充：拿原图的 Right Column，进行垂直翻转 (V -> -V)
    padded[:, :, 1:-1, -1] = torch.flip(uv_plane[:, :, :, -1], dims=[-1])
    
    # 🌟 5. 四个角落填充：物理上同属南极点，呈中心对角线映射
    padded[:, :, 0, 0]   = uv_plane[:, :, -1, -1] 
    padded[:, :, 0, -1]  = uv_plane[:, :, -1, 0]  
    padded[:, :, -1, 0]  = uv_plane[:, :, 0, -1]   
    padded[:, :, -1, -1] = uv_plane[:, :, 0, 0]    
    
    return padded


def sample_from_triplane_oct(plane_features, coords, mode="bilinear"):
    """
    从三平面中根据3D坐标采样特征 (支持等面积映射的无缝拓扑)
    
    参数:
    plane_features: [B, 3, C, H_p, W_p]
    coords: [B, W_out, H_out, 3] (X, Y, Z) 范围在 [-1, 1]
    """
    B, W_out, H_out, _ = coords.shape
    
    # 获取特征维度
    _, _, C, H_p, W_p = plane_features.shape
    
    xy_plane = plane_features[:, 0] # 这是你的 (U, V) 等面积平面
    xz_plane = plane_features[:, 1]
    yz_plane = plane_features[:, 2]
    
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]
    
    # =======================================================
    # 🌟 1. 对等面积平面进行 Padding 和 坐标缩放
    # =======================================================
    padded_xy_plane = pad_equal_area_plane_correct(xy_plane)
    
    # grid_sample 需要的坐标最后一维是 (x_grid, y_grid)，对应宽(W_p)和高(H_p)
    # 根据你的代码：grid_xy_old = torch.stack([y, x], dim=-1)
    # 意味着 y 对应宽 W_p，x 对应高 H_p
    scale_w = W_p / (W_p + 2.0)
    scale_h = H_p / (H_p + 2.0)
    
    y_scaled = y * scale_w
    x_scaled = x * scale_h
    
    # 构造缩放后的网格
    grid_xy_scaled = torch.stack([y_scaled, x_scaled], dim=-1).to(xy_plane.dtype)
    
    # =======================================================
    # 🌟 2. 构造普通的 XZ 和 YZ 网格 (不 Padding)
    # =======================================================
    grid_xz = torch.stack([z, x], dim=-1).to(xy_plane.dtype)
    grid_yz = torch.stack([z, y], dim=-1).to(xy_plane.dtype)
    
    # =======================================================
    # 🌟 3. 采样 (注意 padding_mode 的选择)
    # =======================================================
    # 对 xy 使用 'zeros' 即可，因为在 scaled 坐标下，合法值永远出不了 padding 那一圈
    feat_xy = F.grid_sample(padded_xy_plane, grid_xy_scaled, mode=mode, padding_mode='zeros', align_corners=False)
    
    # 对 xz 和 yz 依然使用 'border' 兜底
    feat_xz = F.grid_sample(xz_plane, grid_xz, mode=mode, padding_mode='border', align_corners=False)
    feat_yz = F.grid_sample(yz_plane, grid_yz, mode=mode, padding_mode='border', align_corners=False)
    
    sampled_features = feat_xy + feat_xz + feat_yz
    
    return sampled_features
