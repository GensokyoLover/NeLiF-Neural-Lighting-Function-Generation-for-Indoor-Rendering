import copy
import os
import sys
import pickle
import random
import json
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
import pyexr
import gzip
import zstandard as zstd
import torch.nn.functional as F
import numpy as np
from utils.data_utils import safe_divide_np, to_cuda, to_cpu
from common.shttools import load_pklzst
import time

import cv2
nan_cnt = 0


def check_data_valid(data, pred=""):
    global nan_cnt
    is_fix = False
    for k in data:
        if isinstance(data[k], dict):
            if not check_data_valid(data[k], pred + k + "_"):
                is_fix = True
        else:
            try:
                nan_mask = np.isnan(data[k])
                if np.any(nan_mask):
                    nan_cnt = nan_cnt + 1
                    data[k][nan_mask] = np.ones_like(data[k])[nan_mask] * 0.5
                    is_fix = True
                inf_mask = np.isinf(data[k])
                if np.any(inf_mask):
                    data[k][inf_mask] = np.ones_like(data[k])[inf_mask] * 0.5
                    is_fix = True
            except TypeError as e:
                raise e
    if is_fix:
        return False
    return True

local_attribute = ["position", "normal", "view_dir", "albedo", "roughness", "specular", "toLight",
                   "emission", "direct_shading", "shadow",
                   "indirect_shading", "pixel_emitter_distance", "occluder_emitter_distance", "shadowmap",
                   "light_dir", "half_vec",
                   "log1p_emission", "log1p_direct_shading", "log1p_indirect_shading", "beauty",
                   "log1p_beauty", "instance_mask"]
gbuffer_feature = ["position", "cposition", "normal", "view_dir", "albedo", "roughness", "specular", "instance",
                   "lposition", "depth", "hiroughness", "fresnel", "second_lposition","lenth"]
shadow_feature = ["pixel_emitter_distance", "occluder_emitter_distance", "shadowmap", "light_normal",
                  "light_view_distance", "light_position", "light_ws_position","depth_gradiant",
                  "light_instance", "light_albedo", "LightDir", "LightPosition","light_flux","light_cos",
                  "light_view_dir","light_specular_ray","light_half_vec","light_depth","light_roughness","light_dot"]
specular_feature = ["light_dir", "half_vec"]
feature_dict = {
    "gbuffer": gbuffer_feature,
    "shadow": shadow_feature,
    "specular": specular_feature
}

from skimage.transform import resize

def recognization(data):
    data["gbuffer"] = {}
    data["lights"] = {"shadow": {}, "specular": {}}
    dellist = []
    for key in feature_dict:
        for name in feature_dict[key]:
            if key == "gbuffer":
                if name not in data.keys():
                    continue
                data[key][name] = data[name]

            else:
                if name not in data.keys():
                    continue
                data["lights"][key][name] = data[name]
            dellist.append(name)
    for key in dellist:
        data.pop(key)
    return data
 
def downsample_tensor_half(x: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    """
    将形状为 (B, W, H, C) 的张量在空间维度 (W, H) 上下采样到 1/2。
    
    参数:
        x: 输入张量，形状必须为 (B, W, H, C)
        mode: 下采样模式，可选 'bilinear' (双线性插值), 'nearest' (最近邻插值), 'avg' (平均池化)
    返回:
        下采样后的张量，形状为 (B, W/2, H/2, C)
    """
    # 1. 将 (B, W, H, C) 转换为 PyTorch 标准的通道优先格式 (B, C, W, H)
    x = x.permute(0, 3, 1, 2)
    
    # 2. 执行 1/2 下采样
    if mode == 'bilinear':
        # 双线性插值：平滑，适合图像和连续特征
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    
    elif mode == 'nearest':
        # 最近邻插值：速度最快，不会产生新的数值，适合 Mask 或分类标签
        x = F.interpolate(x, scale_factor=0.5, mode='nearest')
    
    elif mode == 'avg':
        # 平均池化：特征聚合，适合卷积神经网络的特征图下采样
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
    else:
        raise ValueError(f"不支持的下采样模式: {mode}。请选择 'bilinear', 'nearest' 或 'avg'。")
    
    # 3. 将结果还原回 (B, W, H, C) 格式
    x = x.permute(0, 2, 3, 1)
    
    return x

def video_process_tensor_torch(data, indirect=False,post_relative=False,voxel=False):
    eps = 1e-5
    # 1. Normalize light_dir
    light_dir = data["local"]["light_dir"]
    dir_length = torch.norm(light_dir, dim=-1, keepdim=True).clamp(min=eps)
    data["local"]["light_dir"] = light_dir / dir_length

    # 2. Compute toLight and lposition
    to_light = torch.cat([data["local"]["light_dir"], data["local"]["pixel_emitter_distance"]], dim=-1)
    data["local"]["toLight"] = to_light
    u = data["local"]["light_dir"]
    v = data["local"]["normal"]
    cos_theta = torch.abs(torch.sum(u*v,dim=-1,keepdim=True))
    sin_theta = torch.abs(torch.norm(torch.cross(u, v),dim=-1,keepdim=True))
    tan_theta = sin_theta / (cos_theta + 1e-3)
    data["local"]["tanh_shadow"] = tan_theta

    #exit()
    lpos = -to_light[..., :3] * to_light[..., 3:4]
    zero_mask = (data["local"]["pixel_emitter_distance"] == 0).expand_as(lpos)
    data["local"]["relative"] = torch.where(data["local"]["pixel_emitter_distance"]<0.01,1,data["local"]["pixel_emitter_distance"])
    data["local"]["relative"] = data["local"]["relative"] * data["local"]["relative"]
    data["local"]["lposition"] = torch.where(zero_mask, torch.ones_like(lpos) * 10, lpos)
    data["local"]["lenth"] = torch.norm(data["local"]["lposition"], dim=-1, keepdim=True)
    data["local"]["albedo_mask"] = data["local"]["albedo"] == 0
    data["local"]["specular_mask"] = data["local"]["specular"] == 0
    data["global"]["max_scale"] = data["global"]["max_scale"].unsqueeze(1).unsqueeze(1)
    data["local"]["shadow_mask"] = (data["local"]["diffuse_direct_shading"] * (data["local"]["albedo"] + 1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["specular"] + 1e-3)) < 1e-4
    for key in ["diffuse_direct_shading", "specular_direct_shading"]:
        data["local"][key] = torch.where(data["local"][key]<0,0,data["local"][key])
        if len(data["local"][key].shape) > 4:
            data["local"][key] = data["local"][key][:,0,...]
        data["local"][key] /= data["global"]["max_scale"]
        
        if post_relative:
            data["local"][key] = data["local"][key] *  data["local"]["relative"]
        data["local"][f"log1p_{key}"] = torch.log1p(data["local"][key])
    data["local"]["shadow"] = torch.clamp(data["local"]["shadow"],0,1)
    data["local"]["direct_shadow_shading"] = data["local"]["shadow"] * (data["local"]["diffuse_direct_shading"] * (data["local"]["albedo"]+1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["specular"]+1e-3))
    data["local"]["log1p_direct_shadow_shading"] = torch.log1p(data["local"]["direct_shadow_shading"])
    data["local"]["direct_shading"] = (data["local"]["diffuse_direct_shading"] * (data["local"]["albedo"]+1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["specular"]+1e-3))
    data["local"]["log1p_direct_shading"] = torch.log1p(data["local"]["direct_shading"])
    if indirect:
        for key in ["diffuse_indirect_shading", "specular_indirect_shading"]:
            data["local"][key] = torch.where(data["local"][key]<0,0,data["local"][key])
            if len(data["local"][key].shape) > 4:
                data["local"][key] = data["local"][key][:,0,...]
            data["local"][key] /= data["global"]["max_scale"]
            data["local"][f"log1p_{key}"] = torch.log1p(data["local"][key])
        data["local"]["indirect_shading"] = data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]
        data["local"]["log1p_indirect_shading"] = torch.log1p(data["local"]["diffuse_indirect_shading"]+data["local"]["specular_indirect_shading"])
    pos_sum = data["local"]["position"].sum(dim=-1, keepdim=True) == 0
    pos_mask = pos_sum.expand_as(data["local"]["position"])
    data["local"]["position_mask"] = pos_mask
    data["local"]["roughness"] = data["local"]["roughness"][..., :1]
    data["local"]["roughness_mask"] = data["local"]["roughness"] > 0.75


    instance_mask_raw = (data["local"]["pixel_emitter_distance"] < (data["local"]["occluder_emitter_distance"] - 0.01)) | (data["local"]["occluder_emitter_distance"]==0)
    mask = instance_mask_raw.expand_as(data["local"]["normal"])

    data["local"]["mask"] = mask | pos_mask | zero_mask 
    data["local"]["instance_mask"] = mask
    
    if  indirect:
        data["local"]["light_direction"] = data["local"]["light_direction"][:,0]
        data["local"]["light_view_distance"] = downsample_tensor_half(data["local"]["light_view_distance"])
        data["local"]["light_albedo"] = downsample_tensor_half(data["local"]["light_albedo"])
        data["local"]["light_normal"] = downsample_tensor_half(data["local"]["light_normal"])
        data["local"]["light_position"] = data["local"]["light_view_distance"] * data["local"]["light_direction"]
      

    #print(f"⚡ [forward_data_process] 耗时: {(end_time - start_time) * 1000:.3f} ms")
    return data


def video_process_tensor_torch_lightformer(data, indirect=False,post_relative=False,voxel=False):
    eps = 1e-5
    # 1. Normalize light_dir
    light_dir = data["local"]["light_dir"]
    dir_length = torch.norm(light_dir, dim=-1, keepdim=True).clamp(min=eps)
    data["local"]["light_dir"] = light_dir / dir_length
    to_light = torch.cat([data["local"]["light_dir"], data["local"]["pixel_emitter_distance"]], dim=-1)
    lpos = -to_light[..., :3] * to_light[..., 3:4]
    data["local"]["toLight"] = to_light
    u = data["local"]["light_dir"]
    v = data["local"]["normal"]
    cos_theta = torch.abs(torch.sum(u*v,dim=-1,keepdim=True))
    sin_theta = torch.abs(torch.norm(torch.cross(u, v),dim=-1,keepdim=True))
    tan_theta = sin_theta / (cos_theta + 1e-3)
    data["local"]["tanh_shadow"] = tan_theta
    scale = data["global"]["max_scale"] + 1e-4
    data["global"]["radiance"] = data["global"]["radiance"] / scale
    #exit()
    
    zero_mask = (data["local"]["pixel_emitter_distance"] == 0).expand_as(lpos)
    data["local"]["lposition"] = torch.where(zero_mask, torch.ones_like(lpos) * 10, lpos)
    data["local"]["lenth"] = torch.norm(data["local"]["lposition"], dim=-1, keepdim=True)

    data["local"]["shadow_mask"] = (data["local"]["diffuse_direct_shading"] * (data["local"]["albedo"] + 1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["specular"] + 1e-3)) < 1e-4

    data["local"]["shadow"] = torch.clamp(data["local"]["shadow"],0,1)
    data["local"]["direct_shading"] = (data["local"]["diffuse_direct_shading"] * (data["local"]["albedo"]+1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["specular"]+1e-3)) / scale
    data["local"]["log1p_direct_shading"] = torch.log1p(data["local"]["direct_shading"])

    data["local"]["indirect_shading"] = (data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]) / scale
    data["local"]["log1p_indirect_shading"] = torch.log1p(data["local"]["indirect_shading"])
    pos_sum = data["local"]["position"].sum(dim=-1, keepdim=True) == 0
    pos_mask = pos_sum.expand_as(data["local"]["position"])
    data["local"]["position_mask"] = pos_mask
    data["local"]["roughness"] = data["local"]["roughness"][..., :1]
    data["local"]["roughness_mask"] = data["local"]["roughness"] > 0.75
    instance_mask_raw = (data["local"]["pixel_emitter_distance"] < (data["local"]["occluder_emitter_distance"] - 0.01)) | (data["local"]["occluder_emitter_distance"]==0)
    mask = instance_mask_raw.expand_as(data["local"]["normal"])

    data["local"]["mask"] = mask | pos_mask | zero_mask 
    data["local"]["instance_mask"] = mask | zero_mask
    
    if  True:
        data["local"]["light_direction"] = data["local"]["light_direction"][:,0]
        data["local"]["light_view_distance"] = downsample_tensor_half(data["local"]["light_view_distance"])
        data["local"]["light_albedo"] = downsample_tensor_half(data["local"]["light_albedo"])
        data["local"]["light_normal"] = downsample_tensor_half(data["local"]["light_normal"])
        data["local"]["light_position"] = data["local"]["light_view_distance"] * data["local"]["light_direction"]
      
    return data

def inverse_data_process_tensor_lightformer(data, preds, diffuse=True, specular=False, shadow=False, indirect=False, indirect_direct=False, channel_cut=True, post_relative=False):
    """
    PyTorch 版 shadow_process
    ------------------------
    - 入参 / 返参结构与原函数保持一致
    - 默认所有字段均为 torch.Tensor（除非本来就是标量 / list）
    """
    # ================= ⏱️ 计时开始 =================
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    # ===============================================


    if indirect:
        data["local"]["lights"]["shadow"]["light_albedo"] = data["local"]["lights"]["shadow"]["light_albedo"] 
    
    direct = diffuse & specular
    direct_shadow_bool = direct & shadow 
    shading_bool = direct_shadow_bool & indirect
    scale = data["global"]["max_scale"]
    

    preds["direct_shading"] =    torch.expm1(preds["log1p_direct_shading"]) * scale
    preds["indirect_shading"] =    torch.expm1(preds["log1p_indirect_shading"]) * scale
        
    preds["shading"] = preds["direct_shading"] * preds["shadow"] + preds["indirect_shading"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    return None
inverse_key_list = ["log1p_diffuse_direct_shading","log1p_specular_direct_shading","shadow"]

def inverse_data_process_tensor(data, preds, diffuse=True, specular=False, shadow=False, indirect=False, indirect_direct=False, channel_cut=True, post_relative=False):
    """
    PyTorch 版 shadow_process
    ------------------------
    - 入参 / 返参结构与原函数保持一致
    - 默认所有字段均为 torch.Tensor（除非本来就是标量 / list）
    """
    # ================= ⏱️ 计时开始 =================
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    # ===============================================


    if indirect:
        data["local"]["lights"]["shadow"]["light_albedo"] = data["local"]["lights"]["shadow"]["light_albedo"] 
    
    direct = diffuse & specular
    direct_shadow_bool = direct & shadow 
    shading_bool = direct_shadow_bool & indirect
    scale = data["global"]["max_scale"]
    # print(preds.keys())
    # exit()
    # radiance 的归一化 / demodulation
    if diffuse and "log1p_diffuse_direct_shading" in preds.keys():
        data["local"]["diffuse_direct_shading"]   = torch.expm1(data["local"]["log1p_diffuse_direct_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        preds["diffuse_direct_shading"] = torch.expm1(preds["log1p_diffuse_direct_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        if post_relative:
            data["local"]["diffuse_direct_shading"] = data["local"]["diffuse_direct_shading"]/ data["local"]["relative"]
            preds["diffuse_direct_shading"] = preds["diffuse_direct_shading"]/data["local"]["relative"]
        
    if specular and "log1p_specular_direct_shading" in preds.keys():
        data["local"]["specular_direct_shading"]  = torch.expm1(data["local"]["log1p_specular_direct_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        preds["specular_direct_shading"] = torch.expm1(preds["log1p_specular_direct_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        if post_relative:
            data["local"]["specular_direct_shading"] = data["local"]["specular_direct_shading"]/data["local"]["relative"]
            preds["specular_direct_shading"] = preds["specular_direct_shading"]/data["local"]["relative"]
    if diffuse and specular:
        preds["direct_shading"] = preds["diffuse_direct_shading"] + preds["specular_direct_shading"]

        
    if shadow:
        data["local"]["shadow"] = data["local"]["shadow"]
        preds["shadow"] = preds["shadow"]
        # if not diffuse:
        #     preds["direct_shadow_shading"] = preds["shadow"] * (data["local"]["diffuse_direct_shading"] * (data["local"]["gbuffer"]["albedo"] + 1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["gbuffer"]["specular"] + 1e-3))

    if direct_shadow_bool:
        preds["direct_shadow_shading"] = preds["direct_shading"] * preds["shadow"]
        data["local"]["direct_shadow_shading"] = data["local"]["direct_shading"] * data["local"]["shadow"] * data["global"]["max_scale"]
        
    if indirect:
        data["local"]["diffuse_indirect_shading"] = torch.expm1(data["local"]["log1p_diffuse_indirect_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        data["local"]["demodulate_diffuse_indirect_shading"] = torch.expm1(data["local"]["log1p_diffuse_indirect_shading"]) * scale
        data["local"]["specular_indirect_shading"] = torch.expm1(data["local"]["log1p_specular_indirect_shading"]) * scale * (data["local"]["gbuffer"]["specular"] + 1e-3)
        data["local"]["demodulate_specular_indirect_shading"] = torch.expm1(data["local"]["log1p_specular_indirect_shading"]) * scale 
        preds["diffuse_indirect_shading"] = torch.expm1(preds["log1p_diffuse_indirect_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        preds["specular_indirect_shading"] = torch.expm1(preds["log1p_specular_indirect_shading"]) * scale * (data["local"]["gbuffer"]["specular"] + 1e-3)
        data["local"]["albedo"] =  (data["local"]["gbuffer"]["albedo"] + 1e-3)
        data["local"]["specular"] =  (data["local"]["gbuffer"]["specular"] + 1e-3)
        data["local"]["indirect_shading"] = data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]
        preds["indirect_shading"] = preds["diffuse_indirect_shading"] + preds["specular_indirect_shading"]
        

        
    if shading_bool:
        data["local"]["shading"] = data["local"]["direct_shadow_shading"] + data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]
        preds["shading"] = preds["direct_shadow_shading"] + preds["diffuse_indirect_shading"] + preds["specular_indirect_shading"]

    # ================= ⏱️ 计时结束 =================
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    #print(f"⚡ [inverse_data_process] 耗时: {(end_time - start_time) * 1000:.3f} ms")
    # ===============================================

    return None
def yan_video_process_tensor_torch(data, indirect=False,post_relative=False,voxel=False):
    eps = 1e-5

    # 1. Normalize light_dir
    light_dir = data["local"]["light_dir"]
    dir_length = torch.norm(light_dir, dim=-1, keepdim=True).clamp(min=eps)
    data["local"]["light_dir"] = light_dir / dir_length

    # 2. Compute toLight and lposition
    to_light = torch.cat([data["local"]["light_dir"], data["local"]["pixel_emitter_distance"]], dim=-1)
    data["local"]["toLight"] = to_light
    u = data["local"]["light_dir"]
    v = data["local"]["normal"]
    cos_theta = torch.abs(torch.sum(u*v,dim=-1,keepdim=True))
    sin_theta = torch.abs(torch.norm(torch.cross(u, v),dim=-1,keepdim=True))
    tan_theta = sin_theta / (cos_theta + 1e-3)
    data["local"]["tanh_shadow"] = tan_theta

    #exit()
    lpos = -to_light[..., :3] * to_light[..., 3:4]
    zero_mask = (data["local"]["pixel_emitter_distance"] == 0).expand_as(lpos)
    data["local"]["relative"] = torch.where(data["local"]["pixel_emitter_distance"]<1,1,data["local"]["pixel_emitter_distance"])
    data["local"]["relative"] = data["local"]["relative"] * data["local"]["relative"]
    data["local"]["lposition"] = torch.where(zero_mask, torch.ones_like(lpos) * 10, lpos)
    data["local"]["lenth"] = torch.norm(data["local"]["lposition"], dim=-1, keepdim=True)
    # 3. Create albedo and specular masks
    data["local"]["albedo_mask"] = data["local"]["albedo"] == 0
    data["local"]["specular_mask"] = data["local"]["specular"] == 0
    data["global"]["max_scale"] = data["global"]["max_scale"].unsqueeze(1).unsqueeze(1)
    # 4. Normalize radiance and shading
    #max_scale = (data["global"]["radiance"].reshape(-1, 3).max(dim=0).values + 1e-3) / 50
    
    
    # print("radiance_max : ",data["global"]["radiance"].max())
    # print("radiance_min : ",data["global"]["radiance"].min())
    for key in ["diffuse_direct_shading", "specular_direct_shading"]:
        data["local"][key] = torch.where(data["local"][key]<0,0,data["local"][key])
        if len(data["local"][key].shape) > 4:
            data["local"][key] = data["local"][key][:,0,...]
        # print(data["global"]["max_scale"].shape)
        # print(data["local"][key].shape)
        data["local"][key] /= data["global"]["max_scale"]
        
        if post_relative:
            data["local"][key] = data["local"][key] *  data["local"]["relative"]
        data["local"][f"log1p_{key}"] = torch.log1p(data["local"][key])
    data["local"]["shadow"] = torch.clamp(data["local"]["shadow"],0,1)

    #data["local"]["direct_shadow_shading"] = data["local"]["shadow"] * (data["local"]["diffuse_direct_shading"] + data["local"]["specular_direct_shading"])
    #pyexr.write("./direct_shadow_shading.exr",data["local"]["direct_shadow_shading"][0][0].cpu().numpy())
    data["local"]["direct_shadow_shading"] = data["local"]["shadow"] * (data["local"]["diffuse_direct_shading"] * data["local"]["albedo"]+ data["local"]["specular_direct_shading"] * data["local"]["specular"])
    #pyexr.write("./direct_shadow_shading2.exr",data["local"]["direct_shadow_shading"][0][0].cpu().numpy())
    data["local"]["log1p_direct_shadow_shading"] = torch.log1p(data["local"]["direct_shadow_shading"])
        # if torch.any(data["local"][f"log1p_{key}"].isnan()):
        #     pyexr.write("./nan.exr",data["local"][f"log1p_{key}"][0].cpu().numpy())
        #     exit()
    
    if indirect:
        for key in ["diffuse_indirect_shading", "specular_indirect_shading"]:
            data["local"][key] = torch.where(data["local"][key]<0,0,data["local"][key])
            # print(key,data["local"][key].shape)
            # pyexr.write("./{}ggg.exr".format(key),data["local"][key][0,0,0,...].cpu().numpy())
            # pyexr.write("./albedo.exr".format(key),data["local"]["albedo"][0,0,...].cpu().numpy())
            if len(data["local"][key].shape) > 4:
                data["local"][key] = data["local"][key][:,0,...]
            # if post_relative:
            #     data["local"][key] = data["local"][key] *  data["local"]["relative"]
            # print(key,data["local"][key].shape)
            # print(data["global"]["max_scale"].shape)
            data["local"][key] /= data["global"]["max_scale"]
            data["local"][f"log1p_{key}"] = torch.log1p(data["local"][key])
        data["local"]["indirect_shading"] = data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]
        data["local"]["log1p_indirect_shading"] = torch.log1p(data["local"]["diffuse_indirect_shading"]+data["local"]["specular_indirect_shading"])
        #exit()
    # 5. Position mask
    pos_sum = data["local"]["position"].sum(dim=-1, keepdim=True) == 0
    pos_mask = pos_sum.expand_as(data["local"]["position"])
    data["local"]["position_mask"] = pos_mask
    data["local"]["roughness"] = data["local"]["roughness"][..., :1]
    data["local"]["roughness_mask"] = data["local"]["roughness"] > 0.75
    
    # 6. Instance + normal mask
    if data["local"]["instance"].max() <= 0:
        normal_z = (data["local"]["normal"][..., 2:3] != -1).expand_as(data["local"]["normal"])
        to_light_mask = (to_light[..., 3:4] < 0.65).expand_as(data["local"]["normal"])
        mask = normal_z & to_light_mask
    else:
        instance_mask_raw = data["local"]["instance"] >= data["local"]["instance_mask"]
        mask = instance_mask_raw.expand_as(data["local"]["normal"])

    data["local"]["mask"] = mask | pos_mask | zero_mask 
    data["local"]["instance_mask"] = mask | zero_mask

    # 7. Reduce roughness channel
    
    
    if indirect:
        shape = data["local"]["light_albedo"].shape
        data["local"]["light_albedo"] = F.interpolate(data["local"]["light_albedo"].reshape(shape[0] ,shape[1],shape[2],shape[3]).permute(0,3,1,2), size=(384, 64), mode='bilinear', align_corners=False).permute(0,2,3,1).reshape(shape[0],384,64,shape[-1])
        data["local"]["light_normal"] = F.interpolate(data["local"]["light_normal"].reshape(shape[0] ,shape[1],shape[2],shape[3]).permute(0,3,1,2), size=(384, 64), mode='bilinear', align_corners=False).permute(0,2,3,1).reshape(shape[0],384,64,-1)
        data["local"]["light_depth"] = F.interpolate(data["local"]["light_view_distance"].reshape(shape[0],shape[1],shape[2],1).permute(0,3,1,2), size=(384, 64), mode='bilinear', align_corners=False).permute(0,2,3,1).reshape(shape[0],384,64,1)
        data["local"]["light_position"] = data["local"]["light_depth"] * data["local"]["light_direction"]
        data["local"]["light_ws_position"] = data["local"]["light_depth"] * data["local"]["light_direction"] +  data["local"]["light_pos"].unsqueeze(1).unsqueeze(1)
        square_lenth = torch.sum(data["local"]["light_position"] * data["local"]["light_position"],dim=-1).unsqueeze(dim=-1)
        view_dir = F.normalize(data["local"]["camera_pos"].unsqueeze(1).unsqueeze(1) - data["local"]["light_pos"].unsqueeze(1).unsqueeze(1) - data["local"]["light_position"],dim=-1)
        data["local"]["light_half_vec"] = F.normalize(
            (F.normalize(-data["local"]["light_position"],dim=-1) + view_dir) / 2,dim=-1)
        data["local"]["light_dot"] = torch.sum(
            data["local"]["light_half_vec"] * data["local"]["light_normal"], dim=-1, keepdim=True)
        data["local"]["light_roughness"] = torch.zeros_like(data["local"]["light_dot"]) + 0.0
        data["local"]["light_view_dir"] = view_dir
        data["local"]["light_specular_ray"] = torch.sum(data["local"]["light_view_dir"] * data["local"]["light_normal"],
                                         dim=-1).unsqueeze(-1) * data["local"]["light_normal"] * 2 - \
                               data["local"]["light_view_dir"]
        N = F.normalize(data["local"]["light_normal"], dim=-1)
        L = F.normalize(-data["local"]["light_direction"], dim=-1)
        cos_theta = torch.sum(N * L, dim=-1, keepdim=True).clamp(min=0.0)  # (B,V,H,W,1)
        rho = data["local"]["light_albedo"]  # (B,V,H,W,3)
        r2 = square_lenth.clamp(min=1e-6)   # 避免除 0
        light_flux = rho * cos_theta 
        data["local"]["light_flux"] = light_flux
        data["local"]["light_cos"] = cos_theta
    if voxel:
        data["local"]["voxelRoughness"] = data["local"]["voxelNormal"][...,3:4] 
        data["local"]["voxelFresnel"] = data["local"]["voxelSpecular"][...,3:4] 
        data["local"]["voxelDiffuse"] = data["local"]["voxelDiffuse"][...,:3]
        data["local"]["voxelSpecular"] = data["local"]["voxelSpecular"][...,:3]
        data["local"]["voxelOcclusion"] = torch.cat([data["local"]["voxelPosition"][...,:2],data["local"]["voxelPosition"][...,3:4]],dim=-1)
        data["local"]["voxelPosition"] = data["local"]["voxelPosition"][...,:3]
        data["local"]["voxelNormal"] = data["local"]["voxelNormal"][...,:3]
        data["local"]["voxelMask"] = torch.sum(data["local"]["voxelNormal"], dim=-1, keepdim=True) == 0
        
        # print(data["local"]["camera_pos"].shape)
        # print(data["local"]["voxelPosition"].shape)
        #exit()
        # data["local"]["voxelToCamera"] = (data["local"]["camera_pos"].unsqueeze(1).unsqueeze(1) - data["local"]["voxelPosition"])
        # data["local"]["voxelMask"] = torch.sum(data["local"]["voxelPosition"],dim=-1,keepdim=True)==0
        
        # #exit()
        # cnorm = data["local"]["voxelToCamera"].norm(dim=-1, keepdim=True)  # shape: (B, W, H, 1)
        # cnorm = torch.clamp(cnorm, min=1e-8)
        # data["local"]["voxelToCamera"]  = data["local"]["voxelToCamera"] / cnorm
        # data["local"]["voxelToLight"] = data["local"]["light_pos"].unsqueeze(1).unsqueeze(1) - data["local"]["voxelPosition"]
        # lnorm = data["local"]["voxelToLight"].norm(dim=-1, keepdim=True)  # shape: (B, W, H, 1)
        # lnorm = torch.clamp(lnorm, min=1e-8)
        # data["local"]["voxelToLight"]  = data["local"]["voxelToLight"] / lnorm
        # data["local"]["voxelPosition"] = data["local"]["voxelPosition"] - data["local"]["light_pos"].unsqueeze(1).unsqueeze(1)
            # for key in data["local"]:
            #     print(key,data["local"][key].shape)
            #exit()

        #data["local"]["light_albedo"] = data["local"]["light_albedo"] * square_lenth
        
    return data





def yan_inverse_data_process_tensor(data, preds,diffuse=True,specular=False,shadow=False,indirect=False,indirect_direct=False,channel_cut= True,post_relative=False):
    """
    PyTorch 版 shadow_process
    ------------------------
    - 入参 / 返参结构与原函数保持一致
    - 默认所有字段均为 torch.Tensor（除非本来就是标量 / list）
    """
    
    data["local"]["gbuffer"]["albedo"] = data["local"]["gbuffer"]["albedo"] 
    data["local"]["gbuffer"]["specular"] = data["local"]["gbuffer"]["specular"] 
    if indirect:
        data["local"]["lights"]["shadow"]["light_albedo"] = data["local"]["lights"]["shadow"]["light_albedo"] 
    direct = diffuse & specular
    direct_shadow_bool = direct & shadow 
    shading_bool = direct_shadow_bool & indirect
    scale = data["global"]["max_scale"]
    
    # radiance 的归一化 / demodulation
    if diffuse:
        data["local"]["diffuse_direct_shading"]   = torch.expm1(data["local"]["log1p_diffuse_direct_shading"]) * scale * data["local"]["gbuffer"]["albedo"]
        preds["diffuse_direct_shading"] = torch.expm1(preds["log1p_diffuse_direct_shading"]) * scale * data["local"]["gbuffer"]["albedo"]
        if post_relative:
            data["local"]["diffuse_direct_shading"] = data["local"]["diffuse_direct_shading"]/ data["local"]["relative"]
            preds["diffuse_direct_shading"] = preds["diffuse_direct_shading"]/ data["local"]["relative"]
        
    if specular:
        data["local"]["specular_direct_shading"]  = torch.expm1(data["local"]["log1p_specular_direct_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        preds["specular_direct_shading"] = torch.expm1(preds["log1p_specular_direct_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        if post_relative:
            data["local"]["specular_direct_shading"] = data["local"]["specular_direct_shading"]/data["local"]["relative"]
            preds["specular_direct_shading"] = preds["specular_direct_shading"]/ data["local"]["relative"]
    if direct:
        data["local"]["direct_shading"]   = data["local"]["diffuse_direct_shading"] + data["local"]["specular_direct_shading"]
        preds["direct_shading"] =   preds["diffuse_direct_shading"] +  preds["specular_direct_shading"]
    if shadow:
        data["local"]["shadow"] = data["local"]["shadow"]
        preds["shadow"] = preds["shadow"]

    if direct_shadow_bool and (indirect_direct == False) :
        preds["direct_shadow_shading"] = preds["direct_shading"] * preds["shadow"] 
        data["local"]["direct_shadow_shading"] = data["local"]["direct_shading"] * data["local"]["shadow"] 
    if indirect:
        data["local"]["diffuse_indirect_shading"] = torch.expm1(data["local"]["log1p_diffuse_indirect_shading"]) * scale * data["local"]["gbuffer"]["albedo"]
        data["local"]["specular_indirect_shading"] = torch.expm1(data["local"]["log1p_specular_indirect_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        preds["diffuse_indirect_shading"] = torch.expm1(preds["log1p_diffuse_indirect_shading"]) * scale * data["local"]["gbuffer"]["albedo"]
        preds["specular_indirect_shading"] = torch.expm1(preds["log1p_specular_indirect_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        # if post_relative:
        #     data["local"]["diffuse_indirect_shading"] = data["local"]["diffuse_indirect_shading"] / data["local"]["relative"]
        #     data["local"]["specular_indirect_shading"] = data["local"]["specular_indirect_shading"] / data["local"]["relative"]
        #     preds["diffuse_indirect_shading"] = preds["diffuse_indirect_shading"] / data["local"]["relative"]
        #     preds["specular_indirect_shading"] = preds["specular_indirect_shading"] / data["local"]["relative"]
        preds["demodulate_diffuse_indirect_shading"] = torch.expm1(preds["log1p_diffuse_indirect_shading"]) * scale 
        preds["demodulate_specular_indirect_shading"] = torch.expm1(preds["log1p_diffuse_indirect_shading"]) * scale 
        data["local"]["demodulate_diffuse_indirect_shading"] = torch.expm1(data["local"]["log1p_diffuse_indirect_shading"]) * scale 
        data["local"]["demodulate_specular_indirect_shading"] = torch.expm1(data["local"]["log1p_diffuse_indirect_shading"]) * scale 
        data["local"]["indirect_shading"] = data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]
        preds["indirect_shading"] = preds["diffuse_indirect_shading"] + preds["specular_indirect_shading"]
    if indirect_direct:
        
        preds["direct_shadow_shading"] = torch.expm1(preds["log1p_direct_shadow_shading"]) * scale /  data["local"]["relative"]
        data["local"]["direct_shadow_shading"] = torch.expm1(data["local"]["log1p_direct_shadow_shading"]) * scale /  data["local"]["relative"]
        
    if shading_bool:
        data["local"]["shading"] = data["local"]["direct_shadow_shading"] + data["local"]["diffuse_indirect_shading"] + data["local"]["specular_indirect_shading"]
        preds["shading"] = preds["direct_shadow_shading"] +preds["diffuse_indirect_shading"] + preds["specular_indirect_shading"]
    

    return None

def shadow_process_torch(data, scale: bool = True, is_demodulate: bool = True):
    """
    PyTorch 版本的 shadow_process。
    说明：
    - 仍按 H×W×C 来处理（[..., channel]）。
    - 保持与原 NumPy 版本一致的数值流程（包括两次 /50 的缩放行为）。
    - 所有 where/clip/gradient 等操作都用 torch 等价实现。
    """

    # 简单别名
    local = data["local"]
    global_ = data["global"]

    eps = 1e-3
    device = local["albedo"].device
    dtype  = local["albedo"].dtype

    # 初始化/掩码
    albedo_mask   = (local["albedo"] == 0)
    specular_mask = (local["specular"] == 0)
    local["albedo_mask"]   = albedo_mask
    local["specular_mask"] = specular_mask
    to_light = torch.cat([data["local"]["light_dir"], data["local"]["pixel_emitter_distance"]], dim=-1)
    data["local"]["toLight"] = to_light

    lpos = -to_light[..., :3] * to_light[..., 3:4]
    zero_mask = (data["local"]["pixel_emitter_distance"] == 0).expand_as(lpos)
    data["local"]["lposition"] = torch.where(zero_mask, torch.ones_like(lpos) * 10, lpos)
    data["local"]["lenth"] = torch.norm(data["local"]["lposition"], dim=-1, keepdim=True)
    # 防止除零
    local["specular"] = local["specular"] + eps
    local["albedo"]   = local["albedo"]   + eps

    # 归一化缩放
    if scale:
        # 注意：保持与原始逻辑一致，max_scale 已含一次 /50
        max_scale_vals = global_["radiance"].reshape(-1, 3).max(dim=0).values / 50.0
        global_["max_scale"] = max_scale_vals

        # radiance 再除一次 /50（与原代码保持一致：总计会 /50 两次）
        global_["radiance"] = global_["radiance"] / global_["max_scale"] / 50.0

        # 局部分量缩放
        local["direct_shading"]   = local["direct_shading"]   / global_["max_scale"]
        local["specular_direct_shading"] = local["specular_direct_shading"] / global_["max_scale"]

        # albedo 为 0 的位置把 indirect_shading 置 0
        local["indirect_shading"] = torch.where(
            albedo_mask, torch.zeros_like(local["indirect_shading"]), local["indirect_shading"]
        )

    # relative = clamp((toLight[...,3]^2), 1e-4, 1e9)
    # relative = (local["toLight"][:, :, 3] ** 2.0).clamp(min=1e-4, max=1e9)
    # local["relative"] = relative.unsqueeze(-1)

    # 间接输入（保持逻辑一致）
    local["indirect_input_shading"] = local["direct_shading"] * local["shadow"]

    # diffuse_direct_shading / direct_no_shading / specular_direct_shading 的重排与归一
    local["diffuse_direct_shading"] = local["diffuse_direct_shading"] / local["albedo"]

    local["direct_no_shading"] = local["direct_shading"] 

    local["specular_direct_shading"] = local["specular_direct_shading"] / local["specular"]

    # 掩码与非负裁剪
    local["diffuse_direct_shading"] = torch.where(
        albedo_mask, torch.zeros_like(local["diffuse_direct_shading"]), local["diffuse_direct_shading"]
    )
    local["specular_direct_shading"] = torch.where(
        specular_mask, torch.zeros_like(local["specular_direct_shading"]), local["specular_direct_shading"]
    )
    local["diffuse_direct_shading"]  = torch.clamp(local["diffuse_direct_shading"],  min=0)
    local["specular_direct_shading"] = torch.clamp(local["specular_direct_shading"], min=0)

    # log1p 特征
    local["log1p_diffuse_direct_shading"]  = torch.log1p(local["diffuse_direct_shading"])
    local["log1p_specular_direct_shading"] = torch.log1p(local["specular_direct_shading"])
    local["log1p_indirect_shading"] = torch.log1p(local["indirect_shading"])

    # 深度梯度（对 light_view_distance[...,0] 做二维梯度）
    # torch.gradient 返回与输入同形状的梯度张量（或列表）
    depth0 = local["light_view_distance"][..., 0]
    # 指定在 H、W 两个维度上取梯度（假设张量最后两维是 H、W；若布局为 H,W,... 则用 dim=(0,1)）
    # 这里 depth0 是 H×W，按 (0,1) 求梯度

    # 折射/位置掩码

    position_sum  = local["position"].sum(dim=-1).eq(0).unsqueeze(-1)
    position_mask = position_sum.repeat(1,1, 1, 3)

    # 实例掩码/可见性掩码
    if local["instance"].max() <= 0:
        normal_z_valid = (local["normal"][..., 2:3] != -1).repeat(1,1, 1, 3)
        vis_mask = (local["pixel_emitter_distance"] < 0.65).repeat(1,1, 1, 3)
        local["mask"] = vis_mask & normal_z_valid
        local["instance_mask"] = local["mask"]
    else:
        inst_cmp = (local["instance"] >= local["instance_mask"]).unsqueeze(-1) \
                   if local["instance"].ndim == 2 else (local["instance"] >= local["instance_mask"])
        # 保证是 H×W×1 再扩到 3 通道
        if inst_cmp.ndim == 2:
            inst_cmp = inst_cmp.unsqueeze(-1)
        local["mask"] = inst_cmp.repeat(1,1, 1, 3)
        local["instance_mask"] = local["mask"]

    # 位置为 0 的地方并到 mask 中
    local["mask"] = local["mask"] | position_mask
    local["position_mask"] = position_mask

    # 粗糙度/菲涅尔通道拆分
    local["roughness"]   = local["roughness"][..., :1]

    data["local"]  = local
    data["global"] = global_
    return data

class VideoDatasets(Dataset):
    def __init__(self, configs, datasetsName):
        self.data_type = ".zst"
        self.configs = configs
        resolution = str(configs["light_angular_resolution"]) + "x" + str(configs["light_direction_resolution"])
        self.read_indirect = configs["indirect"]
        self.datasetsName = datasetsName
        self.dataDict = {}
        self.fileList = {}
    
        self.videoLightPath = r"../datasets2/{}{}".format(configs["light"],resolution)
        self.light_mid = configs["light"]
        self.imageLightPath = r"../datasets2/standard_light{}".format(resolution)
        self.videoRootDir = r"../datasets2/" + self.datasetsName
        self.imageRootDir = r"../datasets2/final_special/"  
        self.can_read_image = False
        self.can_read_video = True
        with open("../datasets2/bias_info.json","r") as file:
            self.light_bias = json.load(file)
        file.close()
        
        if os.path.exists(self.imageRootDir + "denoise/") and os.path.exists(self.imageLightPath):
            self.imageFileList = os.listdir(self.imageRootDir + "denoise/")
            self.nodenoiseList = os.listdir(self.imageRootDir)
            self.imageFileList = list(set(self.imageFileList) & set(self.nodenoiseList))
            self.imageLightFileList = os.listdir(self.imageLightPath)
            self.can_read_image = True
        if os.path.exists(self.videoRootDir + "/direct/denoise/") and os.path.exists(self.videoLightPath):
            # print("yes")
            # print(exit())
            self.videoFileList = os.listdir(self.videoRootDir + "/direct/denoise/")
            self.nodenoiseList = os.listdir(self.videoRootDir + "/direct")
            self.videoFileList = list(set(self.videoFileList) & set(self.nodenoiseList))
            if self.read_indirect:
                with open(self.videoRootDir + "/light_position.json","r") as file:
                    self.video_position_data = json.load(file)   
                file.close()
                self.videoFileList = list(set(self.videoFileList) & set(self.video_position_data["camera"].keys()))
            print(len(self.videoFileList))
            self.videoLightFileList = os.listdir(self.videoLightPath)
            print(len(self.videoLightFileList))
            # exit()
            self.can_read_video = True     
            
        self.init_datasets()
        self.read_image = False
        self.dict_to_indices = {}
        for i in range(len(self.fileList)):
            self.dict_to_indices[self.fileList[i]] =  i

        
        self.importance_sampling =False
        self.n = len(self.fileList)
        self.loss_values = np.ones(self.n, dtype=np.float32) # 初始loss全为1
        self.probs = np.ones(self.n, dtype=np.float32) / self.n  # 初始均匀采样

        self.light_dir = pyexr.read(r"../datasets/indirect_dir.exr")

    def init_datasets(self):
        if self.can_read_image:
            file_list = []
            for file in self.imageFileList:
                if file.split(".")[-1] !="zst":
                    continue
                scene,light,_ = file.split(".")[0].split("_")

                if (light + ".pkl.zst") not in self.imageLightFileList:
                    print("light miss")
                    print(light)
                    continue
                file_list.append(file)
            self.imageFileList = file_list
            print("init image dataets from: ",self.imageRootDir)
            print("total cnt: ",len(self.imageFileList))
        if self.can_read_video:
            file_list = []
            for file in self.videoFileList:
                if file.split(".")[-1] !="zst":
                    continue
                scene,light,_,_ = file.split(".")[0].split("_")
                file_list.append(file)
            self.videoFileList = file_list
            print("init video dataets from: ",self.videoRootDir)
            print("total cnt: ",len(self.videoFileList))
        self.videoFileList = self.videoFileList[:] 


    def update_loss(self, name, losses,weight):
        """
        indices: list/tensor，训练过程中得到的样本索引
        losses: 对应样本的loss值
        """
        #print("before ",self.loss_values)
        for idx in range(len(name)):
            n = name[idx]
            l = losses[idx]
            w = weight[idx]
            i = self.dict_to_indices[n + ".pkl.zst"]
            self.loss_values[i] = l * w + 1e-4
        # 根据 loss 更新采样概率，+eps 防止除零
        
        self.probs =  self.loss_values /  self.loss_values.sum()
        # print(i)
        # print("after ",self.loss_values)

    def sample_indices(self):
        """
        按当前权重随机采样一批索引
        """
        return np.random.choice(self.n, size=1, p=self.probs)


    def __len__(self):
        self.scale = 1
        if sys.platform.startswith("win"):
            self.scale = 1
        if self.read_image:
            return int(len(self.imageFileList) * self.scale)
        else:
            return int(len(self.videoFileList)) * self.scale
        #return 1000
    def get_video_data(self, idx):
        #idx = 0
        idx = idx % 1000
        localID = self.videoFileList[idx].split(".")[0]
        #print("local ID ",localID)
        sceneID, lightID, _,_ = str.split(localID, "_")
        scene_file_name = sceneID  + r".pkl.zst"
        lightData_fn = self.videoLightPath + '/{}_final.pkl'.format(lightID)

        with open(lightData_fn + self.data_type, 'rb') as gf:
            dctx = zstd.ZstdDecompressor()
            lightData = pickle.loads(dctx.decompress(gf.read()))
        gf.close()

            # print("yes_mid")
            # print(numpy.array(self.light_bias[lightID + "_final"]) )
        lightData["position"] = lightData["position"] + numpy.array(self.light_bias[lightID + "_final"]) 
            #exit()
        
        data = {}
        directData_fn = self.videoRootDir + r'/direct/' + self.videoFileList[idx]
 
        with open(directData_fn, 'rb') as lf:
            dctx = zstd.ZstdDecompressor()
            directData = pickle.loads(dctx.decompress(lf.read()))
        lf.close()
        directDenoiseData_fn = self.videoRootDir + r'/direct/denoise/' + self.videoFileList[idx]
        with open(directDenoiseData_fn, 'rb') as lf:
            dctx = zstd.ZstdDecompressor()
            directDenoiseData = pickle.loads(dctx.decompress(lf.read()))
        lf.close()
        for key in directDenoiseData:
            directData["noise_" + key] = directData[key]
            directData[key] = directDenoiseData[key]
        if self.read_indirect:
            indirectData_fn = self.videoRootDir + r'/indirect/' + self.videoFileList[idx]
            with open(indirectData_fn, 'rb') as lf:
                dctx = zstd.ZstdDecompressor()
                indirectData = pickle.loads(dctx.decompress(lf.read()))
            lf.close()
            
            denoiseIndirectData_fn = self.videoRootDir + r'/indirect/denoise/' + self.videoFileList[idx]
            with open(denoiseIndirectData_fn, 'rb') as lf:
                dctx = zstd.ZstdDecompressor()
                denoiseIndirectData = pickle.loads(dctx.decompress(lf.read()))
            lf.close()
            indirectData["camera_pos"] = numpy.array(self.video_position_data["camera"][self.videoFileList[idx]])[numpy.newaxis,...]
            indirectData["light_pos"] = numpy.array(self.video_position_data["light"][self.videoFileList[idx]])[numpy.newaxis,...]
            
           
            data.update(indirectData)
        data.update(directData)
        
        data["light_direction"] = self.light_dir[numpy.newaxis, ...]
        return data, lightData,localID


    def get_image_data(self, idx):
        #idx = 0
        print("get image data :",idx)
        localID = self.imageFileList[idx].split(".")[0]
        sceneID, lightID,_ = str.split(localID, "_")
        
        lightData_fn = self.imageLightPath + '/{}.pkl'.format(lightID)
      
    
        with open(lightData_fn + self.data_type, 'rb') as gf:
            dctx = zstd.ZstdDecompressor()
            lightData = pickle.loads(dctx.decompress(gf.read()))
        gf.close()
        data = {}
        directData_fn = self.imageRootDir + "/"+ self.imageFileList[idx]
        print(directData_fn)
        with open(directData_fn, 'rb') as lf:
            dctx = zstd.ZstdDecompressor()
            directData = pickle.loads(dctx.decompress(lf.read()))
        lf.close()
        directDenoiseData_fn = self.imageRootDir + r'/denoise/' + self.imageFileList[idx]
        with open(directDenoiseData_fn, 'rb') as lf:
            dctx = zstd.ZstdDecompressor()
            denoiseData = pickle.loads(dctx.decompress(lf.read()))
        lf.close()
        if True:

            directData["noisy_direct_shading"] = numpy.clip(directData["direct_shading"], 1e-3, 10000)
            directData["noisy_direct_shadow_shading"] = numpy.clip(directData["direct_shading"] * directData["shadow"], 0,
                                                                   10000)
            if "denoise_direct_diffuse" in denoiseData.keys():
                directData["direct_shading"] = numpy.clip(denoiseData["denoise_direct"], 1e-4, 10000)
            directData["indirect_shading"] = denoiseData["denoise_indirect_shading"]
            if True:
                directData["shadow"] = numpy.clip((numpy.clip(denoiseData["denoise_direct_shadow"], 0, 10000)) / (
                (numpy.clip(directData["direct_shading"], 1e-3, 10000))), 0, 1)
            else:
                directData["shadow"] = numpy.clip((numpy.clip(directData["noisy_direct_shadow_shading"], 0, 10000)) / (
                (numpy.clip(directData["noisy_direct_shading"], 1e-3, 10000))), 0, 1)
                directData["denoised_shadow"] = numpy.clip((numpy.clip(denoiseData["denoise_direct_shadow"], 0, 10000)) / (
                (numpy.clip(directData["direct_shading"], 1e-3, 10000))), 0, 1)
                directData["denoised_direct_shading"] = directData["direct_shading"] * directData["denoised_shadow"] 
            directData["direct_shadow_shading"] = numpy.clip(denoiseData["denoise_direct_shadow"], 0, 10000)
            directData["specular_direct_shading"] = denoiseData["denoise_direct"] - denoiseData["denoise_direct_diffuse"]
            
            directData["diffuse_direct_shading"] = denoiseData["denoise_direct_diffuse"]
            directData["direct_shading"] = denoiseData["denoise_direct"]
            directData["noisy_shadow"] = directData["noisy_direct_shadow_shading"].astype(np.float32) / directData[
                "noisy_direct_shading"].astype(np.float32)
            camera_pos = denoiseData["camera_pos"]
            position_sum = numpy.sum(directData["position"], axis=-1) == 0
            position_sum = position_sum[..., np.newaxis]
            position_mask = np.repeat(position_sum, 3, axis=-1)
            to_camera = directData["position"] - numpy.array(camera_pos)
            directData["depth"] = numpy.sqrt(numpy.sum(to_camera * to_camera, axis=-1)[..., numpy.newaxis])
            directData["camera_pos"] = numpy.array(camera_pos)
            directData["depth"][position_mask[..., 0]] = 0
            directData["position_mask"] = position_mask

        # if self.read_indirect:
        #     indirectData_fn = self.root_dir + r'/indirect/' + self.imageFileList[idx]
        #     with open(indirectData_fn, 'rb') as lf:
        #         dctx = zstd.ZstdDecompressor()
        #         indirectData = pickle.loads(dctx.decompress(lf.read()))
        #     lf.close()
        #     data.update(indirectData)
        data.update(directData)
        
        data["light_direction"] = self.light_dir[numpy.newaxis, ...]
        return data, lightData,localID

    def __getitem__(self, idx):
        if self.importance_sampling:
            idx = self.sample_indices()[0]
        else:
            if self.read_image:
                idx = idx % int(len(self.imageFileList))
            else:
                idx = idx % int(len(self.videoFileList))
        start_time = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.read_image:
            sceneData, lightData, localID = self.get_image_data(idx)
        else:
            sceneData, lightData, localID = self.get_video_data(idx)
        time2 = time.time()
        self.idx = idx
        data = {}
        data["global"] = lightData
        data["local"] = sceneData

        # if self.read_indirect:
        #     data = eval(self.configs['preprocess'] + '(data,True)')
        # else:
        #     data = eval(self.configs['preprocess'] + '(data,False)')

        check_data_valid(data)

        return (data,localID)


def get_data(LABEL):
    # 你的数据加载代码
    data = load_pklzst(r"../datasets2/{}/final_data_64_8.pkl.zst".format(LABEL))
    # 为了演示，只取一部分，正式训练请根据需要调整
    data = dict(list(data.items())[:]) 

    lst = list(data.keys())

    d = {v: i for i, v in enumerate(lst)}
    return d
def get_filename_index_dict(directory_path):
    """
    获取指定目录下所有的文件名，并生成 {文件名: 列表下标} 的字典。
    """
    # 1. 使用 os.listdir 获取目录下所有的文件和文件夹名称列表
    file_list = os.listdir(directory_path)
    
    # 2. 使用 enumerate 获取下标和文件名，并通过字典推导式生成字典
    # index 是下标 (从 0 开始)，filename 是文件名
    file_dict = {filename: index for index, filename in enumerate(file_list)}
    
    return file_dict

class NelifDatasets(Dataset):
    """Load complete scene samples from the fixed datasets/scene directory."""

    DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    SCENE_DIR = os.path.join(DATASETS_DIR, "scene")
    LIGHT_DIR = os.path.join(DATASETS_DIR, "Light")

    def __init__(self, configs, isTest=False):
        self.configs = configs
        self.isTest = isTest
        self.read_indirect = configs["indirect"]
        self.load_tri = configs["load_tri"]
        self.isCut = configs["cut"]
        self.channel_cnt = 3
        if not self.isCut:
            self.channel_cnt = 1
        self.cache = False
        self.read_voxel = configs["voxel"]
        self.dataDict = {}
        self.fileList = {}
        self.videoLightPath = self.LIGHT_DIR
        self.videoRootDir = self.SCENE_DIR
        self.init_datasets()
        print("start datasets init")
        self.volume_res = 128
        self.dir = pyexr.read(
                os.path.join(self.DATASETS_DIR, "OutDir.exr")
            )[..., :3].reshape(
                8,
                128,
                8,
                128,
                3,
            ).transpose(
                0,
                2,
                1,
                3,
                4,
            )
        with open(
            os.path.join(
                self.DATASETS_DIR,
                "bias_info.json",
            ),
            "r",
            encoding="utf-8",
        ) as file:
            self.light_bias = json.load(file)
        self.dict_to_indices = {}
        for i in range(len(self.fileList)):
            self.dict_to_indices[self.fileList[i]] =  i
        self.importance_sampling =False
        self.n = len(self.fileList)
        self.loss_values = np.ones(self.n, dtype=np.float32) # 初始loss全为1
        self.probs = np.ones(self.n, dtype=np.float32) / self.n  # 初始均匀采样

        self.light_dir = pyexr.read(r"../datasets/indirect_dir.exr")
        self.read_light = configs["read_light"]
        if self.read_light:
            print("read light yes")


    def init_datasets(self):
        with os.scandir(self.videoRootDir) as entries:
            self.videoFileList = sorted(
                entry.name for entry in entries
                if entry.is_file() and entry.name.endswith(".pkl.zst")
            )
        if not self.videoFileList:
            raise FileNotFoundError(f"No scene .pkl.zst files found in {self.videoRootDir}")
        self.fileList = self.videoFileList

    def update_loss(self, name, losses,weight):
        """
        indices: list/tensor，训练过程中得到的样本索引
        losses: 对应样本的loss值
        """
        #print("before ",self.loss_values)
        for idx in range(len(name)):
            n = name[idx]
            l = losses[idx]
            w = weight[idx]
            i = self.dict_to_indices[n + ".pkl.zst"]
            self.loss_values[i] = l * w + 1e-4
        # 根据 loss 更新采样概率，+eps 防止除零
        
        self.probs =  self.loss_values /  self.loss_values.sum()
        # print(i)
        # print("after ",self.loss_values)

    def sample_indices(self):
        """
        按当前权重随机采样一批索引
        """
        return np.random.choice(self.n, size=1, p=self.probs)


    def __len__(self):
        self.scale = 1
        if sys.platform.startswith("win"):
            self.scale = 1
        if self.isTest:
            self.scale = 1
        #return 1000
        return int(len(self.videoFileList)) * self.scale
    def get_video_data(self, idx):
        localID = self.videoFileList[idx].split(".")[0]
        
        sceneID, lightID, _,_,_ = str.split(localID, "_")

        
        lightData = {}

        lightData_fn = self.videoLightPath + '/{}.pkl.zst'.format(lightID)
        lightData = load_pklzst(lightData_fn)
        lightData["position"] = lightData["position"] + numpy.array(self.light_bias[lightID ]["position"]) 
        radiance = numpy.asarray(lightData["radiance"])
        rgb = radiance[..., :3]
        reduce_axes = tuple(range(rgb.ndim - 1))
        max_scale = numpy.max(rgb, axis=reduce_axes)
        max_scale = numpy.where(max_scale <= 0, 1e-6, max_scale)
        lightData["max_scale"] = numpy.asarray(max_scale, dtype=radiance.dtype)
        lightData["radiance"] = radiance / lightData["max_scale"]
        lightData["direction"] = self.dir
        
        data = load_pklzst(os.path.join(self.videoRootDir, self.videoFileList[idx]))
        if self.read_indirect:
            data["light_direction"] = self.light_dir[numpy.newaxis, ...]
      
        #print("return gg")
        return data, lightData,localID


    def __getitem__(self, idx):
        if self.importance_sampling:
            idx = self.sample_indices()[0]
        else:
            idx = idx % int(len(self.videoFileList))
        start_time = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if self.isTest:
        #     sceneData, lightData, localID = self.get_video_data_test(idx)
        # else:
        sceneData, lightData, localID = self.get_video_data(idx)
        time2 = time.time()
        self.idx = idx
        data = {}
        data["global"] = lightData
        data["local"] = sceneData
        

        return (data,localID)
from pathlib import Path

class PlaneDatasets(Dataset):
    def __init__(self, configs, datasetsName, isTest = False):
        self.configs = configs
        self.isTest = isTest
        resolution = str(configs["light_angular_resolution"]) + "x" + str(configs["light_direction_resolution"])
        self.read_indirect = configs["indirect"]
        self.load_tri = configs["load_tri"]
        self.isCut = configs["cut"]
        self.channel_cnt = 3
        if not self.isCut:
            self.channel_cnt = 1
        self.cache = False
        self.datasetsName = datasetsName
        self.read_voxel = configs["voxel"]
        self.dataDict = {}
        self.fileList = {}
        self.videoLightPath = r"../datasets2/{}{}".format(configs["light"],resolution)
        self.light_mid = configs["light"]
        self.videoRootDir = r"../datasets2/" + self.datasetsName
        with open("../datasets2/TogLightAll8x128/bias_info.json","r") as file:
            self.light_bias = json.load(file)
        file.close()
        print("start datasets init")
        self.volume_res = 128
        if os.path.exists(self.videoRootDir ) and os.path.exists(self.videoLightPath):
            with open(f"../datasets2/{self.datasetsName}/good_configs.json","r") as file:
                self.videoFileList = json.load(file)
            #self.videoFileList = os.listdir(self.videoRootDir + "/voxelNew2/" )[:]

            with open("./light_max_scale.json","r") as file:
                self.light_max_scale = json.load(file)
            self.videoLightFileList = os.listdir(self.videoLightPath)

        self.init_datasets()
        self.dict_to_indices = {}
        for i in range(len(self.fileList)):
            self.dict_to_indices[self.fileList[i]] =  i
        self.importance_sampling =False
        self.n = len(self.fileList)
        self.loss_values = np.ones(self.n, dtype=np.float32) # 初始loss全为1
        self.probs = np.ones(self.n, dtype=np.float32) / self.n  # 初始均匀采样

        self.light_dir = pyexr.read(r"../datasets/indirect_dir.exr")
        self.read_light = configs["read_light"]
        self.plane_path = r"../datasets_plane/" + configs["plane_label"] + "/"
        self.read_plane = configs["plane_label"] != "none"
        if self.read_plane:
            with open(self.plane_path + "/light_to_plane.json","r") as file:
                self.lightToPlaneDict = json.load(file)
        if self.read_light:
            print("read light yes")
 


    def init_datasets(self):
        file_list = []
        light_list = self.videoLightFileList
        miss_cnt = 0
        miss_light_cnt = 0
        None
    def update_loss(self, name, losses,weight):
        """
        indices: list/tensor，训练过程中得到的样本索引
        losses: 对应样本的loss值
        """
        #print("before ",self.loss_values)
        for idx in range(len(name)):
            n = name[idx]
            l = losses[idx]
            w = weight[idx]
            i = self.dict_to_indices[n + ".pkl.zst"]
            self.loss_values[i] = l * w + 1e-4
        # 根据 loss 更新采样概率，+eps 防止除零
        
        self.probs =  self.loss_values /  self.loss_values.sum()
        # print(i)
        # print("after ",self.loss_values)

    def sample_indices(self):
        """
        按当前权重随机采样一批索引
        """
        return np.random.choice(self.n, size=1, p=self.probs)


    def __len__(self):
        self.scale = 5000
        if sys.platform.startswith("win"):
            self.scale = 5000
        if self.isTest:
            self.scale = 1
        #return 1000
        return int(len(self.videoFileList)) * self.scale
    def get_video_data(self, idx):
        localID = self.videoFileList[idx].split(".")[0]
        
        sceneID, lightID, _,_,_ = str.split(localID, "_")

        
        
        lightData = {}
        lightData["max_scale"] = numpy.array(self.light_max_scale[lightID ]) + 1e-4
        if self.read_light:
            lightData_fn = self.videoLightPath + '/{}.pkl.zst'.format(lightID)
            lightData = load_pklzst(lightData_fn)
            lightData["position"] = lightData["position"] + numpy.array(self.light_bias[lightID ]["position"]) 
            lightData["max_scale"] = numpy.array(self.light_max_scale[lightID ]) + 1e-4
            lightData["radiance"] = lightData["radiance"] / lightData["max_scale"]

        if self.read_plane and lightID in self.lightToPlaneDict.keys():
            
            planeID = self.lightToPlaneDict[lightID]
            
            file_path = Path(self.plane_path) / f"{planeID}.pkl.zst"
            if file_path.exists():
                planeData = load_pklzst(self.plane_path + str(planeID) + ".pkl.zst" )
                lightData["plane"] = planeData
                print("plane size",planeData.shape)

        #print("planeID",planeID)
        data = {}
        #data["volumeScope"] = volumeScope[np.newaxis, ...]
        #print("scope complete")
        directData_fn = self.videoRootDir + r'/direct/' + localID + ".pkl.zst"
        direct_data = load_pklzst(directData_fn)
        data.update(direct_data)
        if self.read_indirect:
            indirectData_fn = self.videoRootDir + r'/indirect/' + localID + ".pkl.zst"
            indirect_data = load_pklzst(indirectData_fn)
            data.update(indirect_data)
        denoiseData_fn = self.videoRootDir + r'/denoise/' + localID + ".pkl.zst"
        denoise_data = load_pklzst(denoiseData_fn)
        for key in denoise_data:
            data[key] = denoise_data[key]
        if self.read_voxel:
            voxelData_fn = self.videoRootDir + r'/voxelNew2/' + localID + ".pkl.zst"
            voxel_data = load_pklzst(voxelData_fn)
        
            voxel_data["volumeScope"] = numpy.array([voxel_data["volume"]["VolumeMin"],voxel_data["volume"]["VolumeMax"]])
            data.update(voxel_data)
        if self.read_indirect:
            data["light_direction"] = self.light_dir[numpy.newaxis, ...]
      
        #print("return gg")
        return data, lightData,localID


    def __getitem__(self, idx):
        if self.importance_sampling:
            idx = self.sample_indices()[0]
        else:
            idx = idx % int(len(self.videoFileList))
        start_time = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if self.isTest:
        #     sceneData, lightData, localID = self.get_video_data_test(idx)
        # else:
        sceneData, lightData, localID = self.get_video_data(idx)
        time2 = time.time()
        self.idx = idx
        data = {}
        data["global"] = lightData
        data["local"] = sceneData
        

        return (data,localID)
