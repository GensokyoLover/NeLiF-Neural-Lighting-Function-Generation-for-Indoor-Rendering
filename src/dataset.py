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
from common.shttools import *
import time
from einops import rearrange, repeat
from scipy.ndimage import binary_dilation
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

def video_process_tensor_torch(data, indirect=False):
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


def inverse_data_process_tensor(data, preds, diffuse=True, specular=False, shadow=False, indirect=False, channel_cut=True):
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
    
    # radiance 的归一化 / demodulation
    if diffuse:
        data["local"]["diffuse_direct_shading"]   = torch.expm1(data["local"]["log1p_diffuse_direct_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        preds["diffuse_direct_shading"] = torch.expm1(preds["log1p_diffuse_direct_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)

        
    if specular:
        data["local"]["specular_direct_shading"]  = torch.expm1(data["local"]["log1p_specular_direct_shading"]) * scale * data["local"]["gbuffer"]["specular"]
        preds["specular_direct_shading"] = torch.expm1(preds["log1p_specular_direct_shading"]) * scale * data["local"]["gbuffer"]["specular"]

            
    if direct:
        data["local"]["direct_shading"]   = data["local"]["diffuse_direct_shading"] + data["local"]["specular_direct_shading"]
        preds["direct_shading"] =   preds["diffuse_direct_shading"] +  preds["specular_direct_shading"]
        
    if shadow:
        data["local"]["shadow"] = data["local"]["shadow"]
        preds["shadow"] = preds["shadow"]
        if not diffuse:
            preds["direct_shadow_shading"] = preds["shadow"] * (data["local"]["diffuse_direct_shading"] * (data["local"]["gbuffer"]["albedo"] + 1e-3)+ data["local"]["specular_direct_shading"] * (data["local"]["gbuffer"]["specular"] + 1e-3))

    if direct_shadow_bool:
        preds["direct_shadow_shading"] = preds["direct_shading"] * preds["shadow"] 
        data["local"]["direct_shadow_shading"] = data["local"]["direct_shading"] * data["local"]["shadow"] 
        
    if indirect:
        data["local"]["diffuse_indirect_shading"] = torch.expm1(data["local"]["log1p_diffuse_indirect_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        data["local"]["specular_indirect_shading"] = torch.expm1(data["local"]["log1p_specular_indirect_shading"]) * scale * (data["local"]["gbuffer"]["specular"] + 1e-3)
        preds["diffuse_indirect_shading"] = torch.expm1(preds["log1p_diffuse_indirect_shading"]) * scale * (data["local"]["gbuffer"]["albedo"] + 1e-3)
        preds["specular_indirect_shading"] = torch.expm1(preds["log1p_specular_indirect_shading"]) * scale * (data["local"]["gbuffer"]["specular"] + 1e-3)
        
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


class NelifDatasets(Dataset):
    def __init__(self, configs, datasetsName, isTest = False):
        self.configs = configs
        self.isTest = isTest
        resolution = str(configs["light_angular_resolution"]) + "x" + str(configs["light_direction_resolution"])
        self.read_indirect = configs["indirect"]
        self.datasetsName = datasetsName
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
            self.videoFileList = os.listdir(f"../datasets2/{self.datasetsName}/direct/")
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
        self.plane_path = r"../datasets_plane/" + configs["plane_label"] + "/"
        self.read_plane = configs["plane_label"] != "none"
        self.dir_data = pyexr.read(r"../datasets2/TogLightAll8x128/OutDir.exr")[...,:3].reshape(8,128,8,128,3).transpose(0,2,1,3,4)


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
        
        self.probs =  self.loss_values /  self.loss_values.sum()

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
        lightData["max_scale"] = numpy.array(self.light_max_scale[lightID ]) + 1e-4
        lightData["radiance"] = lightData["radiance"] / lightData["max_scale"]
        lightData["direction"] = self.dir_data
        if self.read_plane:
            planeData = load_pklzst(self.plane_path + str(lightID) + ".pkl.zst" )
            lightData["plane"] = planeData
        
        data = {}
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
        if self.read_indirect:
            data["light_direction"] = self.light_dir[numpy.newaxis, ...]

        return data, lightData,localID


    def __getitem__(self, idx):
        if self.importance_sampling:
            idx = self.sample_indices()[0]
        else:
            idx = idx % int(len(self.videoFileList))
        start_time = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sceneData, lightData, localID = self.get_video_data(idx)
        time2 = time.time()
        self.idx = idx
        data = {}
        data["global"] = lightData
        data["local"] = sceneData
        

        return (data,localID)

