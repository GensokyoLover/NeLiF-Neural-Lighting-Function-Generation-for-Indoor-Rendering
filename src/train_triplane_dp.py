import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler,Subset # 增加 DistributedSampler
import zstandard as zstd
import pickle
import pyexr
import argparse # 增加 argparse
import deepspeed # 增加 deepspeed
from networks.pfgl import * 
from torch.optim.lr_scheduler import LambdaLR
import json
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler, DataLoader
CHANNEL_CUT = True
class InfiniteRandomSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)

    def __iter__(self):
        # 🔥 核心魔法：死循环生成乱序索引，永远不会 StopIteration
        while True:
            # 每次循环生成一个全新的乱序索引序列
            yield from torch.randperm(self.num_samples).tolist()

    def __len__(self):
        # 欺骗 DataLoader，让它以为这是一个普通的有界 Sampler
        # 这样外层的 len(dataloader) 依然能算出正确的 steps_per_epoch
        return self.num_samples
# ==================== 参数解析 ====================
# DeepSpeed 需要通过命令行参数传递 local_rank
def parse_args():
    parser = argparse.ArgumentParser(description="Triplane Training")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

args = parse_args()

# ==================== 模型定义 (保持不变) ====================
# ... (这里保留你原本的 PlaneDecoder, FeatureToImageDecoder, TriplaneImageReconstructor, Dataset 等类定义)
# 为了节省篇幅，这里假设上面的类代码已经包含在内
# 只需复制你原来的类定义即可

def load_planes_from_checkpoint(ckpt_path, target_model, param_name="planes"):
    """
    从 Checkpoint 文件中提取指定的参数（如 planes），并将其数据拷贝到新模型中。
    
    Args:
        ckpt_path (str): 原始 .pth 权重文件的路径。
        target_model (nn.Module): 新实例化的模型对象。
        param_name (str): 要提取的参数名称，默认为 "planes"。
    """
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint file not found: {ckpt_path}")
        return

    print(f"🔄 Loading '{param_name}' from {ckpt_path}...")

    try:
        # 1. 加载 Checkpoint 到 CPU
        # map_location='cpu' 是必须的，否则可能直接加载到 GPU 导致显存不足
        state_dict = torch.load(ckpt_path, map_location='cpu')

        # 2. 寻找对应的 Key
        # DeepSpeed 保存的权重通常带有 "module." 前缀
        source_tensor = None
        found_key = ""

        # 尝试几种可能的 key 命名
        candidates = [param_name, f"module.{param_name}"]
        
        for key in candidates:
            if key in state_dict:
                source_tensor = state_dict[key]
                found_key = key
                break
        
        if source_tensor is None:
            # 如果没找到，尝试模糊搜索（以防万一）
            for k, v in state_dict.items():
                if k.endswith(param_name):
                    source_tensor = v
                    found_key = k
                    break

        if source_tensor is None:
            raise KeyError(f"Key '{param_name}' not found in checkpoint dict.")

        # 3. 获取目标模型的参数对象
        if not hasattr(target_model, param_name):
             raise AttributeError(f"Target model has no attribute '{param_name}'")
        
        target_param = getattr(target_model, param_name)

        # 4. 形状检查
        if target_param.shape != source_tensor.shape:
            print(f"⚠️ Warning: Shape mismatch for '{param_name}'!")
            print(f"   Target shape: {target_param.shape}")
            print(f"   Source shape: {source_tensor.shape}")
            print("   Attempting to reshape/interpolate or fail...")
            # 如果你是要做 32 -> 64 的上采样，这里需要额外的 F.interpolate 逻辑
            # 如果只是为了加载，通常要求形状一致

        # 5. 核心：原地拷贝数据 (In-place Copy)
        # 使用 no_grad 且用 copy_，确保不破坏 target_model 的计算图结构
        with torch.no_grad():
            target_param.data.copy_(source_tensor)

        print(f"✅ Successfully loaded '{found_key}' into model.{param_name}")

    except Exception as e:
        print(f"❌ Failed to load planes: {e}")
        # 如果这个参数至关重要，建议在这里 exit()
        exit(1)

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
        self.encoder_tri = False
        self.decoder_tri = False
        self.compress_tri = True
        grid_map = generate_triplane_coords(self.plane_cnt,self.plane_cnt)
        if self.encoder_tri ==True:
            self.tri_encoder = TriPosEncoder(2,embedding_dim)
        else:
            self.tri_plane_embed = nn.Parameter(0.01 * torch.randn(3, self.plane_cnt, self.plane_cnt, self.embedding_size),
                                            requires_grad=True)
        if self.decoder_tri == True:
            self.tri_transformer_decoder = TriCrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                                           num_heads=16,shared_attn=False)
        else:
            self.transformer_decoder = CrossAttentionBlock(inner_dim=self.embedding_size, cond_dim=self.embedding_size,
                                                       num_heads=16, eps=1e-6)
        if self.compress_tri ==True:
            self.tri_output_layer = TriLinear(self.embedding_size,self.decoder_dim)
        else:
            self.output_layer = nn.Linear(self.embedding_size,self.decoder_dim)
            
    def forward_triplane(self, photon_texture):
        B,_,_,_,_,C = photon_texture.shape
        query = self.tri_plane_embed.reshape(-1, self.embedding_size).unsqueeze(0).repeat(B, 1, 1)
        photon_texture = photon_texture.reshape(B, -1, C)
        # print("photon_texture",photon_texture.shape)
        # print("query",query.shape)
        # exit()
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

class VolumetricDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.key_list = list(data.keys())
    def __len__(self):
        return len(self.key_list) * 3 * 20
    def __getitem__(self, id):
        id = id % (len(self.key_list) * 3)
        channel = id % 3
        index = id // 3
        key = self.key_list[index]
        target = self.data[key]["gt"][...,channel:channel+1]
        radiance = self.data[key]["train"]["radiance"][...,channel:channel+1]
        light_data = {}
        light_data["radiance"] = radiance
        light_data["position"] = self.data[key]["train"]["position"]
        light_data["direction"] = self.data[key]["train"]["direction"]
        return torch.tensor(id, dtype=torch.long), target,light_data

class FeatureToImageDecoder(nn.Module):
    def __init__(self, in_dim, base_channels=128):
        super().__init__()
        self.init_h = 4
        self.init_w = 4
        self.init_ch = base_channels
        self.fc_projection = nn.Sequential(
            nn.Linear(in_dim, self.init_ch * self.init_h * self.init_w),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 4, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        B = x.shape[0]
        x = self.fc_projection(x)
        x = x.view(B, self.init_ch, self.init_h, self.init_w)
        x = self.decoder_layers(x)
        img = x.permute(0, 2, 3, 1)
        return img

class ResBlock_GN(nn.Module):
    """
    带 GroupNorm 的残差块
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
class ResBlock_NOGN(nn.Module):
    """
    带 GroupNorm 的残差块
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
class AdvancedGNFeatureDecoder(nn.Module):
    def __init__(self, in_dim, base_channels=128, out_channels=3):
        super().__init__()
        self.init_h = 2
        self.init_w = 2
        self.init_ch = base_channels
        self.fc = nn.Sequential(
            nn.Linear(in_dim, self.init_ch * self.init_h * self.init_w),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            ResBlock_NOGN(base_channels // 2)
        )
        
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            ResBlock_NOGN(base_channels // 4)
        )
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 4,out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x)
        x = x.view(B, self.init_ch, self.init_h, self.init_w)
        
        x = self.stage1(x)
        x = self.stage2(x)
        #x = self.stage3(x)
        
        img = self.final_conv(x)
        img = img.permute(0, 2, 3, 1) # 调整为 (B, H, W, C)
        return img


def xyz_to_uvd_pixel_center(position):
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
    growth_rate = 1.14
    
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
import torch
import torch.nn.functional as F
def load_pklzst(filename):
    with open(filename, "rb") as f:
        compressed = f.read()
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed)
    data = pickle.loads(raw)
    return data

def read_dir(dir_path):
    dir_fn = dir_path
    with open(dir_fn, 'rb') as gf:
        dctx = zstd.ZstdDecompressor()
        dir_data = pickle.loads(dctx.decompress(gf.read()))
    gf.close()
    return dir_data

def volume_to_2d_grid(vol, z,W,H):
    slice_z = vol[:,:, z]              
    X,Y,Z,_,_,C = vol.shape[:]
    slice_z = slice_z.permute(0,2,1,3,4)
    img = slice_z.reshape(X * W, Y * H, C)
    return img

def volume_to_single_2d(vol,W,H):
    rows = []
    Z= vol.shape[0]
    for z in range(Z):
        slice_img = volume_to_2d_grid(vol, z,W,H)
        rows.append(slice_img)
    return torch.cat(rows, dim=0)

def downSample(data):
    for file in data:
        t = torch.Tensor(data[file]["gt"])
        X, Y, Z, W, H, C = t.shape
        # scale_factor = 8/W
        # t = t.permute(0, 1, 2, 5, 3, 4)
        # t = t.reshape(-1, C, W, H)
        # t = F.interpolate(t, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        # _, _, W2, H2 = t.shape
        # t = t.reshape(X, Y, Z, C, W2, H2)
        # t = t.permute(0, 1, 2, 4, 5, 3)
        data[file]["gt"] = t
    return data

def generate_voxel_mask_6d(bbox, voxel_data):
    """
    生成指定形状的 Mask
    Args:
        bbox: 包含 position 和 box 的字典
        voxel_data: shape (W, H, D, 8, 8, 3)
    Returns:
        mask: shape (W, H, D, 8, 8, 1), 类型为 float32 或 bool
    """
    # 1. 计算世界空间边界
    center = np.array(bbox['position'])
    world_min = np.array(bbox['box'][0])
    world_max = np.array(bbox['box'][1])
    
    # 2. 维度判定
    # 利用 NumPy 广播机制处理 (W, H, D, 8, 8) 形状的坐标
    mask_x = (voxel_data[..., 0] >= world_min[0]) & (voxel_data[..., 0] <= world_max[0])
    mask_y = (voxel_data[..., 1] >= world_min[1]) & (voxel_data[..., 1] <= world_max[1])
    mask_z = (voxel_data[..., 2] >= world_min[2]) & (voxel_data[..., 2] <= world_max[2])
    
    # 3. 合并并增加最后一个维度
    # 使用 np.newaxis 将 (W, H, D, 8, 8) 变为 (W, H, D, 8, 8, 1)
    final_mask = (mask_x & mask_y & mask_z)[..., np.newaxis]
    
    # 如果是为了给神经网络（如 Transformer）使用，建议转换成 float32
    return final_mask

def fill_masked_voxels_batch(voxel_data, mask, kernel_size=3):
    """
    针对 Batch 数据填充体素 Mask 区域
    
    Args:
        voxel_data: Tensor, 形状 (B, W, H, D, C)
        mask: Tensor, 形状 (B, W, H, D, 1) 或 (B, W, H, D), 1代表mask, 0代表有效
        kernel_size: 邻域大小 (通常为 3)
    Returns:
        filled: Tensor, 形状 (B, W, H, D, C)
    """
    B, W, H, D, C = voxel_data.shape
    device = voxel_data.device
    
    # 1. 维度调整: (B, W, H, D, C) -> (B, C, W, H, D)
    # PyTorch 3D卷积要求通道维在第二位
    x = voxel_data.permute(0, 4, 1, 2, 3).contiguous().float()
    
    # 确保 mask 形状为 (B, 1, W, H, D)
    if mask.ndim == 4:
        m = mask.unsqueeze(1).float()
    else:
        m = mask.permute(0, 4, 1, 2, 3).contiguous().float()
    
    # 2. 初始置零：将 mask 区域的数据清空，确保它们不贡献权重
    valid_indicator = 1.0 - m  # 1 代表有效点，0 代表 mask 点
    
    x_valid_only = x * valid_indicator
    
    # 3. 准备卷积核
    # padding 保证输出尺寸一致
    padding = kernel_size // 2
    
    # 用于计算有效邻居数量的核 (1, 1, K, K, K)
    count_kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
    
    # 用于计算通道数值总和的核 (C, 1, K, K, K) -> 使用 groups=C 实现深度卷积
    sum_kernel = torch.ones((C, 1, kernel_size, kernel_size, kernel_size), device=device)
    
    with torch.no_grad():
        # 4. 计算邻域有效点计数 (B, 1, W, H, D)
        neighbor_count = F.conv3d(valid_indicator, count_kernel, padding=padding)
        
        # 5. 计算邻域有效点数值总和 (B, C, W, H, D)
        neighbor_sum = F.conv3d(x_valid_only, sum_kernel, padding=padding, groups=C)
        
        # 6. 计算均值 (避免除以 0)
        # 如果四周全是 mask (count=0)，则结果为 0
        fill_values = torch.where(neighbor_count > 0, neighbor_sum / neighbor_count, torch.zeros_like(neighbor_sum))
        
        # 7. 融合结果：只在原 mask 处应用填充值
        # m.bool() 会根据 batch 和空间位置自动广播到所有通道
        res = torch.where(m.bool(), fill_values, x)
    
    # 8. 恢复原始维度: (B, C, W, H, D) -> (B, W, H, D, C)
    return res.permute(0, 2, 3, 4, 1).contiguous()

def get_data(LABEL,angular_resolution,space_resolution):
    # 你的数据加载代码
    data = load_pklzst(r"../datasets2/{}/final_data_{}_{}.pkl.zst".format(LABEL,angular_resolution,space_resolution))
    pos = load_pklzst(r"../datasets2/{}/final_pos_{}_{}.pkl.zst".format(LABEL,angular_resolution,space_resolution))
 
    #print(data.keys())
    #exit()
    # 为了演示，只取一部分，正式训练请根据需要调整
    data = dict(list(data.items())[:]) 
    print(data.keys())
    print(len(data.keys()))
    # exit()
    # exit()
    new_data = {}
    with open(r"../datasets2/{}/bias_info.json".format(LABEL), 'r') as file:
        bias_info = json.load(file)
    for key in list(bias_info.keys()):
        bias_info[key.split("_")[0]] = bias_info[key]
    for key in data.keys():
        box = bias_info[key.split("_")[0]]
        mask = generate_voxel_mask_6d(box,pos)
        new_data[key.split("_")[0]] ={}
        new_data[key.split("_")[0]]["gt"] =data[key].transpose(1,2,0,3,4,5)
        new_data[key.split("_")[0]]["mask"] =mask.transpose(1,2,0,3,4,5)


import torch.distributed as dist

def reduce_mean(value, device):
    """
    将所有 GPU 上的 value 进行求和并求平均。
    """
    # 1. 转为 Tensor
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    
    # 2. 全局求和 (All-Reduce SUM)
    # 这步操作会阻塞，直到所有 GPU 都把数据传过来相加
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 3. 除以 GPU 数量得到平均值
    avg_value = tensor.item() / dist.get_world_size()
    
    return avg_value

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
def sample_triplane(planes, grid_uvd, mode='nearest', padding_mode='border', align_corners=False):
    """
    使用三平面 (Tri-Plane) 替代 Voxel 进行采样
    
    Args:
        planes: list of 3 tensors, 每个形状为 (B, C, H_plane, W_plane)
                假设顺序为: [XY_Plane, XZ_Plane, YZ_Plane] 
                即: [UV_Plane, UD_Plane, VD_Plane]
        grid_uvd: (B, D, H, W, 3) 
                这是你原本传给 voxel grid_sample 的 uvw 坐标
                最后一维是 (u, v, d)
                
    Returns:
        output: (B, D, H, W, C) -> 保持和你原始代码输出形状一致
    """
    # 1. 准备 Grid
    # grid_uvd 形状是 (B, D, H, W, 3)。
    # 2D grid_sample 需要 grid 形状为 (B, H_out, W_out, 2)。
    # 为了高效，我们将 D, H, W 展平为一个维度，看作是一张 (1, N) 像素的大图。
    
    B, D, H, W, _ = grid_uvd.shape
    print("uvd shape",grid_uvd.shape)
    print("planes shape",planes[0].shape)
    # 变成 (B, D*H*W, 1, 3) 以适配 grid_sample 的输入要求
    # 这里设 W_out=1, H_out=Total_Pixels
    flat_grid = grid_uvd.view(B, D * H * W, 1, 3)
    
    u = flat_grid[..., 0]
    v = flat_grid[..., 1]
    d = flat_grid[..., 2] # 你的深度坐标
    
    # 2. 构建三个平面的投影坐标
    # 注意：这里的组合取决于你训练/生成三平面时的轴定义。
    # 假设标准正交投影：
    # Plane 0 (UV/XY): 使用 (u, v)
    coord_uv = torch.stack([u, v], dim=-1) # (B, N, 1, 2)
    
    # Plane 1 (UD/XZ): 使用 (u, d) -> 对应 x, z
    coord_ud = torch.stack([u, d], dim=-1)
    
    # Plane 2 (VD/YZ): 使用 (v, d) -> 对应 y, z
    coord_vd = torch.stack([v, d], dim=-1)
#     coord_uv = torch.stack([v, u], dim=-1)  # ⭐ 改这里
# coord_ud = torch.stack([d, u], dim=-1)
# coord_vd = torch.stack([d, v], dim=-1)
    # 3. 分别采样 (Sampling)
    # 你的原始代码用了 mode='nearest', padding_mode='border', align_corners=False
    # 这里必须保持一致
    
    sample_kwargs = {
        'mode': mode,
        'padding_mode': padding_mode,
        'align_corners': align_corners
    }
    
    # planes[0] 是 (B, C, H_p, W_p)
    feat_uv = F.grid_sample(planes[0], coord_uv, **sample_kwargs)
    feat_ud = F.grid_sample(planes[1], coord_ud, **sample_kwargs)
    feat_vd = F.grid_sample(planes[2], coord_vd, **sample_kwargs)
    
    # 4. 融合 (Fusion)
    # 通常 Tri-plane 采用简单的相加 (Summation)
    # 现在的形状是 (B, C, N, 1)
    fused_features = feat_uv + feat_ud + feat_vd
    print("fused",fused_features.shape)

    # 5. 恢复形状 (Reshape)
    # (B, C, N, 1) -> (B, C, D, H, W)
    fused_features = fused_features.view(B, -1, D, H, W)
    
    # 你的原始代码最后做了 permute(0,2,3,4,1) -> (B, D, H, W, C)
    return fused_features,feat_uv.reshape(B,-1,D,H,W),feat_ud.reshape(B,-1,D,H,W),feat_vd.reshape(B,-1,D,H,W)

import torch
import torch.nn as nn
def count_graph_nodes(tensor):
    visited = set()
    count = [0]
    def dfs(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        count[0] += 1
        for child, _ in node.next_functions:
            dfs(child)
    dfs(tensor.grad_fn)
    return count[0]


class TriplaneImageReconstructor(nn.Module):
    def __init__(self,stage, plane_res=128, plane_dim=64):
        super().__init__()
        self.plane_res = plane_res
        self.plane_dim = plane_dim
        if stage == 1:
            self.image_encoder = FourDTransformer(angular_size=8 , space_size=128 , angular_block_size=1, space_block_size=16,
                                                num_layers=3, num_heads=16, input_dim=7, hidden_dim=256, mlp_dim=1024,
                                                output_dim=1024, linear_output=True)
            self.image_to_plane = PlaneDecoder(1024,self.plane_dim,plane_res)
        #self.decoder =FeatureToImageDecoder(in_dim=plane_dim, base_channels=plane_dim)\

        self.decoder =AdvancedGNFeatureDecoder(in_dim=plane_dim, base_channels=plane_dim,out_channels=1)

    def forward_light(self,light_data):
        global_light_feature = self.image_encoder(light_data)
        if torch.any(global_light_feature.isnan()):
            print("global_light_feature nan")
            exit()
        return self.image_to_plane.forward_triplane(global_light_feature)
    
    
    def triplane_loss(self,plane_feature):
        print(plane_feature.shape)
        xy_plane = plane_feature[:,0,...]
        other_planes = plane_feature[:,1:,...]
        xy_plane_padding = pad_equal_area_plane_correct(xy_plane)
        print(xy_plane_padding.shape)
        h_diff = other_planes[..., 1:, :] - other_planes[..., :-1, :]
        w_diff = other_planes[..., :, 1:] - other_planes[..., :, :-1]
        h_diff_padding = xy_plane_padding[..., 1:, :] - xy_plane_padding[..., :-1, :]
        w_diff_padding = xy_plane_padding[..., :, 1:] - xy_plane_padding[..., :, :-1]
        loss_tv = torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff)) + torch.mean(torch.abs(h_diff_padding)) + torch.mean(torch.abs(w_diff_padding))
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
    def compute_radius_weights(self,coords,device=None):
        """
        coords: [6] -> sx, ex, sy, ey, sz, ez
        Returns: [Z, X, Y] loss权重张量
        """
        if self.plane_res ==32:
            radius = 1.14
        elif self.plane_res == 64:
            radius = 1.07
        elif self.plane_res == 128:
            radius = 1.033
        sx, ex, sy, ey, sz, ez = coords
        z_dim = ez - sz
        x_dim = ex - sx
        y_dim = ey - sy
    
        # 生成Z轴的index
        z_indices = torch.arange(z_dim, device=device)  # Shape: [Z]
        # r = 0.1 * 1.07 ** sz * 1.07 ** z_idx
        base_r = 0.1 * (radius ** sx)
        r = base_r * (radius ** z_indices)  # Shape: [Z]
        r = torch.where(r<1,1,r)
        # 为每个X,Y位置广播r
        weights = r[:, None, None].expand(z_dim, x_dim, y_dim)
        return weights
    def fetch_local_features(self, plane_feature, coords):
        batch_size = plane_feature.shape[0]
        local_features = []
        local_weights = []
        # print(plane_feature.shape)
        for i in range(batch_size):
            sx, ex, sy, ey, sz, ez = coords[i]
            weights = self.compute_radius_weights(coords[i],coords[i].device)
            # 分离三个平面
            xy_plane = plane_feature[i, 0] # [C, X, Y]
            xz_plane = plane_feature[i, 1] # [C, X, Z]
            yz_plane = plane_feature[i, 2] # [C, Y, Z]

            xy_crop = xy_plane[:, sx:ex, sy:ey] 
            
            # XZ: [sx:ex, sz:ez]
            xz_crop = xz_plane[:, sx:ex, sz:ez]
            
            # YZ: [sy:ey, sz:ez]
            yz_crop = yz_plane[:, sy:ey, sz:ez]
            
            feat = (xy_crop.unsqueeze(-1) + xz_crop.unsqueeze(-2) + yz_crop.unsqueeze(-3))
            local_features.append(feat)
            local_weights.append(weights)
            
        return torch.stack(local_features),torch.stack(local_weights)

    def fetch_local_features_strided(self,vol_feature, coords, crop_size):
        """
        vol_feature: [X, Y, Z, C] (必须连续)
        coords: [B, 6] -> sx, ex, sy, ey, sz, ez
        """
        # 获取原始步长
        st_x, st_y, st_z, st_c = vol_feature.stride()
        
        # 假设所有 batch 的维度一致
        K = crop_size
        C = vol_feature.shape[-1]
        B = coords.shape[0]

        # 计算每个 batch 在底层存储中的起始偏移 (以元素为单位)
        # 注意：as_strided 本身不支持 batch 内每个元素有不同的 Offset 
        # 所以如果 coords 是随机的，这里通常需要配合 storage().view(...)
        
        # 💡 骨灰级替代方案：如果坐标是随机的，使用高级索引 (Advanced Indexing)
        # 相比 for 循环，它在底层会调用单次高效的聚类 Kernel
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.arange(K, device=device),
            torch.arange(K, device=device),
            torch.arange(K, device=device),
            indexing='ij'
        )
        
        # 构造批量索引
        batch_sx = coords[:, 0:1, None, None] # [B, 1, 1, 1]
        batch_sy = coords[:, 2:3, None, None]
        batch_sz = torch.tensor([sz_start], ... ) # 依此类推
        
        # 一次性提取 [B, K, K, K, C]
        # 这种方式在 Backward 时会被优化为一次巨大的原子加法运算（Atomic Add）
        return vol_feature[batch_sx + grid_x, batch_sy + grid_y, batch_sz + grid_z]

    def forward_decoder(self, plane_feature, targets, coords): # 增加 coords
        # print(idx_list)
        # print(coords)
        # print("targets: ",targets.shape)
        # print("plane_feature: ",plane_feature.shape)
        # print("coords: ",coords.shape)
        B = targets.shape[0]
        
        if CHANNEL_CUT:
            targets = targets.permute(0,6,1,2,3,4,5).contiguous().reshape(-1, *targets.shape[1:-1],1)
            coords = coords.reshape(B,1,6).repeat(1,3,1).reshape(B*3,6) 
            plane_feature = plane_feature.reshape(B*3,*plane_feature.shape[2:])
        # print("----------")
        # print("targets: ",targets.shape)
        # print("plane_feature: ",plane_feature.shape)
        # print("coords: ",coords.shape)
   
        plane_loss = self.triplane_loss(plane_feature)
        
        # === 修改核心逻辑 ===
        # 不再生成全量 feature，而是调用切片函数
        feature,weights = self.fetch_local_features(plane_feature, coords)
        # feature shape: [B, C, 32, 32, 32]
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        B, C, X, Y, Z = feature.shape
        _, _, _, _, W, H, _ = targets.shape
        
        # 调整 shape 给 Decoder
        # [B, C, X, Y, Z] -> [B, X, Y, Z, C] -> reshape -> [B*X*Y*Z, C]
        feature = feature.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, self.plane_dim)
        # print("feature ",feature.shape)
        # print("targets ",targets.shape)
    
        img = self.decoder(feature)
        #print("img ",img.shape)
        # img = checkpoint(
        #     self.decoder,
        #     feature,
        #     use_reentrant=False
        # )
        img = img.reshape(B, X, Y, Z, W, H, targets.shape[-1])
        
        visual_loss = F.l1_loss(img, targets, reduction='none')
        # print("weights",weights.shape)
        # print("visual_loss",visual_loss.shape)
        # exit()
        visual_loss = (visual_loss * weights).mean()
        loss = visual_loss + plane_loss * 0.1
        return img, loss, visual_loss, plane_loss,weights
    

    def freeze_decoder_for_stage1(self):
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.image_to_plane.parameters():
            param.requires_grad = True
        print("Model status: Decoder FROZEN, Encoder TRAINABLE.")

    def forward_encoder(self, light_data, targets, coords, return_planes=True, target_planes_gt=None): # 增加 coords
        global_light_feature = self.image_encoder(light_data)
        _, raw_planes = self.image_to_plane.forward_triplane(global_light_feature)
        
        generated_planes = raw_planes.permute(0, 1, 4, 2, 3).contiguous() 
        # generated_planes: [B, 3, C, 64, 64]
        B = targets.shape[0]
        targets = targets.permute(0,6,1,2,3,4,5).contiguous().reshape(-1, *targets.shape[1:-1],1)
        coords = coords.reshape(B,1,6).repeat(1,3,1).reshape(B*3,6) 
        # print("generated_planes",generated_planes.shape)
        # print("targets",targets.shape)
        
        plane_reg_loss = self.triplane_loss(generated_planes)
        
        # === 修改核心逻辑 === 
        # 使用 coords 切割生成的 planes
        # gt_feature, local_weights = self.fetch_local_features(target_planes_gt, coords)
        # feature, local_weights = self.fetch_local_features(generated_planes, coords)
        # feature shape: [B, C, 32, 32, 32]
        
        # B, C, X, Y, Z = feature.shape
        # feature = feature.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, self.plane_dim)
        # gt_feature = gt_feature.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, self.plane_dim)
        
        # img = self.decoder(feature) 
        # gt_img = self.decoder(gt_feature) 
        # _,_,_,_,W,H,_ = targets.shape
        # img = img.reshape(B, X, Y, Z, W, H, targets.shape[-1])
        # gt_img = gt_img.reshape(B, X, Y, Z, W, H, targets.shape[-1])

        # print("img ",img.shape)
        # print("generated_planes ",generated_planes.shape)
        # print("target_planes_gt ",target_planes_gt.shape)
        # print("targets ",targets.shape)
        # exit()
        # print(generated_planes.shape)
        # print(target_planes_gt.shape)
        img = targets
        gt_img = targets
        loss_distill = nn.L1Loss()(generated_planes, target_planes_gt)
        #visual_loss = nn.L1Loss()(img, targets)
        visual_loss = 0

        loss =   1.0 * loss_distill + 0.001 * plane_reg_loss
        #exit()
        if return_planes:
            return img, gt_img,loss, visual_loss, loss_distill, plane_reg_loss, generated_planes
        else:
            return img, gt_img,loss, visual_loss, loss_distill, plane_reg_loss

import time
import torch
import pyexr
def check_nan(tensor, name, rank):
    if torch.any(torch.isnan(tensor)):
        print(f"❌ [Rank {rank}] {name} contains NaN!")
        return True
    if torch.any(torch.isinf(tensor)):
        print(f"❌ [Rank {rank}] {name} contains Inf!")
        return True
    return False

class CutDataset(Dataset):
    def __init__(self, path, light_path,stage,plane_label,volume_size=64, crop_size=32,channel_cut = False):
        
        #self.file_list = ["L3D445S265B19ENDP2Q2MVYUWIDXELUF3P3WM888.pkl.zst","L3D445S265B19ENDP2Q455YUWJRWCLUF3P3WO888.pkl.zst","L3D445S265B19ENDP2QZYTQUWJSTGLUF3P3UK888.pkl.zst","L3D445S265B19ENDPCOFGIYUWLY24LUF3P3XC888.pkl.zst"]
        self.light_path = light_path
        self.target_path = path
        self.stage = stage
        if self.stage==1:
            self.plane_path = r"../datasets_plane/" + plane_label + "/"
        self.crop_size = crop_size
        self.volume_size = volume_size
        self.grid_dim = volume_size // crop_size
        self.blocks_per_volume = self.grid_dim ** 3 
        self.scale = 1
        
        self.dir = pyexr.read(r"../datasets2/TogLightAll8x128/OutDir.exr")[...,:3].reshape(8,128,8,128,3).transpose(0,2,1,3,4)
        self.file_list = []
        file_list = os.listdir(path)
        for file in file_list:
            if file.split(".")[-1]!="zst" or "_" in file : 
                continue
            self.file_list.append(file.split(".")[0])
        mast_file_list = {"L3D445S265B19ENDP2QZGVYUWIVR6LUF3P3WC888": 0, "L3D445S265B19ENDP2QZGBYUWJRWCLUF3P3XI888": 1, "L3D445S265B19ENDP2QZK6YUWJIAULUF3P3UI888": 2, "L3D445S265B19ENDP2RRVMAUWIVR6LUF3P3WK888": 3, "L3D445S265B19ENDP2U6KJQUWJRWCLUF3P3WA888": 4, "L3D445S265B19ENDP2UWAAIUWIDXELUF3P3W6888": 5, "L3D445S265B19ENDP2UWNDAUWIDXELUF3P3XW888": 6, "L3D445S265B19ENDP2UY5EQUWJIAULUF3P3XM888": 7, "L3D445S265B19ENDP2VMDKQUWIVR6LUF3P3X4888": 8, "L3D445S265B19ENDP3WJIPIUWIX3ALUF3P3WI888": 9, "L3D445S265B19ENDP76KVYYUWFCP4LUF3P3WC888": 10, "L3D445S265B19ENDP76KW6QUWFCI4LUF3P3W2888": 11, "L3D445S265B19ENDP76KW4IUWFCIYLUF3P3WW888": 12, "L3D445S265B19ENDP76KW6QUWFCI4LUF3P3WO888": 13, "L3D445S265B19ENDP76KW6QUWFWFMLUF3P3XM888": 14, "L3D445S265B19ENDP76KW6QUWFWFOLUF3P3XM888": 15, "L3D445S265B19ENDP76KWEYUWFWFOLUF3P3WE888": 16, "L3D445S265B19ENDP76KWFYUWFWFMLUF3P3UK888": 17, "L3D445S265B19ENDP76KWLYUWFCIYLUF3P3UI888": 18, "L3D445S265B19ENDP76KWYYUWFWFMLUF3P3W6888": 19, "L3D445S265B19ENDP76LZHQUWFWFOLUF3P3X4888": 20, "L3D445S265B19ENDP76LZGAUWFWFOLUF3P3WY888": 21, "L3D445S265B19ENDP7SLYJQUWF236LUF3P3XE888": 22, "L3D445S265B19ENDP7S4TFIUWFPGULUF3P3XG888": 23, "L3D445S265B19ENDPA27Q2QUWJ6BGLUF3P3UI888": 24, "L3D445S265B19ENDPAACWKAUWICZ6LUF3P3WC888": 25, "L3D445S265B19ENDPAOEOUYUWIICKLUF3P3XK888": 26, "L3D445S265B19ENDPAOEPUIUWIYWCLUF3P3X6888": 27, "L3D445S265B19ENDPAQMHRIUWJ3DYLUF3P3WC888": 28, "L3D445S265B19ENDPBFDY6AUWJWFKLUF3P3WY888": 29, "L3D445S265B19ENDPBGL52AUWLLDOLUF3P3WG888": 30, "L3D445S265B19ENDPCNPSAIUWL3QYLUF3P3WM888": 31, "L3D445S265B19ENDPCNQPEIUWJU2OLUF3P3X4888": 32, "L3D445S265B19ENDPCOFFQQUWJN2YLUF3P3W2888": 33, "L3D445S265B19ENDPCOYQ4QUWIMOGLUF3P3XE888": 34, "L3D445S265B19ENDPCUQ6SYUWJ3MILUFX7TDHZY8": 35, "L3D445S265B19ENDPGFOWSYUWLYUILUF3P3WK888": 36, "L3D445S265B19ENDPGZRN6IUWIIJILUF3P3X6888": 37, "L3D445S265B19ENDPGZUKHAUWISQCLUF3P3XE888": 38, "L3D445S265B19ENDPHPBZAAUWIIDCLUF3P3WM888": 39, "L3D445S265B19ENDPHYLH3YUWIYWCLUF3P3WA888": 40, "L3D445S265B19ENDPHYLHCQUWIYWCLUF3P3WO888": 41, "L3D445S265B19ENDPHYLHZYUWJ7MYLUF3P3X6888": 42, "L3D445S265B19ENDPHYLHFIUWJ7MYLUF3P3W6888": 43, "L3D445S265B19ENDPY4A25QUWICBYLUF3P3WS888": 44, "L3D445S265B19ENDPY4AW5YUWICJMLUF3P3XK888": 45, "L3D445S265B19ENDPY4L3CAUWJWBWLUF3P3XU888": 46, "L3D445S265B19ENDPY4L3EAUWIWP2LUF3P3WC888": 47, "L3D445S265B19ENDPY4L5SYUWJWBWLUF3P3X2888": 48, "L3D445S265B19ENDPY4L5VIUWIWP2LUF3P3W6888": 49, "L3D445S265B19ENDPY4L5TAUWICBYLUF3P3XY888": 50, "L3D445S265B19ENDPY4M36YUWJWBWLUF3P3XW888": 51, "L3D445S265B19ENDPYDZZVQUWIHOILUF3P3W6888": 52, "L3D445S265B19ENDPYZIGGQUWIHJCLUF3P3XY888": 53, "L3D446S265B19ENDPFAE3CQUWLLCKLUF3P3WU888": 54, "L3D446S265B19ENDPFAE3BYUWJDSOLUF3P3WM888": 55, "L3D446S265B19ENDPHFE42YUWLGWSLUF3P3XG888": 56, "L3D446S265B19ENDPHFE4AYUWIDJILUF3P3WY888": 57, "L3D446S265B19ENDPHFE4AYUWIDJILUF3P3XY888": 58, "L3D446S265B19ENDPHFE4GYUWLWVYLUF3P3WG888": 59, "L3D446S265B19ENDPHFE4OAUWJVAELUF3P3X4888": 60, "L3D446S265B19ENDPHFE4OAUWLGWSLUF3P3XK888": 61, "L3D446S265B19ENDPHFILAQUWJVAELUF3P3UI888": 62, "L3D446S265B19ENDPHJTCQYUWLWVYLUF3P3XO888": 63, "L3D446S265B19ENDPM2HCMAUWJIDMLUFX6XWBYQ8": 64, "L3D446S265B19ENDPM42YDQUWJY42LUFX7OWBJY8": 65, "L3D445S265B19ENDP2QY77YUWIDXELUF3P3WG888": 66, "L3D445S265B19ENDP2VPTRQUWJRWCLUF3P3WO888": 67}
        mast_file_list = list(mast_file_list.keys()) + ['L3D445S265B19ENDP2SB2TAUWJIAULUF3P3XK888', 'L3D445S265B19ENDPCOBB3IUWIMOGLUF3P3XY888', 'L3D445S265B19ENDP2SB2ZYUWJRWCLUF3P3WI888', 'L3D445S265B19ENDP76KWFYUWFWFMLUF3P3WS888', 'L3D446S265B19ENDPFTXUUAUWISRYLUF3P3X6888', 'L3D446S265B19ENDPAW53AAUWJNYILUF3P3XM888', 'L3D445S265B19ENDP2UW4TQUWIVR6LUF3P3W2888', 'L3D445S265B19ENDP2VMD4AUWJSTGLUF3P3WU888', 'L3D445S265B19ENDP3WJBSAUWIHHKLUF3P3XY888', 'L3D445S265B19ENDP2UWBHAUWIVR6LUF3P3WO888', 'L3D445S265B19ENDP76KWFYUWFCIYLUF3P3WA888', 'L3D446S265B19ENDPHFILKQUWLWVYLUF3P3UK888', 'L3D445S265B19ENDPHEO2KAUWI3AYLUF3P3WK888', 'L3D445S265B19ENDPCOATSYUWJN2YLUF3P3XM888', 'L3D445S265B19ENDPBAMT5YUWLLDOLUF3P3WS888', 'L3D446S265B19ENDPFTXU7IUWIFXSLUF3P3W4888', 'L3D445S265B19ENDPAZVDZAUWJIZELUF3P3WS888', 'L3D445S265B19ENDP2RSN3IUWJIAULUF3P3WA888', 'L3D445S265B19ENDPY4L2QIUWJULALUF3P3XK888', 'L3D445S265B19ENDPHELXQIUWIDJILUF3P3WI888', 'L3D445S265B19ENDPZ2IVXAUWJTHWLUF3P3W2888', 'L3D445S265B19ENDP2VMNLQUWJSTGLUF3P3W6888', 'L3D445S265B19ENDP3WJDGAUWIX3ALUF3P3XM888', 'L3D446S265B19ENDPFVINGQUWISRYLUF3P3X2888', 'L3D445S265B19ENDPHHUGYQUWIDJILUF3P3XQ888', 'L3D446S265B19ENDPAW55RAUWI23ELUF3P3WC888', 'L3D445S265B19ENDPYZILNIUWIHJCLUF3P3XI888', 'L3D445S265B19ENDP76L2DIUWFCP4LUF3P3UI888', 'L3D445S265B19ENDP76KWFYUWFCP4LUF3P3WG888', 'L3D445S265B19ENDPHHPUPYUWIDJILUF3P3WK888', 'L3D445S265B19ENDPE3QOHAUWLYOILUF3P3W2888', 'L3D445S265B19ENDPE3QOUIUWIXJMLUF3P3WQ888', 'L3D445S265B19ENDP2UWCMIUWJIAULUF3P3XK888', 'L3D445S265B19ENDP76L2CAUWFCI4LUF3P3WK888', 'L3D446S265B19ENDPFTDYQAUWIHLYLUF3P3XI888', 'L3D445S265B19ENDP3WJDHIUWJT2ILUF3P3W4888', 'L3D445S265B19ENDP76L2CIUWFWFMLUF3P3WK888', 'L3D445S265B19ENDP76L2CQUWFCIYLUF3P3XW888', 'L3D445S265B19ENDPY4L2NQUWICJMLUF3P3XQ888', 'L3D445S265B19ENDP76L2CQUWFWFOLUF3P3XC888', 'L3D445S265B19ENDPAZS4CIUWJ6BGLUF3P3XQ888', 'L3D445S265B19ENDP76L2DQUWFWFMLUF3P3WY888', 'L3D445S265B19ENDPE3QOCIUWLYOILUF3P3XW888', 'L3D445S265B19ENDP76KWFIUWFWFOLUF3P3WY888', 'L3D445S265B19ENDP76L2CQUWFCI4LUF3P3XG888', 'L3D446S265B19ENDPHFILKIUWI3AYLUF3P3XW888', 'L3D445S265B19ENDP2UWDZYUWIVR6LUF3P3X6888', 'L3D445S265B19ENDPZ2IW6IUWIVZGLUF3P3X4888', 'L3D445S265B19ENDP2SB32QUWJIAULUF3P3XU888', 'L3D445S265B19ENDP3WJDGAUWIX3ALUF3P3WQ888', 'L3D445S265B19ENDPH53HOYUWJVAELUF3P3WO888', 'L3D445S265B19ENDPY4L2NYUWICBYLUF3P3WU888', 'L3D445S265B19ENDP76KWFYUWFWFOLUF3P3XA888', 'L3D445S265B19ENDPYZILQAUWJSXQLUF3P3WM888', 'L3D445S265B19ENDP3WJD2IUWIDRCLUF3P3WU888', 'L3D445S265B19ENDP2VMETAUWIVR6LUF3P3W2888', 'L3D445S265B19ENDP2UW6VYUWIDXELUF3P3XU888', 'L3D445S265B19ENDP2UW5GQUWJIAULUF3P3XW888', 'L3D445S265B19ENDPAZS7MIUWJ6BGLUF3P3WK888', 'L3D445S265B19ENDPY4L3PIUWIWP2LUF3P3XY888', 'L3D445S265B19ENDP3WJDGYUWIDRCLUF3P3W6888', 'L3D446S265B19ENDPHFILKYUWLWVYLUF3P3WA888', 'L3D445S265B19ENDPE3QO4YUWJWCSLUF3P3WU888', 'L3D445S265B19ENDPGYYWGAUWIIJILUF3P3XU888', 'L3D446S265B19ENDPAW54RAUWI23ELUF3P3WI888', 'L3D445S265B19ENDPGYYY2IUWISQCLUF3P3WS888', 'L3D445S265B19ENDPH4PEFYUWI3AYLUF3P3WY888', 'L3D445S265B19ENDPCOAVUIUWJN2YLUF3P3WW888', 'L3D446S265B19ENDPAW4UDQUWIEAELUF3P3X2888', 'L3D445S265B19ENDPAZVZOQUWJ6BGLUF3P3XY888', 'L3D446S265B19ENDPAW55XAUWI23ELUF3P3XO888', 'L3D445S265B19ENDP3WJDEYUWIHHKLUF3P3WE888', 'L3D445S265B19ENDPAXLP4YUWJ3DYLUF3P3WQ888', 'L3D445S265B19ENDPYZIMKYUWIWWQLUF3P3WY888', 'L3D445S265B19ENDP2VMIRYUWJIAULUF3P3WQ888', 'L3D445S265B19ENDPCOAGWIUWIMOGLUF3P3W6888', 'L3D445S265B19ENDPY4L2PIUWJWBWLUF3P3WY888', 'L3D445S265B19ENDP2VMNRIUWJSTGLUF3P3WU888', 'L3D445S265B19ENDP3WJBTYUWIDRCLUF3P3WK888', 'L3D445S265B19ENDPGZTAMQUWJIR4LUF3P3WI888', 'L3D446S265B19ENDPAW4TTIUWIEAELUF3P3WU888', 'L3D445S265B19ENDPZ2IWOYUWICDWLUF3P3XE888', 'L3D445S265B19ENDP2VMK3YUWIVR6LUF3P3X2888', 'L3D445S265B19ENDP2VM7UAUWJRWCLUF3P3WO888', 'L3D445S265B19ENDP2UWAJAUWJRWCLUF3P3XC888', 'L3D445S265B19ENDPCOAHMIUWIMOGLUF3P3XM888', 'L3D445S265B19ENDPCOATDAUWIMOGLUF3P3XA888', 'L3D446S265B19ENDPAW4VFIUWIEAELUF3P3X4888', 'L3D445S265B19ENDPCOBBEIUWIMOGLUF3P3X6888', 'L3D446S265B19ENDPHFILKQUWLWVYLUF3P3UI888', 'L3D445S265B19ENDPY4L2PIUWJWBWLUF3P3XG888', 'L3D445S265B19ENDPGYYYYAUWJIR4LUF3P3XY888']
        # target_num = 1000

        # # self.file_list 兼容 dict / list
        # if isinstance(self.file_list, dict):
        #     all_files = list(self.file_list.keys())
        # else:
        #     all_files = list(self.file_list)

        # #已有 mast 文件集合，加速判断
        # mast_set = set(mast_file_list)

        # # 还需要补多少个
        # need_num = target_num - len(mast_file_list)

        # if need_num > 0:
        #     # 候选：self.file_list 中不存在于 mast_file_list 的文件
        #     candidate_files = [
        #         f for f in all_files
        #         if f not in mast_set
        #     ]

        #     if len(candidate_files) < need_num:
        #         raise ValueError(
        #             f"候选文件不足：还需要 {need_num} 个，但只有 {len(candidate_files)} 个可选"
        #         )

        #     # 等间隔选取 need_num 个
        #     step = len(candidate_files) / need_num
        #     selected_files = [
        #         candidate_files[int(i * step)]
        #         for i in range(need_num)
        #     ]

        #     # 加入 mast_file_list
        #     mast_file_list.extend(selected_files)
        self.file_list =mast_file_list
        print("最终 mast_file_list 数量:", len(mast_file_list))
        #exit()
                # print(self.file_list)
        print("-------------------------")
        new_file_list = []
        for file in self.file_list:
            new_file_list.append(file + ".pkl.zst")
        self.file_list = new_file_list

        # print(self.file_list)
        # print(len(self.file_list))
        # exit()
        with open(light_path + "/bias_info.json", 'r') as file:
            self.bias_info = json.load(file)
        self.pos = load_pklzst(path + r"/final_pos_64_8.pkl.zst")
        
    def __len__(self):
        # 总长度 = 场景数 * 通道数(3) * 块数(8) * 20 (这个20是你原代码里的倍数)
        return len(self.file_list) * self.blocks_per_volume * self.scale * 1

    def __getitem__(self, id):
        # 1. 解析 ID
        # 去掉倍数因子
        #print(self.blocks_per_volume)
        #exit()
        
        real_id = id % (len(self.file_list)  * self.blocks_per_volume)
        
        # 计算 Block 索引 (0 到 7)
        block_idx = real_id % self.blocks_per_volume
        remain = real_id // self.blocks_per_volume
    
        channel = 0
        index = remain
        #print("start read",index)
        key = self.file_list[index]
        #print(index,key)
        bz = block_idx // (self.grid_dim * self.grid_dim)
        rem_z = block_idx % (self.grid_dim * self.grid_dim)
        by = rem_z // self.grid_dim
        bx = rem_z % self.grid_dim

        sx = bx * self.crop_size
        sy = by * self.crop_size
        sz = bz * self.crop_size
        
        ex = sx + self.crop_size
        ey = sy + self.crop_size
        ez = sz + self.crop_size

        targets = load_pklzst(self.target_path + "/" + key)
        light = load_pklzst(self.light_path + "/" + key)
        

        box = self.bias_info[key.split(".")[0]]
        mask = generate_voxel_mask_6d(box,self.pos)
        #print(targets.shape)
        targets = targets.transpose(1,2,0,3,4,5)
        mask = mask.transpose(1,2,0,3,4,5)

        target_crop = targets[sx:ex, sy:ey, sz:ez, ...] # 裁切对应的 32x32x32 区域
        mask_crop = mask[sx:ex, sy:ey, sz:ez, ...] # 裁切对应的 32x32x32 区域

        # 5. 返回坐标信息，以便模型知道我们在训练哪一块
        coords = torch.tensor([sx, ex, sy, ey, sz, ez], dtype=torch.long)
        id = real_id // self.blocks_per_volume
        if self.stage==1:
            plane = load_pklzst(self.plane_path + key.split(".")[0] + ".pkl.zst")
            #plane = load_pklzst(self.plane_path + str(id) + ".pkl.zst")

            light["plane"] = plane
        light["direction"] = self.dir
        #print("read after",id)
        #print(key)
        return torch.tensor(id, dtype=torch.long), target_crop, mask_crop,light, coords,key



def save_compressed_pickle(data, file_path):
    # 将数据序列化为 pickle 格式
    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)

    # 创建 zstd 压缩器
    cctx = zstd.ZstdCompressor()

    # 压缩数据
    compressed_data = cctx.compress(pickled_data)
    
    # 将压缩后的数据写入文件
    with open(file_path, 'wb') as f:
        f.write(compressed_data)
def process_one_batch(
    batch_idx, batch_data, model_engine, device, STAGE, args, epoch, W, H, LABEL,
    local_tri_planes=None,
    local_optimizer=None,
    local_scheduler=None,
    start_idx = None,
    update_decoder=True,
    end_of_last_iter=0.0,
    step=0  # 🔥 新增：传入全局 step 用于打印
):

    #print("start_forward start idx",start_idx)
    # ── 计时器初始化 ──────────────────────────────────────────
    _t = {}
    def tick(name): _t[name] = time.perf_counter()
    def tock(name): return time.perf_counter() - _t[name]
    # ─────────────────────────────────────────────────────────

    tick("total")

    # 1. 解包数据
    tick("unpack")
    indices, targets, mask, lights, coords,key = batch_data
    for param in model_engine.module.parameters():
        param.requires_grad = update_decoder
    t_unpack = tock("unpack")

    # 2. 数据转移至 Device
    tick("to_device")
    key = key[0]
    indices = indices.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    BB, WW, HH, DD, _, _, CC = targets.shape
    
    radiance = lights["radiance"].to(device, non_blocking=True)
    position = lights["position"].to(device, non_blocking=True)
    direction = lights["direction"].to(device, non_blocking=True)
    B = radiance.shape[0]

    torch.cuda.synchronize()  # 确保 to() 真正完成
    t_to_device = tock("to_device")

    # 3. 归一化
    tick("normalize")
    if check_nan(radiance, "Original Radiance", args.local_rank): exit()
    
    maxi = radiance.reshape(radiance.shape[0], -1, radiance.shape[-1]).max(dim=1).values
    if (maxi <= 0).any():
        print(f"⚠️ [Rank {args.local_rank}] Detected zero or negative maxi! Value: {maxi}")
        maxi = maxi.clamp(min=1e-6)

    radiance = radiance / maxi.reshape(-1, 1, 1, 1, 1, radiance.shape[-1])
    

    targets_afternorm = targets / maxi.reshape(-1, 1, 1, 1, 1, 1, radiance.shape[-1])
    if check_nan(targets_afternorm, "Targets After Norm", args.local_rank): exit()

    targets_masked = torch.where(mask, 0, targets_afternorm)
    targets = fill_masked_voxels_batch(targets_masked.reshape(BB, WW, HH, DD, -1), mask[:,:,:,:,0,0,:]).reshape(BB, WW, HH, DD, 8, 8, 3)

    radiance = radiance.permute(0,-1,1,2,3,4).reshape(B*3,*radiance.shape[1:-1],1)
    position = position.unsqueeze(1).repeat(1,3,1,1,1,1,1).reshape(B*3,*position.shape[1:])
    direction = direction.unsqueeze(1).repeat(1,3,1,1,1,1,1).reshape(B*3,*direction.shape[1:])

    lights_tensor = torch.cat([radiance, position,direction], dim=-1).float()
    if check_nan(lights_tensor, "Normalized Lights", args.local_rank): exit()
  
 
    if check_nan(targets, "Filled Targets", args.local_rank): exit()
    torch.cuda.synchronize()
    t_normalize = tock("normalize")

    indices = indices - start_idx

    # 4. 前向传播
    tick("forward")
    distill_loss = None
    weights = None
    if STAGE == 0:
        #print(indices,local_tri_planes.shape[0])
        if local_tri_planes is not None:
            current_batch_planes = local_tri_planes[indices]
        else:
            current_batch_planes = None
        if check_nan(current_batch_planes, "Current Batch Planes", args.local_rank): exit()
        outputs, loss, visual_loss, plane_loss, weights = \
            model_engine.module.forward_decoder(current_batch_planes, targets, coords)
    else:
        current_batch_planes = lights["plane"].to(device, non_blocking=True)
        current_batch_planes = current_batch_planes.reshape(current_batch_planes.shape[0] * 3, *current_batch_planes.shape[2:])


        outputs,outputs_from_plane, loss, visual_loss, distill_loss, plane_loss, generated_plane = \
            model_engine.module.forward_encoder(
                lights_tensor, targets, coords=coords, return_planes=True,
                target_planes_gt=current_batch_planes
            )
        #exit()
    torch.cuda.synchronize()
    t_forward = tock("forward")

    if check_nan(outputs, "Model Outputs", args.local_rank): exit()
    # if True:
    #     print(key)
    #     #save_compressed_pickle(generated_plane.detach().cpu().numpy(),r"../datasets_plane/160_5_16/" + key)
    #     now_plane =generated_plane.detach().cpu().numpy()
    #     old_generated_plane = load_pklzst(r"../datasets_plane/160_5_16/" + key)
    #     print("mean ",numpy.abs(old_generated_plane).mean(),numpy.abs((now_plane-old_generated_plane)).mean())
    # 提取 Loss（在 backward 前）
    loss_val = loss.item()
    visual_loss_val = visual_loss.item() if visual_loss != 0 else 0
    plane_loss_val = plane_loss.item()
    distill_loss_val = distill_loss.item() if distill_loss is not None else None

    # 5. 反向传播
    tick("backward")
    model_engine.backward(loss)
    torch.cuda.synchronize()
    t_backward = tock("backward")

    # 6. 参数更新
    tick("step")
    model_engine.step()
    if STAGE == 0:
        local_optimizer.step()
        local_optimizer.zero_grad()
    torch.cuda.synchronize()
    t_step = tock("step")

    # 7. 保存 EXR
    tick("save")
    if epoch % SAVE_INTERVAL == 0 and batch_idx <3 and args.local_rank == 0 :
        vol_2d = volume_to_single_2d(outputs[0].float().detach().cpu(), W, H)
        tar_2d = volume_to_single_2d(targets[0].float().detach().cpu(), W, H)
        if STAGE==1:
            outputs_from_plane_2d = volume_to_single_2d(outputs_from_plane[0].float().detach().cpu(), W, H)
            pyexr.write(f"../{STAGE}Stage/output2/{LABEL}/00output_epoch_{epoch+1}_batch_{batch_idx+1}_planes.exr", outputs_from_plane_2d.numpy())
        pyexr.write(f"../{STAGE}Stage/output2/{LABEL}/00output_epoch_{epoch+1}_batch_{batch_idx+1}.exr", vol_2d.numpy())
        pyexr.write(f"../{STAGE}Stage/output2/{LABEL}/00output_epoch_{epoch+1}_batch_{batch_idx+1}_gt.exr", tar_2d.numpy())
        if weights is not None:
            weight_2d = volume_to_single_2d(weights[0].detach().cpu(), 1, 1)
            pyexr.write(f"../{STAGE}Stage/output2/{LABEL}/00output_epoch_{epoch+1}_batch_{batch_idx+1}_weight.exr", weight_2d.numpy())
    # if epoch % SAVE_INTERVAL == 0 and batch_idx < 5 and args.local_rank == 0 and update_decoder and STAGE ==1:
    #     for bat in range(3):
    #         for tri in range(3):
    #             pyexr.write(f"../{STAGE}Stage/output2/{LABEL}/00output_epoch_{epoch+1}_batch_{batch_idx+1}_{bat}_{tri}.exr", generated_plane[bat,tri,:3,...].permute(1,2,0).detach().cpu().numpy())
    #             pyexr.write(f"../{STAGE}Stage/output2/{LABEL}/00output_epoch_{epoch+1}_batch_{batch_idx+1}_{bat}_{tri}_gt.exr", current_batch_planes[bat,tri,:3,...].permute(1,2,0).detach().cpu().numpy())
    
    t_save = tock("save")

    t_total = tock("total")

    # ── 每 20 步打印一次，只在 Rank 0 打印 ──────────────────
    if step % 20 == 0 and args.local_rank == 0:
        print(
            f"\n[Step {step:06d}] Total: {t_total:.3f}s\n"
            f"  unpack    : {t_unpack:.3f}s\n"
            f"  to_device : {t_to_device:.3f}s\n"
            f"  normalize : {t_normalize:.3f}s\n"
            f"  forward   : {t_forward:.3f}s\n"
            f"  backward  : {t_backward:.3f}s\n"
            f"  step      : {t_step:.3f}s\n"
            f"  save      : {t_save:.3f}s\n"
        )
    # ─────────────────────────────────────────────────────────

    del outputs, loss, visual_loss, plane_loss
    if distill_loss is not None:
        del distill_loss
    if STAGE == 1 and 'generated_plane' in locals():
        del generated_plane
    #print("after_forward")
    return loss_val, visual_loss_val, plane_loss_val, distill_loss_val


def get_infinite_dataloader(dataloader):

    """无限循环提取数据的生成器"""
    while True:
        # 每次原本的 dataloader 耗尽时，自动无缝重启
        for batch in dataloader:
            yield batch

# =========================================================================
# 🌟 新增：混合重要性采样核心算法
# =========================================================================
def compute_hybrid_weights(losses, alpha=0.1, T=0.5):
    """
    结合均匀探索(alpha)与基于Loss平滑利用的采样权重计算。
    """
    N = len(losses)
    # 1. 基础保底概率
    base_probs = torch.ones(N, dtype=torch.float32, device=losses.device) * (alpha / N)
    
    # 2. 带有温度T平滑的剩余概率分布
    epsilon = 1e-8
    smoothed_losses = losses ** T
    loss_sum = smoothed_losses.sum() + epsilon
    loss_probs = (1.0 - alpha) * (smoothed_losses / loss_sum)
    
    return base_probs + loss_probs

def create_weighted_iterator(local_dataset, local_scene_loss_history, dataset_scale, batch_size, num_workers=8):
    """
    根据历史 Loss 动态创建加权采样的 DataLoader Iterator。
    """
    # 计算 Scene 级别的权重
    scene_weights = compute_hybrid_weights(local_scene_loss_history, alpha=0.1, T=0.5)
    
    # 扩展铺平到 View 级别 (每个 Scene 对应 dataset_scale 个样本)
    view_weights = torch.repeat_interleave(scene_weights, dataset_scale)
    
    sampler = WeightedRandomSampler(
        weights=view_weights,
        num_samples=len(local_dataset), 
        replacement=True
    )

    loader = DataLoader(
        local_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False, # ⚠️ 必须为 False，因为我们会定期销毁重建此 Loader
        prefetch_factor=4
    )
    return iter(loader)

if __name__ == "__main__":
    STAGE = 1
    BATCH_SIZE = 1
    
    
    angular_resolution = 64
    space_resolution = 8
    W, H = space_resolution, space_resolution
    NUM_EPOCHS = 50000
    if STAGE==0:
        CROP_SIZE = 32
    else:
        CROP_SIZE = 64
    warmup_min_lr = 2e-4
    warmup_max_lr = 2e-4
    warmup_steps = 1000
    C = 192                      
    if CHANNEL_CUT:
        C=64
    
    #LABEL = f"{STAGE}_{warmup_max_lr}_{warmup_min_lr}_CUT64_32_NEWLIGHTINFO_128_16"
    LABEL = f"160_5_18_NOGN_{STAGE}_{warmup_max_lr}_{warmup_min_lr}_PLANE{angular_resolution}_{space_resolution}_"
    PLANE_LEBEL =r"160_5_16"
    #RESUME_LABLE = r"FULL_1_0.0002_0.0002_PLANE64_8_"
    RESUME_LABLE = r"160_5_16_NOGN_1_0.0002_0.0002_PLANE64_8_"
    #RESUME_LABLE = r"NONE"

    # 1. 准备数据
    print("Loading Data...")
    model = TriplaneImageReconstructor(STAGE,plane_res=angular_resolution, plane_dim=C)
    if STAGE == 1:
        DECODER_EPOCH = 1170
        model_file_path = (
            f"../1Stage/ckpts/{RESUME_LABLE}/"
            f"epoch_{DECODER_EPOCH}/mp_rank_00_model_states.pt"
        )

        # =====================================================
        # 1. 先加载 Stage1 的完整模型参数
        # =====================================================
        if os.path.exists(model_file_path):
            print(f"🔄 正在读取完整模型参数: {model_file_path} ...")

            checkpoint = torch.load(model_file_path, map_location="cpu")
            state_dict = checkpoint["module"]

            # 去掉 DeepSpeed 外层 module. 前缀
            clean_state_dict = {
                k.removeprefix("module."): v
                for k, v in state_dict.items()
            }

            try:
                model.load_state_dict(clean_state_dict, strict=True)
                print("✅ Stage1 完整模型参数加载成功！")
            except Exception as e:
                print(f"❌ Stage1 完整模型严格加载失败: {e}")
                raise e
        else:
            print("⚠️ 未找到完整模型权重，将从头初始化。")

        # =====================================================
        # 2. 再单独加载另一份 decoder 参数，并覆盖当前 decoder
        # =====================================================
        decoder_path = (
            r"/seaweedfs_tmp/training/wangjiu/new/general_shading/0Stage/ckpts/160_CONTINUE_NOGN_0_0.0002_0.0002_PLANE64_8_/epoch_506/mp_rank_00_model_states.pt"
        )

        if os.path.exists(decoder_path):
            print(f"🔄 正在读取新的 decoder 参数: {decoder_path} ...")

            decoder_checkpoint = torch.load(decoder_path, map_location="cpu")
            decoder_full_state_dict = decoder_checkpoint["module"]

            # 先去掉 DeepSpeed 的 module. 前缀
            decoder_clean_state_dict = {
                k.removeprefix("module."): v
                for k, v in decoder_full_state_dict.items()
            }

            # 只保留 decoder.xxx，并去掉前缀 decoder.
            decoder_state_dict = {
                k.removeprefix("decoder."): v
                for k, v in decoder_clean_state_dict.items()
                if k.startswith("decoder.")
            }

            if len(decoder_state_dict) == 0:
                raise RuntimeError(
                    "❌ 没有在 decoder checkpoint 中找到以 'decoder.' 开头的参数！"
                )

            try:
                model.decoder.load_state_dict(decoder_state_dict, strict=True)
                print("✅ decoder 参数已成功替换！")
            except Exception as e:
                print(f"❌ decoder 参数严格加载失败: {e}")
                raise e
        else:
            print("⚠️ 未找到 decoder 权重文件，保留原 Stage1 模型中的 decoder。")

        # =====================================================
        # 3. 冻结替换后的 decoder
        # =====================================================
        for param in model.decoder.parameters():
            param.requires_grad = False

        model.decoder.eval()

        print("❄️ decoder 已替换、冻结，并切换至 eval 模式。")
    start_epoch = 0
    #RESUME_PATH = None
    RESUME_EPOCH = 132
    save_dir = f"../{STAGE}Stage/ckpts/{RESUME_LABLE}"
    model_save_path = f"{save_dir}/triplane_model_epoch_{RESUME_EPOCH}.pth"
    rank = args.local_rank
    plane_save_path = os.path.join(save_dir, f"tri_planes_epoch_{RESUME_EPOCH}_rank_{rank}.pt")
    tag = f"epoch_{RESUME_EPOCH}"
    # ==========================================
    # 1. 加载共享模型 (Decoder/Encoder)
    # ==========================================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=warmup_max_lr)

    def lr_lambda(step):
        if step < warmup_steps:
            return (warmup_min_lr + (warmup_max_lr - warmup_min_lr) * step / warmup_steps) / warmup_max_lr
        else:
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ==================== DeepSpeed Config ====================
    ds_config = {
        "train_batch_size": BATCH_SIZE * int(os.environ.get("WORLD_SIZE", 1)), # 全局 Batch Size
        "train_micro_batch_size_per_gpu": BATCH_SIZE, # 单卡 Batch Size
        "steps_per_print": 100,
        
        "optimizer": {
            "type": "Adam", 
            "params": {
                "lr": warmup_max_lr,
            }
        },
        
        # 【修改点 1】关闭混合精度，使用纯 FP32
        # 🛑 1. 明确关闭 FP16 (因为 FP16 容易数值溢出)
        "fp16": {
            "enabled": False, 
        },
        "zero_optimization": {
            "stage": 0, 
        }
    }
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_params,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config
    )
    checkpoint_tag_dir = os.path.join(save_dir, tag) # 检查这个文件夹是否存在
    print(checkpoint_tag_dir)
    if STAGE==0:
       
        if os.path.exists(checkpoint_tag_dir):
            if args.local_rank == 0:
                print(f"🔄 正在从 {checkpoint_tag_dir} 恢复 DeepSpeed 存档...")
            
            _, client_state = model_engine.load_checkpoint(save_dir, tag=tag,load_optimizer_states=False,load_lr_scheduler_states=False )
            if client_state is not None:
                print(f"✅ 成功从 {save_dir}/{tag} 恢复训练！")
            else:
                print(f"⚠️ 未找到 DeepSpeed 存档，将从头开始训练。")
        else:
            if args.local_rank == 0:
                print("⚠️ 未找到模型权重文件，将从头开始训练。")

    torch.distributed.barrier()
    world_size = int(os.environ.get("WORLD_SIZE", 1))


    # 🛑 再次同步
    torch.distributed.barrier()
    if STAGE == 1:
        model.freeze_decoder_for_stage1()

    # 3. 优化器和 Scheduler
    

    # 获取 DeepSpeed 分配的设备
    device = model_engine.device
    path = r"../datasets2/TogLightAll/"
    light_path = r"../datasets2/TogLightAll8x128/"
    # ==================== 分布式 DataLoader ====================
    # 必须使用 DistributedSampler 确保每张卡拿到不同的数据
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    dataset_scale = 64 // CROP_SIZE
    dataset_scale = dataset_scale ** 3
    print(dataset_scale)
    #exit()
    # 2. 实例化完整的全局数据集
    full_dataset = CutDataset(path, light_path, STAGE,PLANE_LEBEL,volume_size=64, crop_size=CROP_SIZE,channel_cut = CHANNEL_CUT)
    full_testDataset = CutDataset(path, light_path, STAGE,PLANE_LEBEL,volume_size=64, crop_size=64,channel_cut = CHANNEL_CUT)
    # 3. 计算当前显卡 (Rank) 应该负责的专属索引范围
    
    total_views = len(full_dataset)
    total_scenes_global = total_views // dataset_scale
    
    # 2. 计算当前显卡 (Rank) 应该负责的【场景级】索引范围
    scenes_per_gpu = total_scenes_global // world_size
    # scene_start_idx 是该卡负责的第一个场景 ID
    scene_start_idx = rank * scenes_per_gpu
    
    # 处理不能整除的余数情况，分给最后一张卡
    if rank == world_size - 1:
        scene_end_idx = total_scenes_global
    else:
        scene_end_idx = scene_start_idx + scenes_per_gpu

    # 当前卡负责的实际场景数量
    local_scene_count = scene_end_idx - scene_start_idx
    
    # 3. 将【场景范围】还原回【View 范围】（用于 Subset 数据集）
    # 每一张卡拿到的 view_indices 必须是 8 的倍数，这样 indices // 8 才不会错位
    view_start_idx = scene_start_idx * dataset_scale
    view_end_idx = scene_end_idx * dataset_scale
    local_view_indices = list(range(view_start_idx, view_end_idx))

    # 4. 实例化本地数据集
    local_dataset = Subset(full_dataset, local_view_indices)
    local_testDataset = Subset(full_testDataset, local_view_indices) 

    # 打印调试信息，确保万无一失
    print(f"[Rank {rank}] View Range: {view_start_idx}-{view_end_idx} ({len(local_dataset)} views)")
    print(f"[Rank {rank}] Scene Range: {scene_start_idx}-{scene_end_idx} ({local_scene_count} scenes)")
    # 5. 定义参数池
    # 参数池的大小直接等于该卡负责的场景数
    local_N = local_scene_count 
    SAVE_INTERVAL = 1800 //(local_N * BATCH_SIZE)
    
    plane_W = plane_H = angular_resolution
    scene_offset = scene_start_idx
    local_tri_planes = None
    local_optimizer = None
    local_scheduler = None
    if STAGE==0:
        local_tri_planes = nn.Parameter(
            torch.empty(local_N,3, 3, C, plane_W, plane_H, device=device)
        )

        if os.path.exists(plane_save_path):
            print(f"[Rank {rank}] 🔄 正在恢复本地三平面参数...")
            local_planes_state = torch.load(plane_save_path, map_location='cpu', weights_only=False)
            
            # ... 省略你的世界大小校验代码 ...
            
            # 🌟 修复关键：千万不要写 .to(device)！保持它在 CPU 上！
            planes_data_cpu = local_planes_state["planes_data"] 
            
            with torch.no_grad():
                # copy_ 支持直接将 CPU 数据拷贝到已经分配好显存的 GPU Parameter 中
                # 整个过程不会在 GPU 上产生第二份拷贝！
                local_tri_planes.copy_(planes_data_cpu)
                
            print(f"[Rank {rank}] ✅ 三平面恢复成功！")
            
            # 🧹 手动清理垃圾，释放 CPU 内存和潜在的 GPU 碎片
            del local_planes_state, planes_data_cpu
            torch.cuda.empty_cache()
        else:
            # 从头训练时的初始化
            with torch.no_grad():
                local_tri_planes.normal_(mean=0, std=0.1)
        local_optimizer = torch.optim.Adam(
            [local_tri_planes], 
            lr=warmup_max_lr*5 ,
            weight_decay=0,
            fused=True  # 🔥 极限省去中间临时显存，还能提速约 10%
        )
        
        local_scheduler = LambdaLR(local_optimizer, lr_lambda)
    #exit()

    infinite_sampler = InfiniteRandomSampler(local_dataset)

    dataloader = DataLoader(
        local_dataset,
        batch_size=BATCH_SIZE,
        sampler=infinite_sampler,   # 🔥 使用无限采样器
        # shuffle=True,             # 🛑 必须注释掉！因为 sampler 本身已经包含了乱序逻辑
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,    # 配合无限采样器，Worker 会真正永不歇息
        prefetch_factor=4
    )

    testDataloader = DataLoader(
        local_testDataset,
        batch_size=1,
        shuffle=False,              # 测试集通常不需要打乱
        num_workers=4,              # 🔥 维持你的高性能配置
        pin_memory=True,            # 🔥
        persistent_workers=True,    # 🔥
        prefetch_factor=4           # 🔥
    )

    if args.local_rank == 0: # 只在主进程写日志
        writer = SummaryWriter(
            log_dir=f"../{STAGE}Stage/runs/{LABEL}",
            max_queue=100,        # 积攒100条再一起写
            flush_secs=60         # 最多60秒flush一次
        )
        os.makedirs(f"../{STAGE}Stage/output2/{LABEL}",exist_ok=True)
        os.makedirs(f"../{STAGE}Stage/ckpts/{LABEL}",exist_ok=True)

    import time
    print(f"Start Training on Rank {args.local_rank}...")

    steps_per_epoch = len(dataloader)  
    TOTAL_STEPS = NUM_EPOCHS * steps_per_epoch  
    
    # 2. 直接获取迭代器，它现在是一个真正的无限黑洞
    infinite_batch_iterator = iter(dataloader)
    
    model_engine.train()
    start_time = time.time()
    total_loss, total_visual_loss, total_plane_loss, total_distill_loss = 0, 0, 0, 0
    
    from tqdm import tqdm
    pbar = tqdm(range(TOTAL_STEPS), desc="Training Steps", disable=(args.local_rank != 0))
    num_inner_steps = 8
    # ==================== 🚀 极致丝滑的训练循环 🚀 ====================
    first_epoch = False
    #exit()
    for step in pbar:
    # 1. 拿到一个 Batch（预处理后的数据，包含 32x32x32 的裁剪块）
        batch_data = next(infinite_batch_iterator)
        current_epoch = step // steps_per_epoch 


        num_inner_steps = 1
        # --- 核心修改：增加局部 8 次迭代 ---
        for i in range(num_inner_steps):
            is_last_step = (i == num_inner_steps - 1)
            
            # 调用 process_one_batch
            # 关键：传入 update_decoder 标志位（前 7 次 False，最后一次 True）
            loss_v, vis_loss_v, plane_loss_v, dist_loss_v = process_one_batch(
                batch_idx=(step % steps_per_epoch),
                batch_data=batch_data,
                model_engine=model_engine,
                device=device,
                STAGE=STAGE,
                args=args,
                epoch=current_epoch,
                W=W, H=H, LABEL=LABEL,
                local_tri_planes=local_tri_planes,
                local_optimizer=local_optimizer,
                local_scheduler=local_scheduler,
                start_idx=scene_offset,
                update_decoder=is_last_step,
                  step = step  # 🔥 确保你的 process_one_batch 接收这个参数
            )
        
        if step % 20 == 0 and args.local_rank == 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            import time
            print(f"Step {step} | Time: {time.time()-start_time:.1f}s | "
          f"Alloc: {allocated:.3f}GB | Reserved: {reserved:.3f}GB")
        if args.local_rank == 0:
            print(current_epoch,step,steps_per_epoch)
        # 2. 累加最后一次迭代产生的有效 Loss 用于显示
        total_loss += loss_v
        total_visual_loss += vis_loss_v
        total_plane_loss += plane_loss_v
        if dist_loss_v is not None: total_distill_loss += dist_loss_v
        if dist_loss_v == None:
            dist_loss_v = 0
        # 3. 实时更新面板
        if args.local_rank == 0:
            postfix_dict = {
                "Epoch": f"{current_epoch+1}/{NUM_EPOCHS}",
                "Loss": f"{loss_v:.4f}",
                "Plane": f"{plane_loss_v:.4f}",
                "Distill": f"{dist_loss_v:.4f}"
            }
            pbar.set_postfix(postfix_dict)
        is_end_of_epoch = ((step + 1) % steps_per_epoch == 0)
        
        if is_end_of_epoch:
            local_avg_loss = total_loss / steps_per_epoch
            local_avg_visual_loss = total_visual_loss / steps_per_epoch
            local_avg_plane_loss = total_plane_loss / steps_per_epoch
            local_avg_distill_loss = total_distill_loss / steps_per_epoch
            global_avg_loss = reduce_mean(local_avg_loss, device)
            global_avg_visual_loss = reduce_mean(local_avg_visual_loss, device)
            global_avg_plane_loss = reduce_mean(local_avg_plane_loss, device)
            global_avg_distill_loss = reduce_mean(local_avg_distill_loss, device)

            if args.local_rank == 0 and current_epoch % SAVE_INTERVAL == 0:
                print(
                    f"\n[Epoch {current_epoch+1}] "
                    f"Loss: {global_avg_loss:.6f} | "
                    f"Visual: {global_avg_visual_loss:.6f} | "
                    f"Plane: {global_avg_plane_loss:.6f} | "
                    f"Distill: {global_avg_distill_loss:.6f}"
                )
                loss_val_py = float(global_avg_loss.item()) if isinstance(global_avg_loss, torch.Tensor) else float(global_avg_loss)
                vis_loss_py = float(global_avg_visual_loss.item()) if isinstance(global_avg_visual_loss, torch.Tensor) else float(global_avg_visual_loss)
                plane_loss_py = float(global_avg_plane_loss.item()) if isinstance(global_avg_plane_loss, torch.Tensor) else float(global_avg_plane_loss)
                
                # 2. 写入极其轻量的原生数值
                writer.add_scalar("train/loss_epoch", loss_val_py, current_epoch)
                writer.add_scalar("train/visual_loss_epoch", vis_loss_py, current_epoch)
                writer.add_scalar("train/plane_loss_epoch", plane_loss_py, current_epoch)
                
                if global_avg_distill_loss is not None:
                    dist_loss_py = float(global_avg_distill_loss.item()) if isinstance(global_avg_distill_loss, torch.Tensor) else float(global_avg_distill_loss)
                    writer.add_scalar("train/distill_loss_epoch", dist_loss_py, current_epoch)
                
            # 4. 分布式保存模型与三平面 (每 20 个 Epoch)
            if (current_epoch + 1) % SAVE_INTERVAL == 0:
                save_dir = f"../{STAGE}Stage/ckpts/{LABEL}"
                # (A) Rank 0 负责创建目录和保存共享模型
                if args.local_rank == 0:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir, tag=f"epoch_{current_epoch+1}")
                # 🛑 强制同步：确保文件夹已建好
                torch.distributed.barrier()

                if STAGE ==0:
                    local_planes_state = {
                        "epoch": current_epoch + 1,
                        "rank": rank,
                        "world_size": world_size,
                        "local_N": local_N,             # 本地三平面数量
                        "planes_data": local_tri_planes.detach().cpu().clone()
                    }
                    
                    plane_save_path = os.path.join(save_dir, f"tri_planes_epoch_{current_epoch+1}_rank_{rank}.pt")
                    torch.save(local_planes_state, plane_save_path)
                
                if args.local_rank == 0:
                    print(f"Local Tri-planes chunks saved at epoch {current_epoch+1}")

                # 🛑 再次强制同步：确保所有卡都完成了 IO 写入
                torch.distributed.barrier()

            # 5. 重置累加器，为下一个 Epoch 做准备
            total_loss = 0
            total_visual_loss = 0
            total_plane_loss = 0
            total_distill_loss = 0
