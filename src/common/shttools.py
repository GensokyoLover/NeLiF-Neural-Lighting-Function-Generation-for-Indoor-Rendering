import abc
from dataclasses import dataclass
from functools import cached_property

import numpy
import numpy as np
import math
import torch
#import open3d as o3d
import os
from PIL import Image
import pyexr
import pickle
import zstandard as zstd
import zarr
import blosc2 as blosc

import numpy as np

def normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x) + eps)


def build_camera_basis(forward):
    forward = normalize(np.asarray(forward, dtype=np.float32))

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(forward, right))

    return right, up, forward


def project_world_to_uv(
    world_pos,
    cam_position,
    cam_forward,
    fov_y_deg,
    width,
    height,
    y_down=True,
):
    """
    world_pos: [..., 3]
    return: uv [..., 2], valid [...]
    """

    world_pos = np.asarray(world_pos, dtype=np.float32)
    cam_position = np.asarray(cam_position, dtype=np.float32)

    right, up, forward = build_camera_basis(cam_forward)

    vTan = np.tan(np.deg2rad(fov_y_deg * 0.5))
    hTan = vTan * (width / height)

    d = world_pos - cam_position

    x_cam = np.sum(d * right, axis=-1)
    y_cam = np.sum(d * up, axis=-1)
    z_cam = np.sum(d * forward, axis=-1)

    valid = z_cam > 1e-6

    x_ndc = x_cam / (z_cam * hTan + 1e-8)
    y_ndc = y_cam / (z_cam * vTan + 1e-8)

    u = x_ndc * 0.5 + 0.5

    if y_down:
        v = 0.5 - y_ndc * 0.5
    else:
        v = y_ndc * 0.5 + 0.5

    uv = np.stack([u, v], axis=-1)

    valid = (
        valid &
        (uv[..., 0] >= 0.0) & (uv[..., 0] <= 1.0) &
        (uv[..., 1] >= 0.0) & (uv[..., 1] <= 1.0)
    )

    return uv, valid
def check_data_valid_tensor(data):
    """
    检查并修复字典中的 NaN 和 Inf 值 (PyTorch Tensor 版本)。
    :return: 如果数据健康且未修改，返回 True；如果发生了修复，返回 False
    """
    is_fix = False
    
    for k, v in data.items():
        if isinstance(v, dict):
            # 递归检查子字典
            if not check_data_valid_tensor(v):
                is_fix = True
                
        elif isinstance(v, torch.Tensor):
            # 仅处理浮点数类型 (float16, float32, float64等)
            # 因为整型 Tensor 在 PyTorch 中根本不可能出现 NaN/Inf，跳过可以提速
            if not v.is_floating_point():
                continue
                
            # 检查当前 tensor 中是否存在 NaN 或 Inf
            # .item() 将单个元素的 GPU/CPU tensor 转换为 Python 原生 bool，极其轻量
            has_nan = torch.isnan(v).any().item()
            has_inf = torch.isinf(v).any().item()
            
            if has_nan or has_inf:
                is_fix = True
                
                # 🚀 极致性能优化：使用 PyTorch 内置的底层原位替换 (注意末尾的下划线 _)
                # 这比写 v[torch.isnan(v)] = 0.5 速度快得多，且无需分配任何额外的掩码显存
                v.nan_to_num_(nan=0.5, posinf=0.5, neginf=0.5)
                
    return not is_fix # 发生修复返回 False，完全健康返回 True
def to_cuda(data):
    for k in data:
        #print(k)
        if isinstance(data[k], np.ndarray):
            data[k] = torch.from_numpy(data[k])
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].cuda(non_blocking=True)
         
            data[k] = data[k].float()
        elif isinstance(data[k], dict):
            data[k] = to_cuda(data[k])
        else:
            continue
    return data
def to_cuda_type(data, target_dtype=torch.bfloat16):
    """
    将数据批量送入 GPU 并进行极速类型转换。
    - 普通输入：转为 target_dtype (默认 BF16) 以节省 PCIe 带宽。
    - 包含 'shading' 的 Ground Truth：强制保持 float32，防止 Loss 精度丢失。
    """
    for k in data:
        # 1. 处理嵌套字典 (递归时记得把 target_dtype 传下去)
        if isinstance(data[k], dict):
            data[k] = to_cuda_type(data[k], target_dtype)
            continue
            
        # 2. 将 numpy 转换为 Tensor
        if isinstance(data[k], np.ndarray):
            data[k] = torch.from_numpy(data[k])
            
        # 3. 核心：设备迁移与精度分发
        if isinstance(data[k], torch.Tensor):
            # 🌟 关键拦截：如果 key 包含 shading，死锁在 FP32
            if "shading" in str(k).lower() or "position" in str(k).lower() or "scope" in str(k).lower() or "distance" in str(k).lower():
                data[k] = data[k].to(
                    device='cuda', 
                    dtype=torch.float32, 
                    non_blocking=True
                )
            # 🌟 其他所有输入：转成极速的 BF16 (或其他指定的 target_dtype)
            else:
                data[k] = data[k].to(
                    device='cuda', 
                    dtype=target_dtype, 
                    non_blocking=True
                )
    return data
def to_numpy(data):
    """
    递归遍历字典，将所有的 PyTorch Tensor (无论在 CPU 还是 GPU) 转换回 NumPy ndarray。
    """
    for k in data:
        if isinstance(data[k], torch.Tensor):
            # 必须严格按照 detach -> cpu -> numpy 的顺序
            # 1. .detach(): 剥离梯度计算图（如果它是模型输出的话必须要这一步）
            # 2. .cpu(): 从 GPU 显存拉回系统内存
            # 3. .numpy(): 转换为 numpy 数组
            data[k] = data[k].detach().cpu().numpy()
            
        elif isinstance(data[k], dict):
            # 递归处理嵌套字典
            data[k] = to_numpy(data[k])
            
        else:
            # 跳过普通标量、字符串、列表等其他类型
            continue
            
    return data

def save_compressed_pickle(data, file_path):
    # 将数据序列化为 pickle 格式
    pickled_data = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)

    # 创建 zstd 压缩器
    cctx = zstd.ZstdCompressor()

    # 压缩数据
    compressed_data = cctx.compress(pickled_data)
    #print(file_path)
    # 将压缩后的数据写入文件
    with open(file_path, 'wb') as f:
        f.write(compressed_data)
def check_data_valid(data):
    """
    检查并修复字典中的 NaN 和 Inf 值。
    :return: 如果数据健康且未修改，返回 True；如果发生了修复，返回 False
    """

    is_fix = False
    
    for k, v in data.items():
        if isinstance(v, dict):
            # 递归检查子字典
            if not check_data_valid(v):
                is_fix = True
                
        elif isinstance(v, np.ndarray):
            # 仅处理数值类型的数组，安全跳过字符串等类型
            if not np.issubdtype(v.dtype, np.number):
                continue
                
            # 1. 检查并修复 NaN
            nan_mask = np.isnan(v)
            if np.any(nan_mask):
                v[nan_mask] = 0.5 # $O(1)$ 内存的极速原位赋值
                is_fix = True
                
            # 2. 检查并修复 Inf (正无穷和负无穷)
            inf_mask = np.isinf(v)
            if np.any(inf_mask):
                v[inf_mask] = 0.0 # 同样极速赋值
                is_fix = True
                
    return not is_fix # 发生修复返回 False，完全健康返回 True
def load_compressed_pickle(file_path):
    dctx = zstd.ZstdDecompressor()

    with open(file_path, "rb") as compressed_file:
        with dctx.stream_reader(
            compressed_file
        ) as reader:
            data = pickle.load(reader)

    return data
def load_dat(filename):
    try:
        with open(filename, 'rb') as f:
                data_compressed = f.read()
    except Exception as e:
        print(f"Error when reading file: {filename}")
        print(f"Exception: {e}")
        sys.exit(1)  # 非零退出码，
    dctx = zstd.ZstdDecompressor()
    final_pickled_loaded = dctx.decompress(data_compressed)
    
    compressed_structure_loaded = pickle.loads(final_pickled_loaded)
    
    decompressed_data_loaded = {}
    for key, meta in compressed_structure_loaded.items():
        # 解压单个 Blosc2 块
        #decompressed_arr = blosc.decompress(meta['data'])
        raw = bytearray(blosc.decompress(meta['data']))
        decompressed_data_loaded[key] = np.frombuffer(
            raw, 
            dtype=meta['dtype']
        ).reshape(meta['shape']).copy()
        # 重塑并转换 dtype (因为 blosc.decompress 默认返回 bytes 或 flat array)
        # decompressed_data_loaded[key] = np.frombuffer(
        #     decompressed_arr, 
        #     dtype=meta['dtype']
        # ).reshape(meta['shape']).copy()
    return decompressed_data_loaded

def load_and_merge_tri_planes(save_dir, epoch, expected_world_size=None):
    """
    读取并合并所有 rank 的三平面分包`
    """
    print(f"🔍 正在从 {save_dir} 合并 Epoch {epoch} 的数据...")
    
    # 1. 自动寻找该 Epoch 所有的 rank 文件
    # 匹配模式: tri_planes_epoch_{epoch}_rank_{rank}.pt
    pattern = re.compile(rf"tri_planes_epoch_{epoch}_rank_(\d+)\.pt")
    
    files = os.listdir(save_dir)
    rank_files = []
    for f in files:
        match = pattern.match(f)
        if match:
            rank_idx = int(match.group(1))
            rank_files.append((rank_idx, f))
    
    # 2. 按 Rank 从小到大排序 (核心步骤)
    rank_files.sort(key=lambda x: x[0])
    
    if not rank_files:
        raise FileNotFoundError(f"❌ 未找到 Epoch {epoch} 的任何三平面分包！")
        
    actual_world_size = len(rank_files)
    if expected_world_size and actual_world_size != expected_world_size:
        print(f"⚠️ 警告: 预期的 World Size 是 {expected_world_size}，但实际只找到 {actual_world_size} 个分包")

    # 3. 逐个加载并放入列表
    all_chunks = []
    for rank, filename in rank_files:
        path = os.path.join(save_dir, filename)
        # map_location='cpu' 确保不会把显存撑爆
        state = torch.load(path, map_location='cpu')
        
        planes_data = state["planes_data"]
        print(f"   -> 加载 Rank {rank}: Shape {list(planes_data.shape)}")
        all_chunks.append(planes_data)

    # 4. 在维度 0 上进行合并 (Concatenate)
    full_tri_planes = torch.cat(all_chunks, dim=0)
    
    print(f"✅ 合并完成！最终 Shape: {list(full_tri_planes.shape)}")
    return full_tri_planes

def load_pklzst(filename):
    """
    统一读取 .pkl.zst 文件。
    自动支持：
        - 直接 pickle dump 的 dict
        - dict 内部带 blosc2 压缩 array 的结构

    如果读取、解压或 pickle 解析失败，会输出错误文件路径。
    """
    try:
        with open(filename, "rb") as f:
            compressed = f.read()

        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)

        data = pickle.loads(raw)
        return data

    except Exception as e:
        print("=" * 80)
        print("[ERROR] Failed to load pkl.zst file:")
        print(filename)
        print("Error type:", type(e).__name__)
        print("Error msg :", str(e))
        print("=" * 80)

        raise RuntimeError(
            f"Failed to load pkl.zst file: {filename} | "
            f"{type(e).__name__}: {str(e)}"
        ) from e
def WrapEqualAreaSquare(uv):
    if (uv[0] < 0):
        uv[0] = -uv[0]
        uv[1] = 1 - uv[1]
    elif (uv[0] > 1):
        uv[0] = 2 - uv[0]
        uv[1] = 1 - uv[1]
    if (uv[1] < 0):
        uv[0] = 1 - uv[0]
        uv[1] = -uv[1]
    elif (uv[1] > 1):
        uv[0] = 1 - uv[0]
        uv[1] = 2 - uv[1]
    return uv

def wrap_equal_area_square_tensor(uv):
    """
    uv: Tensor of shape (..., 2), each element in uv may be out of [0, 1] range.
    Returns a wrapped uv tensor of same shape, mapped back to [0, 1] with equal-area warping.
    """
    x, y = uv[..., 0], uv[..., 1]

    # Step 1: wrap x
    flip_x = (x < 0) | (x > 1)
    x = torch.where(x < 0, -x, x)
    x = torch.where(x > 1, 2 - x, x)

    # Step 2: flip y when x is wrapped
    y = torch.where(flip_x, 1 - y, y)

    # Step 3: wrap y
    flip_y = (y < 0) | (y > 1)
    y = torch.where(y < 0, -y, y)
    y = torch.where(y > 1, 2 - y, y)

    # Step 4: flip x when y is wrapped
    x = torch.where(flip_y, 1 - x, x)

    return torch.stack([x, y], dim=-1)

def cartesian_to_spherical_norm(tensor, r_min=0.1, r_max=7.67):
    """
    将笛卡尔坐标 (x,y,z) 转换为球面坐标 (r, theta, phi)，并全部归一化到 [0,1]
    输入:
        tensor: np.ndarray, shape (..., 3)
    输出:
        spherical_norm: np.ndarray, shape (..., 3), 各通道范围均为 [0,1]
    """
    x, y, z = np.moveaxis(tensor, -1, 0)

    # --- 原始球面坐标 ---
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-8), -1.0, 1.0))  # [0, π]
    phi = np.arctan2(y, x)                                 # [-π, π]

    # --- 归一化 ---
    r_norm = np.clip((r - r_min) / (r_max - r_min), 0.0, 1.0)
    theta_norm = theta / np.pi
    phi_norm = (phi + np.pi) / (2 * np.pi)

    spherical_norm = np.stack([r_norm, theta_norm, phi_norm], axis=-1)
    return spherical_norm


def spherical_to_cartesian_norm(spherical_norm, r_min=0.1, r_max=7.67):
    """
    将归一化后的球面坐标 (r_norm, theta_norm, phi_norm) 还原为笛卡尔坐标 (x,y,z)
    输入:
        spherical_norm: np.ndarray, shape (..., 3)，三个通道都在 [0,1]
    输出:
        tensor: np.ndarray, shape (..., 3)
    """
    r_norm, theta_norm, phi_norm = np.moveaxis(spherical_norm, -1, 0)

    # --- 反归一化 ---
    r = r_norm * (r_max - r_min) + r_min
    theta = theta_norm * np.pi
    phi = phi_norm * 2 * np.pi - np.pi

    # --- 转换为笛卡尔坐标 ---
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    tensor = np.stack([x, y, z], axis=-1)
    return tensor

def cartesian_to_spherical_norm_torch(tensor, r_min=0.1, r_max=7.67):
    """
    输入输出形状: (N, W, H, 3)
    将 (x, y, z) 转换为 [r_norm, theta_norm, phi_norm] ∈ [0,1]
    """
    assert tensor.shape[-1] == 3, "输入必须是 (N, W, H, 3)"

    x, y, z = tensor[..., 0], tensor[..., 1], tensor[..., 2]
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))
    phi = torch.atan2(y, x)

    r_norm = torch.clamp((r - r_min) / (r_max - r_min), 0.0, 1.0)
    theta_norm = theta / math.pi
    phi_norm = (phi + math.pi) / (2 * math.pi)

    return torch.stack([r_norm, theta_norm, phi_norm], dim=-1)


def spherical_to_cartesian_norm_torch(spherical, r_min=0.1, r_max=7.67):
    """
    输入输出形状: (N, W, H, 3)
    将 [r_norm, theta_norm, phi_norm] ∈ [0,1] 还原为 (x, y, z)
    """
    assert spherical.shape[-1] == 3, "输入必须是 (N, W, H, 3)"

    r_norm, theta_norm, phi_norm = spherical[..., 0], spherical[..., 1], spherical[..., 2]

    r = r_norm * (r_max - r_min) + r_min
    theta = theta_norm * math.pi
    phi = phi_norm * 2 * math.pi - math.pi

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    return torch.stack([x, y, z], dim=-1)

def equal_area_square_to_sphere(p):
    # p shape: (B, W, H, 2)
    u = 2 * p[..., 0] - 1  # shape (B, W, H)
    v = 2 * p[..., 1] - 1  # shape (B, W, H)

    up = torch.abs(u)
    vp = torch.abs(v)

    signed_distance = 1 - (up + vp)
    d = torch.abs(signed_distance)
    r = 1 - d

    # Avoid division by zero in phi calculation
    r_safe = torch.where(r == 0, torch.ones_like(r), r)
    phi = ((vp - up) / r_safe + 1) * (math.pi / 4)

    # For r == 0, set phi = pi/4
    phi = torch.where(r == 0, torch.full_like(phi, math.pi / 4), phi)

    z = torch.abs(1 - r * r)
    z = torch.where(signed_distance < 0, -z, z)

    cos_phi = torch.abs(torch.cos(phi))
    cos_phi = torch.where(u < 0, -cos_phi, cos_phi)

    sin_phi = torch.abs(torch.sin(phi))
    sin_phi = torch.where(v < 0, -sin_phi, sin_phi)

    factor = r * torch.sqrt(2 - r * r)
    x = cos_phi * factor
    y = sin_phi * factor

    result = torch.stack((x, y, z), dim=-1)  # shape (B, W, H, 3)
    return result

def EqualAreaSquareToSphere(p):
    u = 2 * p[0] - 1
    v = 2 * p[1] - 1
    up = math.fabs(u)
    vp = math.fabs(v)

    signedDistance = 1 - (up + vp)
    d = math.fabs(signedDistance)
    r = 1 - d
    if r == 0:
        phi = 1 * math.pi / 4
    else:
        phi = ((vp - up) / r + 1) * math.pi / 4
    z = math.fabs(1 - r * r)
    if (signedDistance < 0):
        z = -z;
    cosPhi = math.fabs(math.cos(phi))
    if (u < 0):
        cosPhi = - cosPhi
    sinPhi = math.fabs(math.sin(phi))
    if (v < 0):
        sinPhi = - sinPhi
    return (cosPhi * r * math.sqrt(2 - r * r), sinPhi * r * math.sqrt(2 - r * r), z)

def equal_area_square_to_sphere_tensor(p):
    """
    p: Tensor of shape (..., 2), values in [0, 1]
    Returns: Tensor of shape (..., 3), sphere-mapped coordinates
    """
    u = 2 * p[..., 0] - 1  # shape (...,)
    v = 2 * p[..., 1] - 1

    up = torch.abs(u)
    vp = torch.abs(v)

    signed_distance = 1 - (up + vp)
    d = torch.abs(signed_distance)
    r = 1 - d

    # avoid divide-by-zero
    eps = 1e-4
    r_safe = torch.where(r == 0, torch.full_like(r, eps), r)

    phi = ((vp - up) / r_safe + 1) * (math.pi / 4)
    phi = torch.where(r == 0, torch.full_like(phi, math.pi / 4), phi)

    z = torch.abs(1 - r * r)
    z = torch.where(signed_distance < 0, -z, z)

    cos_phi = torch.abs(torch.cos(phi))
    cos_phi = torch.where(u < 0, -cos_phi, cos_phi)

    sin_phi = torch.abs(torch.sin(phi))
    sin_phi = torch.where(v < 0, -sin_phi, sin_phi)

    xy_factor = r * torch.sqrt(torch.clamp(2 - r * r, min=0))
    x = cos_phi * xy_factor
    y = sin_phi * xy_factor

    return torch.stack([x, y, z], dim=-1)


def EqualAreaSphereToSquare(d):
    xyz = torch.abs(d)
    r = torch.sqrt(1 - xyz[...,2])
    a,_ = xyz[...,:2].max(dim=-1)
    b,_ = xyz[...,:2].min(dim=-1)
    a = torch.where(a == 0,1e-5,a)
    b = b / a


    phi = torch.atan(b) * 2 / math.pi

    phi = torch.where(xyz[...,0] < xyz[...,1], 1-phi, phi)
    v = phi * r
    u = r - v

    v2 = v
    u2 = u
    q = u2
    u2 = v2
    v2 = q
    u2 = 1 - u2
    v2 = 1 - v2
    u = torch.where(d[...,2] < 0,u2,u)
    v = torch.where(d[...,2] < 0,v2,v)
    #// Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = abs(u)
    u = torch.where(d[...,0] < 0,-u,u)
    v = abs(v)
    v = torch.where(d[...,1]<0,-v,v)
    '''if(d.x<0)
        u = -u;
    v = abs(v);
    if(d.y<0)
        v = -v;'''

    u = u.unsqueeze(dim = -1)
    v = v.unsqueeze(dim = -1)
    uv = torch.cat([u,v],dim = -1)
    return uv

def spherical_texture(radius, width, height):
    spherical_texture = numpy.zeros((width, height, 3))
    for i in range(width):
        for j in range(height):
            uv = [(i+0.5)/(width), (j+0.5)/(height)]
            uv = WrapEqualAreaSquare(uv)
            uv = list(uv)
            dir = EqualAreaSquareToSphere(uv)
            dir = np.array(list(dir))
            spherical_texture[i, j] = dir * radius

    return spherical_texture

def add_texture_border(texture):
    N,W,H,C = texture.shape
    tensor_border = torch.zeros((N,W+2,H+2,C)).cuda()
    tensor_border[:,1:-1,1:-1,:] = texture
    tensor_border[:, 0, 0, :] = tensor_border[:,W,H,:]
    tensor_border[:, W+1, H+1, :] = tensor_border[:,1,1,:]
    tensor_border[:, 0, H+1, :] = tensor_border[:,W,1,:]
    tensor_border[:, W+1, 0, :] = tensor_border[:,1,H,:]
    for i in range(W+2):
        if i == 0 or i == W+1:
            continue
        tensor_border[:,i,0,:] = tensor_border[:,W+1-i,1,:]
        tensor_border[:,i,H+1,:] = tensor_border[:,W+1-i,H,:]
    for i in range(H+2):
        if i == 0 or i == H+1:
            continue
        tensor_border[:,0,i,:] = tensor_border[:,1,H+1-i,:]
        tensor_border[:,W+1,i,:] = tensor_border[:,W,H+1-i,:]
    return tensor_border
def reright_uv(w,h,uv):
    uv[...,0] = uv[...,0] * (w/(w+2))
    uv[...,1] = uv[...,1] * (h/(h+2))
    return uv

def save_picture_from_dict(image_dict):
    for key in image_dict:
        image = image_dict[key]
        if type(image) != numpy.ndarray and type(image) != torch.Tensor:
            continue
        image = numpy.array(image)
        if image.shape[2] == 1:
            continue
        else:
            pyexr.write("../testdata/{}.exr".format(key),image)


def where_is_nan(image_dict):
    for key in image_dict:
        image = image_dict[key]
        if type(image) != numpy.ndarray and type(image) != torch.Tensor:
            continue
        image = numpy.array(image.cpu())
        if numpy.any(numpy.isnan(image)):
            print(key)
            for i in range(4):
                value = image[i,...]
                pyexr.write("./{}_{}.exr".format(key,i),value)
'''
def view_point_cloud(position):
    if len(list(position.shape)) == 3:
        position = position[0]
    if position.shape[0] == 3:
        position_new_view = position.permute(1, 0)
    else:
        position_new_view = position
    if isinstance(position_new_view, torch.Tensor):
        position_new_view = position_new_view.cpu().detach().numpy()
    pcd = o3d.geometry.PointCloud()
    # 将 NumPy 数组中的点赋值给点云
    pcd.points = o3d.utility.Vector3dVector(position_new_view)
    # 保存点云到文件
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    # 如果需要可视化点云
    o3d.visualization.draw_geometries([pcd])'''


def camera_perspective_dir(uv):
    # uv shape: (N, W, H, 2)

    # Scale from [0, 1] to [-1, 1]
    direction = uv * 2 - 1  # (N, W, H, 2)

    abs_x = direction[..., 0].abs()  # (N, W, H)
    abs_y = direction[..., 1].abs()  # (N, W, H)

    u = torch.max(abs_x, abs_y)  # (N, W, H)
    v = torch.min(abs_x, abs_y)  # (N, W, H)

    # Avoid division by zero
    phi = torch.zeros_like(u)  # (N, W, H)
    mask = u > 0.0
    phi[mask] = (torch.pi / 4.0) * (v[mask] / u[mask])

    # Compute spherical coordinates
    r = u  # (N, W, H)
    sqrt_term = torch.sqrt(2.0 - r * r)  # (N, W, H)

    x = torch.cos(phi) * r * sqrt_term  # (N, W, H)
    y = torch.sin(phi) * r * sqrt_term  # (N, W, H)
    z = 1.0 - r * r  # (N, W, H)

    # Swap x and y if abs_x < abs_y
    swap_mask = abs_x < abs_y  # (N, W, H)
    x, y = torch.where(swap_mask.unsqueeze(-1), y.unsqueeze(-1), x.unsqueeze(-1)), torch.where(swap_mask.unsqueeze(-1), x.unsqueeze(-1), y.unsqueeze(-1))

    # Restore direction
    x *= torch.sign(direction[..., 0:1])  # (N, W, H)
    y *= torch.sign(direction[..., 1:2])  # (N, W, H)

    # Stack to get the final direction (N, W, H, 3)
    direction_out = torch.cat([x, y, z.unsqueeze(-1)], dim=-1)

    # Normalize and return
    return torch.nn.functional.normalize(direction_out, p=2, dim=-1)

def concentric_mapping_hemisphere_3D_to_2D(direction):
    """
    批量将 3D 坐标 (w, h, 3) 映射回 2D (w, h, 2) (PyTorch 版本)

    参数:
    - direction: torch.Tensor, 形状为 (..., 3)，表示单位半球上的 3D 方向向量

    返回:
    - torch.Tensor, 形状为 (..., 2)，表示投影到 2D 平面的坐标
    """
    abs_x = torch.abs(direction[..., 0])
    abs_y = torch.abs(direction[..., 1])

    x = torch.maximum(abs_x, abs_y)
    y = torch.minimum(abs_x, abs_y)
    r = torch.sqrt(1.0 - direction[..., 2])
    alpha = torch.zeros_like(x)
    mask = x > 0
    alpha[mask] = y[mask] / x[mask]

    phi_2DivPI = (
            0.00000406531 +
            0.636227 * alpha +
            0.00615523 * alpha ** 2 -
            0.247326 * alpha ** 3 +
            0.0881627 * alpha ** 4 +
            0.0419157 * alpha ** 5 -
            0.0251427 * alpha ** 6
    )

    u = r
    v = 2.0 * phi_2DivPI * u

    # 交换 u, v 在 abs_x < abs_y 的情况下
    swap_mask = abs_x < abs_y
    u_temp, v_temp = u.clone(), v.clone()
    u[swap_mask] = v_temp[swap_mask]
    v[swap_mask] = u_temp[swap_mask]

    # 还原方向
    u *= torch.sign(direction[..., 0])
    v *= torch.sign(direction[..., 1])

    # 归一化到 [0,1] 作为最终输出
    return torch.stack([u, v], dim=-1)

def fill_tensor(tensor): # B,C,W,H
    B,C,W,H = tensor.shape
    for i in range(W):
        for j in range(H):
            tensor[0,0,i,j] = i * H + j
    return tensor

import itertools
def compute_4d_prefix_sum(x):
    # x: (B, C, H, W), must be float type to avoid overflow
    return x.cumsum(dim=0).cumsum(dim=1).cumsum(dim=2).cumsum(dim=3)


def batch_get_4d_region_sum(prefix, queries):
    """
    prefix: (W1, H1, W2, H2, C) 前缀和张量
    queries: (N, 8) 张量，每行为 [w1_1, h1_1, w2_1, h2_1, w1_2, h1_2, w2_2, h2_2]
    返回: (N, C) 每个区域的特征和
    """
    device = prefix.device
    N = queries.shape[0]
    C = prefix.shape[4]

    # 16 种 inclusion-exclusion 组合
    combos = torch.tensor(list(itertools.product([0, 1], repeat=4)), device=device)
    signs = (-1) ** combos.sum(dim=1).float()  # (16,)

    # 提取左、右边界
    lower = queries[:, :4] - 1  # (N, 4)
    upper = queries[:, 4:] - 1  # (N, 4)

    # 构造索引 (N, 16, 4)
    idx = torch.where(
        combos.unsqueeze(0) == 1,
        upper.unsqueeze(1),
        lower.unsqueeze(1)
    )  # (N, 16, 4)

    # Clamp 防越界
    shape = torch.tensor(prefix.shape[:4], device=device)
    #idx = torch.clamp(idx, min=0)
    # print(idx.min())
    # print(idx.max())
    for i in range(4):
        idx[..., i] = torch.min(idx[..., i], shape[i] - 1)
    mask = (idx <0).any(dim=-1).unsqueeze(-1)
    # 拆出维度索引
    w1, h1, w2, h2 = idx.unbind(-1)  # 每个都是 (N, 16)

    # 利用高级索引 (广播) 提取所有位置的值 -> (N, 16, C)
    vals = prefix[w1, h1, w2, h2]  # (N, 16, C)
    vals = torch.where(mask,0,vals)
    # 加权求和
    signs = signs.view(1, 16, 1)  # (1, 16, 1)
    result = (vals * signs).sum(dim=1)  # (N, C)
    return result


def ggx_ndf(cos_theta, alpha):
    alpha2 = alpha * alpha
    denom = (cos_theta ** 2 * (alpha2 - 1) + 1) ** 2
    return alpha2 / (np.pi * denom)


# 角度空间
def get_roughness_angular_table():
    roughness_to_angular_array = []
    for i in range(1000):
        alpha = (i / 1000) ** 2
        retain_ratio = 0.85
        theta = np.linspace(0, np.pi / 2, 2000)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        d = ggx_ndf(cos_theta, alpha)
        pdf = d * sin_theta  # 加权后用于积分

        # 累积积分（近似 CDF）
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]  # 归一化为 0~1

        # 找到第一个大于 99% 的位置
        cut_index = np.searchsorted(cdf, retain_ratio)
        theta_cut = theta[cut_index]

        # 截断后的 D
        mask = theta <= theta_cut
        d_trunc = d * mask
        s_trunc = np.trapz(d_trunc * sin_theta, theta)
        d_trunc /= s_trunc  # 再归一化，保持能量守恒
        print(theta_cut)
        roughness_to_angular_array.append(theta_cut)
    return torch.Tensor(roughness_to_angular_array)

def print_image_exr(img,name):
    if type(img)== torch.Tensor:
        img = numpy.array(img.detach().cpu())
    shape = img.shape
    #print(name,img.shape)
    if len(shape) == 4:
        img = img.reshape(shape[0]*shape[1],shape[2],shape[3])
    elif len(shape) == 6:
        img = img[0]
        shape = shape[1:]
    if len(shape) == 5:
        lenth = shape[0] * shape[2]
        img = img.transpose(0,2,1,3,4).reshape(lenth,lenth,-1)

    pyexr.write(name,img)


def cone_union_percentage_radians(theta1, theta2, phi):
    """
    计算两个圆锥在单位球内的交集体积角占第二个圆锥体积角的百分比。

    参数：
        theta1, theta2, phi: Tensor，形状为 (B, W, H, 1)，弧度制
    返回：
        percentage: Tensor，形状为 (B, W, H, 1)，并集体积角占第二个圆锥体积角的百分比
    """
    # 计算余弦
    cos_th1 = torch.cos(theta1)
    cos_th2 = torch.cos(theta2)
    cos_phi = torch.cos(phi)

    # 计算每个圆锥的体积角
    omega1 = 2 * torch.pi * (1 - cos_th1)
    omega2 = 2 * torch.pi * (1 - cos_th2)

    # 计算交集体积角（近似）
    omega_overlap = 2 * torch.pi * torch.clamp(cos_phi - cos_th1 - cos_th2 + 1, min=0.0)

    # 计算并集体积角


    # 计算并集体积角占第二个圆锥体积角的百分比
    percentage = omega2 / (omega_overlap + 1e-6)
    return percentage,omega2,omega_overlap



def collect_specialdata_from_path(job_name,label,file_name):
    path = r"../outputs/{}/{}/latest/data/".format(job_name,label)
    file_list = os.listdir(path)
    result_list = {}
    for id in file_list:
        file_path = path + id + "/" + file_name.format(id)
        if file_name.split(".")[-1] == "exr":
            result = pyexr.read(file_path)
        else:
            result = numpy.array(Image.open(file_path))
        result_list[id] = result
    return result_list