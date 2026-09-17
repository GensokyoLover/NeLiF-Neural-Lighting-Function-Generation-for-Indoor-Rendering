"""NeLiF lighting decoder and the components required by its forward paths."""

import math
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .loss_functions import Loss
from .shadow_network import NelifShadowNetwork


def fc_layer(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(inplace=True, negative_slope=0.2),
    )


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


def normalize(x):
    return x / torch.sqrt((x * x).sum(dim=-1)).unsqueeze(dim=-1)


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


def get_light_space_gbuffer(data,forward_indirect,forward_volume):
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
    return localGbuffer


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
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        # MLP block
        if moe_config == None:
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            print("lets create moe")
            # self.mlp = PositionwiseFeedforwardLayer(hidden_dim, moe_config)

    def forward(self, x: torch.Tensor, c=None):
        prex = x
        sa = self.ln_1(x)
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
        x3 = x3.reshape(x3.shape[0], -1)          # [B,512*6*1]

        return self.fc(x3)


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

    def train_step(self, rsm_input, gbuffer_input, gt, mask):

        B = gbuffer_input.shape[0]
        H = gbuffer_input.shape[1]
        W = gbuffer_input.shape[2]

        # ---------------------------------------------------------
        # encoder
        # ---------------------------------------------------------
        diffuse_decoder_input = self.compress_layer(rsm_input[..., :64])

        light_albedo  = rsm_input[..., 64:65]
        light_position = rsm_input[..., 65:68]
        light_normal   = rsm_input[..., 68:71]

        indirect_encoder_input = torch.cat([
            diffuse_decoder_input,
            light_albedo,
            light_position,
            light_normal,
        ], dim=-1)
        indirect_feature = self.indirect_encoder(indirect_encoder_input)

        C = indirect_feature.shape[-1]
        indirect_feature = (
            indirect_feature
            .view(B * 3, 1, 1, C)
            .expand(-1, H, W, -1)
        )
        D = gbuffer_input.shape[-1]

        gbuffer_expand = (
            gbuffer_input
            .unsqueeze(1)                  # [B,1,H,W,D]
            .expand(-1,3,-1,-1,-1)         # [B,3,H,W,D]
            .reshape(B*3,H,W,D)
        )
        indirect_decoder_input = torch.cat(
            [indirect_feature, gbuffer_expand],
            dim=-1,
        )

        indirect_result = self.indirect_decoder(indirect_decoder_input)
        # [B*3,H,W,2]

        # =========================================================
        # 恢复 batch
        # =========================================================

        indirect_result = indirect_result.view(
            B,
            3,
            H,
            W,
            2,
        )

        # ---------------------------------------------------------
        # diffuse
        # [B,H,W,3]
        # ---------------------------------------------------------
        indirect_diffuse_result = (
            indirect_result[...,0]
            .permute(0,2,3,1)
            .contiguous()
        )

        # ---------------------------------------------------------
        # specular
        # [B,H,W,3]
        # ---------------------------------------------------------
        indirect_specular_result = (
            indirect_result[...,1]
            .permute(0,2,3,1)
            .contiguous()
        )

        # ---------------------------------------------------------
        # mask
        # ---------------------------------------------------------
        indirect_diffuse_result = torch.where(
            mask[..., :3],
            gt[..., :3],
            indirect_diffuse_result,
        )

        indirect_specular_result = torch.where(
            mask[..., 3:6],
            gt[..., 3:6],
            indirect_specular_result,
        )

        # [B,H,W,6]
        return torch.cat(
            [
                indirect_diffuse_result,
                indirect_specular_result,
            ],
            dim=-1,
        )


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
    elif plane_res == 4:
        growth_rate = 3
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

    return torch.cat([uv_x,uv_y, d_val], dim=-1)


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
    feat_xy = F.grid_sample(padded_xy_plane, grid_xy_scaled, mode=mode, padding_mode='border', align_corners=False)

    # 对 xz 和 yz 依然使用 'border' 兜底
    feat_xz = F.grid_sample(xz_plane, grid_xz, mode=mode, padding_mode='border', align_corners=False)
    feat_yz = F.grid_sample(yz_plane, grid_yz, mode=mode, padding_mode='border', align_corners=False)

    sampled_features = feat_xy + feat_xz + feat_yz

    return sampled_features.permute(0,2,3,1), feat_xy.permute(0,2,3,1), feat_xz.permute(0,2,3,1), feat_yz.permute(0,2,3,1), grid_xy_scaled.permute(0,2,3,1), grid_xz.permute(0,2,3,1), grid_yz.permute(0,2,3,1)


def buffer_process(data,forward_indirect,forward_voluome=False):
    specular_reflect_dir = torch.sum(data["local"]["gbuffer"]["view_dir"] * data["local"]["gbuffer"]["normal"],
                                        dim=-1).unsqueeze(-1) * data["local"]["gbuffer"]["normal"] * 2 - \
                            data["local"]["gbuffer"]["view_dir"]
    data["local"]["gbuffer"]["specular_ray"] = specular_reflect_dir
    data["local"]["gbuffer"]["half_vec"] = normalize(
        (normalize(-data["local"]["gbuffer"]["lposition"]) + data["local"]["gbuffer"]["view_dir"]) / 2)
    data["local"]["gbuffer"]["dot"] = torch.sum(
        data["local"]["gbuffer"]["half_vec"] * data["local"]["gbuffer"]["normal"], dim=-1, keepdim=True)

    light_space_gbuffer = get_light_space_gbuffer(data,forward_indirect,forward_voluome)
    return light_space_gbuffer




class PlaneDecoder(nn.Module):
    def __init__(self, embedding_dim=1024, decoder_dim=64, plane_resolution=32):
        super().__init__()
        self.embedding_size = embedding_dim
        self.decoder_dim = decoder_dim
        self.plane_cnt = plane_resolution
        self.tri_plane_embed = nn.Parameter(
            0.01 * torch.randn(
                3,
                plane_resolution,
                plane_resolution,
                embedding_dim,
            )
        )
        self.transformer_decoder = CrossAttentionBlock(
            inner_dim=embedding_dim,
            cond_dim=embedding_dim,
            num_heads=16,
            eps=1e-6,
        )
        self.tri_output_layer = TriLinear(embedding_dim, decoder_dim)
        self.output_resolution = plane_resolution
        self.resize_mode = "query"

    def set_output_resolution(self, resolution: int, mode: str = "query"):
        """Set the triplane output resolution.

        ``mode='query'`` promotes ``tri_plane_embed`` itself to ``resolution``:
        the resized tensor becomes a new trainable Parameter and is saved in
        state_dict. Recreate the optimizer after this call.

        ``mode='output'`` retains the query Parameter and only interpolates
        decoded features for inference.
        """
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        if mode not in ("query", "output"):
            raise ValueError("mode must be 'query' or 'output'")
        if mode == "query":
            self.resize_query_parameter(resolution)
        else:
            self.output_resolution = resolution
            self.resize_mode = mode

    @torch.no_grad()
    def resize_query_parameter(self, resolution: int):
        """Replace the learned query grid with its bilinearly resized version.

        Call after loading a lower-resolution checkpoint and before creating
        the optimizer. For example, a 64x64 checkpoint can be promoted to
        128x128, then trained and saved as a native 128x128 checkpoint.
        """
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        if self.tri_plane_embed.shape[1:3] == (resolution, resolution):
            self.plane_cnt = resolution
            self.output_resolution = resolution
            self.resize_mode = "query"
            return

        resized_query = self._resize_feature_grid(self.tri_plane_embed, resolution)
        self.tri_plane_embed = nn.Parameter(
            resized_query.detach().contiguous(),
            requires_grad=self.tri_plane_embed.requires_grad,
        )
        self.plane_cnt = resolution
        self.output_resolution = resolution
        self.resize_mode = "query"

    @staticmethod
    def _resize_feature_grid(features, resolution):
        """Resize [..., H, W, C] grids independently with pixel-center alignment."""
        if features.shape[-3:-1] == (resolution, resolution):
            return features
        leading = features.shape[:-3]
        height, width, channels = features.shape[-3:]
        planes = features.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
        planes = F.interpolate(
            planes, size=(resolution, resolution), mode="bilinear", align_corners=False
        )
        return planes.permute(0, 2, 3, 1).reshape(
            *leading, resolution, resolution, channels
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Adapt a freshly constructed decoder to a saved query resolution."""
        query_key = prefix + "tri_plane_embed"
        saved_query = state_dict.get(query_key)
        if saved_query is not None:
            if (
                saved_query.ndim != 4
                or saved_query.shape[0] != 3
                or saved_query.shape[1] != saved_query.shape[2]
                or saved_query.shape[3] != self.embedding_size
            ):
                raise RuntimeError(
                    f"{query_key} must have shape [3, H, W, {self.embedding_size}], "
                    f"but has {tuple(saved_query.shape)}"
                )
            if self.tri_plane_embed.shape != saved_query.shape:
                self.tri_plane_embed = nn.Parameter(
                    torch.empty_like(saved_query),
                    requires_grad=self.tri_plane_embed.requires_grad,
                )
            self.plane_cnt = saved_query.shape[1]
            self.output_resolution = saved_query.shape[1]
            self.resize_mode = "query"
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward_triplane(self, photon_texture):
        batch = photon_texture.shape[0]
        channels = photon_texture.shape[-1]
        query_grid = self.tri_plane_embed
        resolution = query_grid.shape[1]
        query = query_grid.reshape(-1, self.embedding_size)
        query = query.unsqueeze(0).expand(batch, -1, -1)
        photon_texture = photon_texture.reshape(batch, -1, channels)
        triplane_feature = self.transformer_decoder(query, photon_texture)
        triplane_feature = triplane_feature.reshape(
            batch,
            3,
            resolution,
            resolution,
            self.embedding_size,
        )
        compressed = self.tri_output_layer(triplane_feature)
        if self.resize_mode == "output":
            triplane_feature = self._resize_feature_grid(triplane_feature, self.output_resolution)
            compressed = self._resize_feature_grid(compressed, self.output_resolution)
        return triplane_feature, compressed


class TriplaneOutputLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        """
        输入/输出：
            [B, 3, C, H, W]
            或 [B * 3, C, H, W]

        每个平面使用独立卷积分支：
            output = input + tail(head(input))

        tail 零初始化，保证初始化时 output == input。
        """
        super().__init__()



        self.plane_nets = nn.ModuleList([
            self._make_plane_net(in_channels, hidden_channels)
            for _ in range(3)
        ])

    @staticmethod
    def _make_plane_net(in_channels, hidden_channels):
        net = nn.Sequential(OrderedDict([
            ("head", nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )),
            ("tail", nn.Conv2d(
                hidden_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            )),
        ]))

        # 只将最后一层零初始化，head 保留默认初始化。
        nn.init.zeros_(net.tail.weight)
        nn.init.zeros_(net.tail.bias)

        return net

    def forward(self, x):
        flattened_input = x.ndim == 4

        if flattened_input:
            if x.shape[0] % 3 != 0:
                raise ValueError(
                    "Flattened triplane input must have "
                    "a batch dimension divisible by 3"
                )
            x = x.reshape(-1, 3, *x.shape[1:])
        elif x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(
                "Expected input shaped [B, 3, C, H, W] "
                "or [B * 3, C, H, W]"
            )

        residual = torch.stack(
            [
                plane_net(x[:, plane_index])
                for plane_index, plane_net in enumerate(self.plane_nets)
            ],
            dim=1,
        )

        output = x + residual

        return output.flatten(0, 1) if flattened_input else output

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """将旧共享分支的 head/tail 参数复制到三个独立分支。"""
        legacy_names = (
            "head.0.weight",
            "head.0.bias",
            "tail.weight",
            "tail.bias",
        )

        for legacy_name in legacy_names:
            legacy_key = prefix + legacy_name
            legacy_value = state_dict.get(legacy_key)

            if legacy_value is None:
                continue

            for plane_index in range(3):
                new_key = (
                    prefix
                    + f"plane_nets.{plane_index}."
                    + legacy_name
                )
                if new_key not in state_dict:
                    state_dict[new_key] = legacy_value.clone()

            state_dict.pop(legacy_key)

        super()._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )

class HeavyUpsampler(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, upsample=True):
        """
        Args:
            in_channels (int): 输入通道数 (C)
            hidden_channels (int): 隐藏层通道数
            upsample (bool): 是否进行上采样。
                             True -> 输出尺寸为 4H x 4W
                             False -> 输出尺寸为 H x W (原地卷积)
        """
        super(HeavyUpsampler, self).__init__()
        self.enable_upsample = upsample
        
        # 1. 头部：特征映射到高维空间
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1,padding_mode='replicate'),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1,padding_mode='replicate'),
            nn.ReLU(inplace=True)
        )
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
        self.output_dim = 3
        if self.channel_cut:
            self.light_input_dim = 7
            self.output_dim = 1
        self.decoder_light_feature = configs["plane"]["decoder_direct_dim"]
        self.plane_res = 128
        self.image_encoder = FourDTransformer(angular_size=8 , space_size=128 , angular_block_size=1, space_block_size=16,
                                                num_layers=3, num_heads=4, input_dim=7, hidden_dim=256, mlp_dim=1024,
                                                output_dim=256, linear_output=True)
        self.image_to_plane = PlaneDecoder(
            embedding_dim=256, decoder_dim=64,plane_resolution=self.plane_res
        )
        self.loss_func = Loss(loss_config) if loss_config else None
       
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

        self.shadow_network = NelifShadowNetwork(16 + 5)
        self.direct_compress_layer = nn.Linear(self.decoder_light_feature, 16)
        self.indirect_feature =64
        self.indirect_proxy = IndirectFarwardProxy(self.indirect_feature)
        self.trioutputlayer = TriplaneOutputLayer(64,128)
        self.plane_pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                3,
                self.decoder_light_feature,
                self.plane_res,
                self.plane_res,
            )
        )
        nn.init.normal_(self.plane_pos_embedding, std=0.02)

    @torch.no_grad()
    def resize_plane_pos_embedding(self, resolution: int):
        """Replace the positional embedding with an interpolated trainable one.

        Call after loading a checkpoint and before creating the optimizer. The
        target resolution must match image_to_plane's query resolution.
        """
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")

        embedding = self.plane_pos_embedding
        if embedding.shape[-2:] != (resolution, resolution):
            batch, planes, channels, height, width = embedding.shape
            resized = F.interpolate(
                embedding.reshape(batch * planes, channels, height, width),
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, planes, channels, resolution, resolution)
            self.plane_pos_embedding = nn.Parameter(
                resized.detach().contiguous(),
                requires_grad=embedding.requires_grad,
            )
        self.plane_res = resolution

    def set_plane_resolution(self, resolution: int):
        """Promote the learned query and positional embedding together."""
        self.image_to_plane.set_output_resolution(resolution, mode="query")
        self.resize_plane_pos_embedding(resolution)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Adapt a new 64-resolution model to saved positional embeddings."""
        embedding_key = prefix + "plane_pos_embedding"
        saved_embedding = state_dict.get(embedding_key)
        if saved_embedding is not None:
            if (
                saved_embedding.ndim != 5
                or saved_embedding.shape[0] != 1
                or saved_embedding.shape[1] != 3
                or saved_embedding.shape[2] != self.decoder_light_feature
                or saved_embedding.shape[-2] != saved_embedding.shape[-1]
            ):
                raise RuntimeError(
                    f"{embedding_key} must have shape [1, 3, {self.decoder_light_feature}, R, R], "
                    f"but has {tuple(saved_embedding.shape)}"
                )
            if self.plane_pos_embedding.shape != saved_embedding.shape:
                self.plane_pos_embedding = nn.Parameter(
                    torch.empty_like(saved_embedding),
                    requires_grad=self.plane_pos_embedding.requires_grad,
                )
            self.plane_res = saved_embedding.shape[-1]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward_light(self,data,gt_plane=None):


        lightData = data["global"]
        print(lightData.keys())
        B = lightData["radiance"].shape[0]

        radiance = lightData["radiance"].permute(
            0,5,1,2,3,4
        ).contiguous()

        radiance = radiance.reshape(
            B*3,
            *radiance.shape[2:],
            1
        )
        light_input = torch.cat(
            [radiance ,lightData["position"].repeat_interleave(3, dim=0),lightData["direction"].repeat_interleave(3, dim=0)], dim=-1)
        global_light_feature = self.image_encoder(light_input)
        _, raw_planes = self.image_to_plane.forward_triplane(global_light_feature)

        generated_plane = raw_planes.permute(0, 1, 4, 2, 3).contiguous()
        return generated_plane


    def forward_direct(self,decoder_type,data,result,light_feature,light_space_gbuffer,shading,mask,channel_cnt=3):
        gbuffer_input = torch.cat(
            [
                light_space_gbuffer["normal"],
                light_space_gbuffer["half_vec"],
                light_space_gbuffer["specular_ray"],
                data["local"]["gbuffer"]["roughness"],
                data["local"]["gbuffer"]["dot"],
                data["local"]["gbuffer"]["lenth"],
            ],
            dim=-1,
        ).repeat_interleave(3, dim=0)

        specular_input = torch.cat(
            [light_feature, gbuffer_input],
            dim=-1,
        )

        target_dtype = next(self.diffuse_decoder.parameters()).dtype
        specular_input = specular_input.to(target_dtype)

        if decoder_type == "diffuse":
            direct_shading = self.diffuse_decoder(specular_input)
        elif decoder_type == "specular":
            direct_shading = self.specular_decoder(specular_input)
        B = shading.shape[0]

        direct_shading = direct_shading.reshape(
            B,
            3,
            direct_shading.shape[1],
            direct_shading.shape[2],
            1,
        )

        direct_shading = (
            direct_shading
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .squeeze(-1)
        )

        direct_shading = torch.where(
            mask,
            shading,
            direct_shading,
        )

        return direct_shading

    def forward_shadow(self,data,result,light_feature,light_space_gbuffer):
        localData = data["local"]
        data["local"]["shadow_light_repr"] = self.direct_compress_layer(light_feature)
        data["local"]["shadow_input"] = get_lightformer_input(data["local"]["lights"], data["local"],False).to(data["local"]["shadow_light_repr"].dtype)
        shadow_result = self.shadow_network.step(data)
        shadow_result = shadow_result.clamp(0.0, 1.0)

        if torch.any(torch.isnan(shadow_result)):
            print("nan")
            exit()
        shadow_result = torch.where(
            data["local"]["mask"] | data["local"]["shadow_mask"],
            localData["shadow"],
            shadow_result)

        return shadow_result

    def forward(self, data, need_diffuse, need_specular, need_shadow, need_indirect, need_volume, channel_cnt):
        timers = {}   # <---- 用于存储每个阶段的耗时(ms)

        light_space_gbuffer = buffer_process(data, False,need_volume)
        for key in light_space_gbuffer:
            data["local"][key] = light_space_gbuffer[key]
        self.plane_res = self.image_to_plane.output_resolution
        B = data["global"]["radiance"].shape[0]
        sampled_plane = self.forward_light(data)
        voxel_coord = xyz_to_uvd_pixel_center(data["local"]["gbuffer"]["lposition"],self.plane_res)
        voxel_coord[...,:2] = voxel_coord[...,:2] * 2 - 1
        voxel_coord[...,2:3] = voxel_coord[...,2:3] /self.plane_res * 2 - 1
        voxel_coord = voxel_coord.repeat_interleave(3, dim=0)

        sampled_plane = sampled_plane + self.plane_pos_embedding
        sampled_plane= self.trioutputlayer(sampled_plane)

        light_feature,_,_,_,_,_,_  = sample_from_triplane_oct(sampled_plane,voxel_coord[...,[0,1,2]],mode='bilinear')

        BL,WL,HL,CL = light_feature.shape
        data["local"]["voxel_coord"] = ((voxel_coord* 0.5 + 0.5) * self.plane_res)

        data["local"]["planeuv_visualize"] = sampled_plane[:,0,...].permute(0,2,3,1)[...,:3]
        data["local"]["planeud_visualize"] = sampled_plane[:,1,...].permute(0,2,3,1)[...,:3]
        data["local"]["planevd_visualize"] = sampled_plane[:,2,...].permute(0,2,3,1)[...,:3]

        if torch.any(light_feature.isnan()):
            print("light_feature nan")
            exit()
        data["local"]["sampled_principle_value"] = light_feature[..., :3]
        data["local"]["direct_light_reprs"] = light_feature


        result = {}


        if need_indirect:
            indirect_data = data["local"]["lights"]["shadow"]
            invoxel_coord = xyz_to_uvd_pixel_center(indirect_data["light_position"],self.plane_res)
            invoxel_coord[...,:2] = invoxel_coord[...,:2] * 2 - 1
            invoxel_coord[...,2:3] = invoxel_coord[...,2:3] /self.plane_res * 2 - 1
            invoxel_coord = invoxel_coord.repeat_interleave(3,0)
            indirect_feature,_,_,_,_,_,_ = sample_from_triplane_oct(sampled_plane,invoxel_coord[...,[0,1,2]],mode='bilinear')
            print("indirect_feature",indirect_feature.shape)

            lp = indirect_data["light_position"]
            sp = data["local"]["gbuffer"]["lposition"]

            print("light albedo shape",indirect_data["light_albedo"].shape)
            print(" indirect_feature shape",indirect_feature.shape)
            light_albedo = (
                indirect_data["light_albedo"]
                .permute(0, 3, 1, 2)      # [B,3,H,W]
                .reshape(B * 3, 384, 64, 1)  # [B*3,H,W,1]
            )
            rsm_input = torch.cat([indirect_feature,light_albedo,lp.repeat_interleave(3,0),indirect_data["light_normal"].repeat_interleave(3,0)],dim=-1)
            data["local"]["rsm_input"] = rsm_input

            screen_input = torch.cat([data["local"]["gbuffer"]["normal"],
                data["local"]["gbuffer"]["view_dir"],
                data["local"]["gbuffer"]["roughness"],
                sp],dim=-1)
            data["local"]["screen_input"] = screen_input
            indirect_result = self.indirect_proxy.train_step(
                rsm_input,screen_input,torch.cat([data["local"]["log1p_diffuse_indirect_shading"],
                        data["local"]["log1p_specular_indirect_shading"]], dim=-1),
                torch.cat([data["local"]["albedo_mask"],
                        data["local"]["specular_mask"]], dim=-1)
            )


            result["log1p_diffuse_indirect_shading"] = indirect_result[...,:3]
            result["log1p_specular_indirect_shading"] = indirect_result[...,3:6]
        if need_diffuse:
            pred_diffuse = self.forward_direct(
                "diffuse", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_diffuse_direct_shading"],
                data["local"]["mask"] | data["local"]["albedo_mask"], channel_cnt
            )
            result["log1p_diffuse_direct_shading"] = pred_diffuse

        if need_specular:
            pred_specular = self.forward_direct(
                "specular", data, result, light_feature, light_space_gbuffer,
                data["local"]["log1p_specular_direct_shading"],
                data["local"]["mask"] | data["local"]["specular_mask"], channel_cnt
            )
            result["log1p_specular_direct_shading"] = pred_specular
        if need_shadow:
            result["shadow"] = self.forward_shadow(
                data, result, light_feature, light_space_gbuffer
            )


        if self.loss_func is not None:
            loss_map = self.loss_func(result, data)
        else:
            loss_map = None

        return data, result, loss_map, timers
