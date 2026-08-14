# This is a sample Python script.
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.partition_pyramid import PartitioningPyramid, NoSoftMaxPartitioningPyramid,TemporalPartitioningPyramid
from torchvision.models import vgg16
from torchvision.transforms import Normalize
import lpips
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def reversedl(it):
    return list(reversed(list(it)))



class ConvUNet(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (36, 36),
                     (48, 48),
                     (64, 64),
                     (76, 76),
                     (96, 96),
                     (128, 128, 96)
                 ],
                 pool=F.max_pool2d,
                 specular_feature=32):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        cnt = 0

        for dims in dims_and_depths:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))
        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x, ):
        skips = []
        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)
        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear"), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                # print("x shape :",x.shape)
                outputs.append(self.out_path[i](x))
                # print("outputs shape :",outputs[-1].shape)

        return reversedl(outputs)


class MidConvUNet(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (20, 20),
                     (36, 36),
                     (54, 54),
                     (76, 76),
                     (96, 96),
                     (128, 128, 96)
                 ],
                 pool=F.max_pool2d,
                 specular_feature=32):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        cnt = 0

        for dims in dims_and_depths:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))
        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x ):
        skips = []
        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)
        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear"), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                # print("x shape :",x.shape)
                outputs.append(self.out_path[i](x))
                # print("outputs shape :",outputs[-1].shape)

        return reversedl(outputs)
    

class ResBlock(nn.Module):
    """带 GroupNorm 的残差块，增强梯度稳定性"""
    def __init__(self, in_dim, out_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, padding='same')
        self.gn1 = nn.GroupNorm(groups, out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, padding='same')
        self.gn2 = nn.GroupNorm(groups, out_dim)
        
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1),
                nn.GroupNorm(groups, out_dim)
            )

    def forward(self, x):
        out = F.leaky_relu(self.gn1(self.conv1(x)), 0.3)
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.3)

class MidConvUNetLight(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (24, 24), (40, 40), (56, 56), (80, 80), (96, 96), (128, 128)
                 ],
                 groups=8):
        super().__init__()

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        
        # --- Encoder (每个 Stage 仅 1 个 ResBlock) ---
        for dims in dims_and_depths:
            # 取 dims[-1] 保证通道数对齐
            self.ds_path.append(ResBlock(prev_dim, dims[-1], groups=groups))
            prev_dim = dims[-1]

        # --- Decoder (每个 Stage 仅 1 个 ResBlock) ---
        self.us_path = nn.ModuleList()
        encoder_dims = [d[-1] for d in dims_and_depths]
        prev_dim = encoder_dims[-1] 

        for i in range(len(encoder_dims) - 2, -1, -1):
            skip_dim = encoder_dims[i]
            out_dim = dims_and_depths[i][-1]
            in_f = prev_dim + skip_dim 
            
            # 这里精简为单个 ResBlock
            self.us_path.append(ResBlock(in_f, out_dim, groups=groups))
            prev_dim = out_dim

        # --- Output Heads ---
        self.out_path = nn.ModuleList()
        for i in range(len(out_chans)):
            target_dim = dims_and_depths[i][0]
            self.out_path.append(nn.Conv2d(target_dim, out_chans[i], 1))
        self.pool = F.max_pool2d
    def forward(self, x):
        skips = []
        print("light")
        # Downsample
        for i, stage in enumerate(self.ds_path):
            x = stage(x)
            skips.append(x)
            if i < len(self.ds_path) - 1:
                x = self.pool(x, 2)
        
        final_outputs = [None] * len(self.out_path)
        
        # # 1. Bottleneck 预测
        # final_outputs[-1] = self.out_path[-1](skips[-1])
        
        # 2. Upsample 路径
        curr_x = skips[-1]
        for i, stage in enumerate(self.us_path):
            skip_idx = len(skips) - 2 - i
            skip = skips[skip_idx]
            
            curr_x = F.interpolate(curr_x, scale_factor=2, mode="bilinear", align_corners=False)
            curr_x = torch.cat((curr_x, skip), 1)
            curr_x = stage(curr_x)

            final_outputs[skip_idx] = self.out_path[skip_idx](curr_x)

        return final_outputs
class TemporalConvUNet(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (36, 36),
                     (54, 54),
                     (76, 76),
                     (96, 96),
                     (128, 128),
                     (156, 156, 128)
                 ],
                 pool=F.max_pool2d,
                 specular_feature=32):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        cnt = 0

        for dims in dims_and_depths:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))
        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x, ):
        skips = []
        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)
        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear"), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                # print("x shape :",x.shape)
                outputs.append(self.out_path[i](x))
                # print("outputs shape :",outputs[-1].shape)

        return reversedl(outputs)

class ConvUNetForLight(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (36, 36),
                     (48, 48),
                     (64, 64),
                     (76, 76),
                     (96, 96),
                     (128, 128, 96)
                 ],
                 pool=F.max_pool2d,
                 specular_feature=32):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        cnt = 0

        for dims in dims_and_depths:
            layers = []
            gg = 0
            for dim in dims:
                if cnt < 4 and cnt > 0 and gg == 0:
                    layers.append(nn.Conv2d(prev_dim + 32, dim, 3, padding='same'))
                else:
                    layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim
                gg = gg + 1
            cnt = cnt + 1
            self.ds_path.append(nn.Sequential(*layers))
        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x, light_input, training_level=None):
        skips = []
        cnt = 0
        for stage in self.ds_path:
            if cnt < 4 and cnt > 0:
                x = torch.cat([x, F.avg_pool2d(light_input[cnt:cnt + 1, ...].permute(0, 3, 1, 2), 2 ** cnt)], dim=1)
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)
            cnt = cnt + 1
        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear"), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                # print("x shape :",x.shape)
                outputs.append(self.out_path[i](x))
                # print("outputs shape :",outputs[-1].shape)

        return reversedl(outputs)


class SmallConvUNet(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (12, 12),
                     (16, 16),
                     (24, 24),
                     (36, 36),
                     (54, 54),
                     (72, 72, 54)
                 ],
                 pool=F.max_pool2d):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans

        for dims in dims_and_depths:
            layers = []

            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x):
        skips = []

        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)

        x = skips[-1]
        outputs = []

        if len(self.out_path) == len(self.ds_path):
            # print("x shape :",x.shape)
            outputs.append(self.out_path[-1](x))
            # print("outputs shape :",outputs[-1].shape)

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                # print("x shape :",x.shape)
                outputs.append(self.out_path[i](x))
                # print("outputs shape :",outputs[-1].shape)

        return reversedl(outputs)
def warp_image_torch(img_src, uv, align_corners=False):
    """
    img_src : (C,H,W) float32 torch  (注意：C 放前面！)
    uv      : (H,W,2) torch (u,v)  像素坐标
    返回     : (C,H,W)
    """
    img_src = img_src.permute(0,3,1,2)
    B,C, H, W = img_src.shape

    # 1) 像素坐标 → [-1,1] 归一化，grid_sample 期望 (x,y)
    x = (uv[...,0] / W) * 2 - 1
    y = (uv[...,1] / H) * 2 - 1
    grid = torch.stack([x, y], dim=-1)              # (H,W,2)


    warped = F.grid_sample(img_src, grid,
                           mode='bilinear',
                           padding_mode='zeros',
                           align_corners=align_corners)
    return warped.permute(0,2,3,1)



class ShadowLPIPSLoss(nn.Module):
    def __init__(self, net='alex', spatial_mode=False):
        """
        net: 'alex', 'vgg', or 'squeeze'
        spatial_mode: True 时返回每个像素 patch 的 LPIPS map，否则返回 scalar loss
        """
        super().__init__()
        self.lpips = lpips.LPIPS(net=net).eval().requires_grad_(False)
        self.spatial = spatial_mode

    def train(self, mode: bool = True):
        # 强制始终保持 eval 模式
        super().train(False)
        self.lpips.eval()
        return self
    
    def preprocess(self, x):
        # x: [B, 1, H, W] in [0, 1]
        x = x.expand(-1, 3, -1, -1)  # 灰度 → RGB
        x = (x * 2) - 1              # [0, 1] → [-1, 1] (LPIPS 需要)
        x = torch.clamp(x,-1,1)
        return x
    def spatial_forward(self, pred, gt):
        """
        LPIPS 空间损失：当前帧预测与 GT 的结构差异
        """
        pred = self.preprocess(pred)
        gt   = self.preprocess(gt)
        loss = self.lpips(pred, gt)
        return loss.mean() if not self.spatial else loss  # spatial 模式返回 LPIPS map

    def temporal_forward(self, pred_t, pred_tm1, gt_t, gt_tm1):
        """
        LPIPS 时间损失：预测帧间变化 ≈ GT 帧间变化
        """
        pred_t = self.preprocess(pred_t)
        pred_tm1 = self.preprocess(pred_tm1)
        gt_t = self.preprocess(gt_t)
        gt_tm1 = self.preprocess(gt_tm1)

        # 差分（帧间变化）
        pred_diff = pred_t - pred_tm1
        gt_diff = gt_t - gt_tm1

        loss = self.lpips(pred_diff, gt_diff)
        return loss


class KernelShadowNetwork(nn.Module):
    def __init__(self, input_dim=37):
        super().__init__()

        self.filter = PartitioningPyramid()

        self.weight_predictor = ConvUNet(
            input_dim,
            self.filter.inputs
        )
        self.loss = ShadowLPIPSLoss()
        # self.features = Features(transfer='pu') # Broken with pytorch

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.94)
        return [opt], [sched]

    def step(self, x, color, light_reprs):
        # Reprojection
        prev_color = color

        weight_predictor_input = torch.cat([x, light_reprs,color], dim=1)
        #weight_predictor_input = torch.cat([x, light_reprs], dim=1)
        prev_output = color
        batch_size = weight_predictor_input.shape[0]


        weights = self.weight_predictor(weight_predictor_input)
        # weights = [weight.to(torch.float32) for weight in weights]

        # print("weight predict time: ",starter.elapsed_time(ender))


        output,shadow_list = self.filter(weights, color)
        

        # print("filter time: ",starter.elapsed_time(ender))
        part_weights = F.softmax(weights[0][:, 25:30], 1)
        final_weights = torch.cat([part_weights], dim=1).permute(0, 2, 3, 1)
        # print(final_weights.shape)
        # exit()
        return output.permute(1, 2, 3, 0), final_weights,shadow_list


import pyexr

class NelifShadowNetwork(nn.Module):
    def __init__(self,input_dim,dims_and_depths=[
                    (20, 20),
                    (36, 36),
                    (54, 54),
                    (76, 76),
                    (96, 96),
                    (128, 128, 96)
                ],group_num=8):

        super().__init__()
        self.filter = PartitioningPyramid()

        self.weight_predictor = MidConvUNet(
            input_dim,
            self.filter.inputs,
            dims_and_depths=dims_and_depths

        )
        self.final_activate = nn.LeakyReLU(0.3)
        # self.features = Features(transfer='pu') # Broken with pytorch

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.94)
        return [opt], [sched]

        prev_color = color

    def step(self, data):
    
        B = data["local"]["shadow_input"].shape[0]

        shadow_input = data["local"]["shadow_input"].repeat_interleave(
            3,
            dim=0,
        )

        x = torch.cat(
            [
                data["local"]["shadow_light_repr"],
                shadow_input,
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)

  
        shadow = (
            data["local"]["hard_shadow"]
            .permute(0, 3, 1, 2)
            .repeat_interleave(3, dim=0)
        )

        weights = self.weight_predictor(x)

        output = self.filter(weights, shadow)

        B3, _, H, W = output.shape

        assert B3 == B * 3

        output = output.permute(0, 2, 3, 1)
        # (B*3,H,W,1)

        output = output.reshape(
            B,
            3,
            H,
            W,
            1,
        )

        output = (
            output.permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(B, H, W, 3)
        )


        return output       

    def forward(self,input):
        x = input.permute(0,3,1,2)
        shadow =x[:,-1:,...]
        weights = self.weight_predictor(x)
        output,shadow_list = self.filter(weights, shadow)
        return output.permute(0, 2, 3, 1)

class SmallKernelShadowNetwork(nn.Module):
    def __init__(self,input_dim,dims_and_depths=[
                    (20, 20),
                    (36, 36),
                    (54, 54),
                    (76, 76),
                    (96, 96),
                    (128, 128, 96)
                ],group_num=8):

        super().__init__()
        self.filter = PartitioningPyramid()

        self.weight_predictor = MidConvUNetLight(
            input_dim,
            self.filter.inputs,
            dims_and_depths=dims_and_depths

        )
        self.final_activate = nn.LeakyReLU(0.3)
        # self.features = Features(transfer='pu') # Broken with pytorch

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.94)
        return [opt], [sched]

        prev_color = color

    def step(self, data):
        
        x = torch.cat([data["local"]["shadow_light_repr"],data["local"]["shadow_input"][...]], dim=-1).permute(0,3,1,2)
        shadow =data["local"]["hard_shadow"].permute(0,3,1,2)
        # print(shadow.shape)
        # pyexr.write("shadow_mask.exr",shadow.permute(0,2,3,1)[0].cpu().numpy())
        weights = self.weight_predictor(x)
        for i in range(len(weights)):
            print(i,weights[i].shape)
        output,shadow_list = self.filter(weights, shadow)
        
        three_weight = weights[0][:, -1, ...].unsqueeze(dim=1)
        part_weights = F.softmax(weights[0][:, 25:30], 1)
        light_weights = self.final_activate(three_weight)
        final_weights = torch.cat([part_weights, light_weights], dim=1).permute(0, 2, 3, 1)
        return output.permute(0, 2, 3, 1), final_weights,shadow_list,0
    
    def forward(self,input):
        x = input.permute(0,3,1,2)
        shadow =x[:,-1:,...]
        weights = self.weight_predictor(x)
        #output,shadow_list = self.filter(weights, shadow)
        #return output.permute(0, 2, 3, 1)
        return weights

class TemporalShadowNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(21, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )
    
        self.filter = TemporalPartitioningPyramid()


        self.weight_predictor = TemporalConvUNet(
            66,
            self.filter.inputs
        )
        self.loss = ShadowLPIPSLoss()
    
    def step(self, x,shadow,temporal,grid):

        # Reprojection
        
        reprojected = warp_image_torch(
            temporal,
            grid,
        )

        prev_shadow = reprojected[:, :1]
        prev_output = reprojected[:, 1:2]
        prev_feature = reprojected[:, 2:]
    
        # Sample encoder

        batch_size = x['color'].shape[0]

        encoder_input = torch.cat([x,shadow],dim=1)

        feature = self.encoder(encoder_input)

        feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)

        weight_predictor_input = torch.concat((
            torch.concat((
                prev_shadow,
                shadow
            ), 1),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        #weights = [weight.to(torch.float32) for weight in weights]

        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        shadow = t_lambda * prev_shadow + (1 - t_lambda) * shadow
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, shadow, prev_output)

        return output, torch.concat((
            shadow,
            output,
            feature
        ), 1), grid

    def one_step_loss(self, ref, pred):
        ref, mean = normalize_radiance(ref, True)
        pred = pred / mean

        spatial = SMAPE(ref, pred) * 0.8 * 10 + \
            self.features.spatial_loss(ref, pred) * 0.5 * 0.2
        
        return spatial, pred
    
    def two_step_loss(self, refs, preds, grid):
        refs, mean = normalize_radiance(refs, True)
        preds = preds / mean

        prev_ref = F.grid_sample(
            refs[:, 0],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_pred = F.grid_sample(
            preds[:, 0],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        spatial = SMAPE(refs.flatten(0, 1), preds.flatten(0, 1)) * 2.0 + \
            self.features.spatial_loss(refs.flatten(0, 1), preds.flatten(0, 1)) * 0.025

        diff_ref = refs[:, 1] - prev_ref
        diff_pred = preds[:, 1] - prev_pred

        temporal = SMAPE(diff_ref, diff_pred) * 0.2 + \
            self.features.temporal_loss(refs[:, 1], preds[:, 1], prev_ref, prev_pred) * 0.025

        return spatial + temporal, preds[:, 1]
    
    def temporal_init(self, x):
        
        return torch.zeros(1,512,512,2+32)

    def bptt_step(self, x):
        if x['frame_index'] == 0:
            temporal = self.temporal_init(x)
            y, _, _ = self.step(x, temporal)
            loss, y = self.one_step_loss(x['reference'], y)
        else:
            y_1, temporal, _    = self.step(self.prev_x, self.temporal)
            y_2, _       , grid = self.step(x, temporal)
            loss, y = self.two_step_loss(
                torch.stack((self.prev_x['reference'], x['reference']), 1),
                torch.stack((y_1, y_2), 1),
                grid
            )
        
        self.prev_x = x
        self.temporal = temporal.detach()
        return loss, y

    def training_step(self, x, batch_idx):
        loss, y = self.bptt_step(x)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, x, batch_idx):
        loss, y = self.bptt_step(x)
        
        if x['frame_index'] == 63:
            y = normalize_radiance(y)
            y = torch.pow(y / (y + 1), 1/2.2)
            y = dist_cat(y).cpu().numpy()

            # Writing images can block the rank 0 process for a couple seconds
            # which often breaks distributed training so we start a separate thread
            Thread(target=self.save_images, args=(y, batch_idx, self.trainer.current_epoch)).start()
            
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, x):
        y, temporal, _ = self.step(x, self.temporal)
        self.temporal = temporal.detach()
        return y


    def save_images(self, images, batch_idx, epoch):
        for i, image in enumerate(images):
            self.logger.experiment.add_image(f'denoised/{batch_idx}-{i}', image, epoch)



def benchmark_shadow_network():
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        raise RuntimeError("必须使用 CUDA 才能进行准确的静态图测速。")

    # 开启 cuDNN benchmark 以获得最优的卷积算法
    torch.backends.cudnn.benchmark = True

    # 2. 初始化模型并转换为半精度 (FP16)
    # 输入张量维度为 21 (根据 input 1, 512, 512, 21 推断)
    input_dim = 37 
    model = SmallKernelShadowNetwork(32 + 5,[(32,32),(64,64),(96,96),(144,144),(216,216),(288,216)]).to(device).half()
    model.eval()

    # 3. 创建 Dummy Input，形状要求为 (1, 512, 512, 21)，并转为半精度
    dummy_input = torch.randn(1, 37,512, 512, device=device, dtype=torch.float16)

    print("开始预热 (Warm-up)...")
    with torch.inference_mode():
        # 普通预热：让 GPU 和 cuDNN 找到最优卷积算法
        for _ in range(10):
            _ = model(dummy_input)

        print("捕获 CUDA Graph (生成静态图)...")
        # 4. 构建静态图 (CUDA Graph)
        g = torch.cuda.CUDAGraph()

        # CUDA Graph 要求在单独的 stream 中进行捕获预热
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(100):
                _ = model(dummy_input)
        torch.cuda.current_stream().wait_stream(s)

        # 正式捕获静态图
        with torch.cuda.graph(g):
            # 这次前向传播的指令会被录制下来，以后直接回放
            static_out = model(dummy_input)

        # 5. 正式性能测试
        print("开始精准测速...")
        iterations = 1000
        
        # 使用 CUDA Event 进行精准的高精度计时，排除 CPU/GPU 异步带来的误差
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # 确保之前的所有操作都已完成
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iterations):
            # 静态图回放 (Replay)，无 CPU 开销
            g.replay()
        end_event.record()

        # 等待测速完成
        torch.cuda.synchronize()

        # 6. 计算结果
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iterations
        fps = 1000.0 / avg_time_ms

        print("\n" + "="*30)
        print("=== 测速结果 ===")
        print("="*30)
        print(f"分辨率: 1x512x512x21")
        print(f"数据精度: FP16 (Half Precision)")
        print(f"执行模式: CUDA Graph (静态图回放)")
        print(f"测试次数: {iterations} 次")
        print(f"平均耗时: {avg_time_ms:.4f} ms")
        print(f"等效 FPS: {fps:.2f} 帧/秒")
        print("="*30)

if __name__ == "__main__":
    benchmark_shadow_network()