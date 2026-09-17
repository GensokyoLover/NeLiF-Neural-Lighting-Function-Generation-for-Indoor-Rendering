from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import VGG16_Weights
import torchvision
import pyexr
import numpy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np

from . import pytorch_ssim

import lpips
class LossFunction():
    @abstractmethod
    def forward(self, pred, gt, **kwards):
        pass

    def __call__(self, pred, gt, **kwards):
        return self.forward(pred, gt, **kwards)

def HDR2LDR_torch(img_tensor):
    """
    【注意】：这里需要替换为你原本 HDR2LDR 的真实逻辑。
    下面提供的是最常见的标准算法（Clip 截断 + Gamma 2.2 校正）的 PyTorch 实现：
    """
    # 限制在 0~1 之间
    img_tensor = torch.clamp(img_tensor, min=0.0, max=1.0)
    # Gamma 校正 (加一个 1e-8 防止底数为 0 导致 NaN)
    img_ldr = torch.pow(img_tensor + 1e-8, 1.0 / 2.2) 
    return img_ldr

def calculate_psnr_ldr_torch(gt, pred):
    """
    完全在 GPU 上运行的极速 PSNR 计算
    """
    # 1. 取出 batch 中的第一张图 (跳过耗时的 permute)
    # 原先形状 [B, C, H, W] -> 变成 [C, H, W]
    pred_first = pred[0]
    gt_first = gt[0]

    # 2. 在 GPU 上直接执行 HDR 到 LDR 的映射
    pred_ldr = HDR2LDR_torch(pred_first)
    gt_ldr = HDR2LDR_torch(gt_first)

    # 3. 计算 MSE (均方误差)
    mse = torch.mean((gt_ldr - pred_ldr) ** 2)

    # 如果两张图完全一样，避免除以 0 的报错
    if mse < 1e-10:
        return torch.tensor(float('inf'), device=pred.device)

    # 4. 完全复刻你原代码的 data_range 逻辑
    data_range = pred_ldr.max() - pred_ldr.min()

    # 5. 计算 PSNR
    psnr_val = 10.0 * torch.log10((data_range ** 2) / mse)
    
    return psnr_val
class L1MaskLoss(LossFunction):
    def forward(self, pred, gt, mask=None):
        if mask is not None:
            l = [
                F.l1_loss(torch.masked_select(pred[i], mask[i]), torch.masked_select(gt[i], mask[i])) 
                if mask[i].sum() > 0 else 
                torch.tensor(0.0, dtype=pred[i].dtype).cuda() 
                for i in range(pred.shape[0])
                ]
            l = torch.stack(l, dim=0)
        else:
            l = F.l1_loss(pred, gt, reduction='none')
            l = l.view(l.shape[0], -1).mean(dim=-1)
        return l

class RelativeLoss(LossFunction):
    def forward(self, pred, gt):
        # return F.l1_loss(pred, gt, reduction='none').mean(-1).mean(-1).mean(-1)
        return torch.abs(pred-gt) / (torch.abs(gt) + 1e-4)


class L1Loss(LossFunction):
    def forward(self, pred, gt):
        # return F.l1_loss(pred, gt, reduction='none').mean(-1).mean(-1).mean(-1)
        return torch.abs(pred-gt)
    
class L2Loss(LossFunction):
    def forward(self, pred, gt):
        # return F.l1_loss(pred, gt, reduction='none').mean(-1).mean(-1).mean(-1)
        return (pred-gt) * (pred-gt)


class FFTLoss(LossFunction):
    def forward(self, pred, gt):
        l = F.l1_loss(torch.fft.fft2(pred), torch.fft.fft2(gt), reduction='none')
        l = l.view(l.shape[0], -1).mean(dim=-1)
        return l

class SSIMLoss(LossFunction):
    def forward(self, pred, gt):
        # return 1-pytorch_ssim.ssim(pred, gt, size_average=False).mean(-1).mean(-1).mean(-1)
        # return 1.0 - pytorch_ssim.ssim(pred, gt, size_average=False) # It seems like ssim already means along the dimensions of rgb images
        # return 1 - pytorch_ssim.ssim(pred, gt, size_average=False) # It seems like ssim already means along the dimensions of rgb images
        # return (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # It seems like ssim already means along the dimensions of rgb images
        # print('Ours ssim:', 1.0 - pytorch_ssim.ssim(pred, gt), pred.shape)
        return 1.0 - pytorch_ssim.ssim(pred, gt) # It seems like ssim already means along the dimensions of rgb images

from utils.image_utils import HDR2LDR
class LpipsTrainLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #self.loss_network = lpips.LPIPS(net='vgg')
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, pred, gt):
        # print(pred.shape, gt.shape, max(pred), max(gt))
        # tone-mapping to LDR & normalize to [-1, 1] for Lpips
        # pred = torch.clamp(pred ** (1/2.2), min=0, max=1) * 2 - 1
        # gt = torch.clamp(gt ** (1/2.2), min=0, max=1) * 2 - 1
        # with torch.no_grad():
        #self.loss_map = self.loss_network(pred, gt, normalize=True)

        return {}

class Loss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
        #self.lpips_loss = LpipsTrainLoss() # Compatible with DataParallel
        self.loss_functions = {
            'L1': L1Loss(),
            'L2':  nn.MSELoss(),
            'SSIM': SSIMLoss(),
        }
        
        self.configs = configs
       
    def forward(self, preds, data):
        data = data["local"]
        loss_map = {
            'final_loss': 0.0
        }
        albedo = data["gbuffer"]["albedo"]
        specular = data["gbuffer"]["specular"]
        for l_name in self.configs['losses']:
            cfg = self.configs['losses'][l_name]
            if cfg['loss'] in self.loss_functions:
                loss_func = self.loss_functions[cfg['loss']]
            else:
                raise ValueError('Loss {} is not supported!'.format(cfg['loss']))
            if cfg['pname'] == "clip_radiance" or cfg["pname"] == "radiance":
                pred = preds[cfg['pname']]
                gt = data[cfg['gname']]
                loss = torch.abs(pred-gt)
                loss_map[l_name] = loss
                loss_map['final_loss'] += cfg['weight'] * loss.mean()
                continue
            pred = preds[cfg['pname']].permute(0, 3, 1, 2)
            gt = data[cfg['gname']].permute(0, 3, 1, 2)
            if cfg['mask'] is not None:
                mask = data[cfg['mask']].permute(0, 3, 1, 2)
                pred = pred * mask
                gt = gt * mask
            # print(pred.dtype)
            # print(gt.dtype)
            if cfg['loss'] == 'L1Mask':
                loss = loss_func(pred, gt, mask=mask.bool())
            else:
                loss = loss_func(pred, gt)
            

            # if "diffuse" in cfg['pname']:
            #     #print("loss shape",loss.shape,loss,cfg['pname'])
            #     loss *= albedo.permute(0,3,1,2)
            #     #print(cfg,cfg['pname'],"albedo")
            # if "specular" in cfg['pname']:
            #     loss *= specular.permute(0,3,1,2)
            #     #print(cfg,cfg['pname'],"specular")
            
            if cfg['reweigh']:
                # loss = loss * data['I_scale'].mean(dim=-1)[:, 0, 0]
                liangdu = data["log1p_diffuse_direct_shading"].permute(0,3,1,2) 
                liangdu = liangdu + data["log1p_specular_direct_shading"].permute(0,3,1,2) 
                loss *= liangdu * 0.3

            if "special_weight" in cfg.keys():
                pixel_wise_weight = data["log1p_direct_shading"]
                pixel_wise_weight = pixel_wise_weight.permute(0,3,1,2)
                loss = loss * pixel_wise_weight
            
            if cfg["relative"]:
                liangdu = (data["log1p_diffuse_direct_shading"].permute(0,3,1,2)  + data["log1p_specular_direct_shading"].permute(0,3,1,2)).mean(dim=(1,2,3), keepdim=True) + 1e-3
                # print("liangdu ",l_name,liangdu)
                # print(liangdu.shape)
            #loss = loss * data["relative"].permute(0,3,1,2)
            #print("{} loss  ".format(l_name),loss.mean())
            if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                print('NaN/Inf in {} Loss '.format(l_name))

            loss_map['final_loss'] += cfg['weight'] * loss.mean()
            # loss_map['final_loss'] += cfg['weight'] * loss
            if cfg['visualize']:
                loss_map[l_name] = loss
        # print('loss_map_shape', loss_map['final_loss'].shape, loss_map['final_loss'].mean())
        return loss_map
    
    @property
    def data_format(self):
        data_channels = {}
        for l_name in self.configs['losses']:
            cfg = self.configs['losses'][l_name]
            data_channels[cfg['gname']] = 3
        return data_channels

    def __call__(self, preds, data):
        return self.forward(preds, data)

# From AE
import torch


class LpipsLoss(torch.nn.Module):

    def __init__(self):
        super(LpipsLoss, self).__init__()
        self.loss_map = torch.zeros([0])
        self.loss_network = lpips.LPIPS().to('cuda')

    def forward(self, pred, gt):
        self.loss_map = self.loss_network(pred, gt)

        return self.loss_map.mean()


class DssimL1Loss(torch.nn.Module):

    def __init__(self):
        super(DssimL1Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = 2 * torch.abs(pred-gt) + (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False))

        return self.loss_map.mean()


class DssimSMAPELoss(torch.nn.Module):

    def __init__(self):
        super(DssimSMAPELoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = 2 * torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+0.01) + (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False))

        return self.loss_map.mean()


class SMAPELoss(torch.nn.Module):

    def __init__(self):
        super(SMAPELoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+0.01)

        return self.loss_map.mean()


class Dssim(torch.nn.Module):

    def __init__(self):
        super(Dssim, self).__init__()

    def forward(self, pred, gt):
        return pytorch_ssim.ssim(pred, gt)


class AllMetrics(torch.nn.Module):

    def __init__(self, init_lpips=False):
        super(AllMetrics, self).__init__()

        self.metrics = {}

        self.metrics['l1'] = 0
        self.metrics['l2'] = 0
        self.metrics['lpips'] = 0
        self.metrics['dssim'] = 0
        self.metrics['mape'] = 0
        self.metrics['smape'] = 0
        self.metrics['mrse'] = 0

        # LDR metrics
        self.metrics['psnr_ldr'] = 0
        self.metrics['ssim_ldr'] = 0
       

    def reset(self):
        self.metrics['l1'] = 0
        self.metrics['l2'] = 0
        self.metrics['lpips'] = 0
        self.metrics['dssim'] = 0
        self.metrics['mape'] = 0
        self.metrics['smape'] = 0
        self.metrics['mrse'] = 0

        self.metrics['psnr_ldr'] = 0
        self.metrics['ssim_ldr'] = 0

    def forward(self, pred, gt):
        # Add permutation
        pred = pred.permute(0, 3, 1, 2)
        gt = gt.permute(0, 3, 1, 2)

        diff = gt - pred
        eps = 1e-2

        self.metrics['l1'] = (torch.abs(diff)).mean()
        # self.metrics['l2'] = (diff*diff).mean()
        # self.metrics['mrse'] = (diff*diff/(gt*gt+eps)).mean()
        # self.metrics['mape'] = (torch.abs(diff)/(gt+eps)).mean()
        # self.metrics['smape'] = (2 * torch.abs(diff)/(gt+pred+eps)).mean()
        # self.metrics['dssim'] = (1.0 - (pytorch_ssim.ssim(pred, gt))).mean()
        #self.metrics['lpips'] = self.lpips_network(pred, gt)
        # conver to LDR
        # self.metrics['ssim_ldr'] = pytorch_ssim.ssim(torch.from_numpy(HDR2LDR(pred.cpu().numpy())), torch.from_numpy(HDR2LDR(gt.cpu().numpy())))
        pred = pred.permute(0, 2, 3, 1)[0] # TODO: only calculate the first image over the patch
        gt = gt.permute(0, 2, 3, 1)[0]
        pred_ldr = HDR2LDR(pred.cpu().numpy())
        gt_ldr = HDR2LDR(gt.cpu().numpy())
        self.metrics['psnr_ldr'] = psnr(gt_ldr, pred_ldr, data_range=pred_ldr.max()-pred_ldr.min())
        #self.metrics['ssim_ldr'] = ssim(gt_ldr, pred_ldr, data_range=pred_ldr.max()-pred_ldr.min(), channel_axis=-1)

        return self

    def __add__(self, other):
        self.metrics['l1'] = self.metrics['l1'] + other.metrics['l1']
        self.metrics['l2'] = self.metrics['l2'] + other.metrics['l2']
        self.metrics['lpips'] = self.metrics['lpips'] + other.metrics['lpips']
        self.metrics['dssim'] = self.metrics['dssim'] + other.metrics['dssim']
        self.metrics['mape'] = self.metrics['mape'] + other.metrics['mape']
        self.metrics['smape'] = self.metrics['smape'] + other.metrics['smape']
        self.metrics['mrse'] = self.metrics['mrse'] + other.metrics['mrse']
        self.metrics['lpips'] = self.metrics['lpips'] + other.metrics['lpips']

        self.metrics['psnr_ldr'] = self.metrics['psnr_ldr'] + other.metrics['psnr_ldr']
        self.metrics['ssim_ldr'] = self.metrics['ssim_ldr'] + other.metrics['ssim_ldr']
        return self

    def __truediv__(self, other):
        self.metrics['l1'] = self.metrics['l1'] / other.metrics['l1']
        self.metrics['l2'] = self.metrics['l2'] / other.metrics['l2']
        self.metrics['lpips'] = self.metrics['lpips'] / other.metrics['lpips']
        self.metrics['dssim'] = self.metrics['dssim'] / other.metrics['dssim']
        self.metrics['mape'] = self.metrics['mape'] / other.metrics['mape']
        self.metrics['smape'] = self.metrics['smape'] / other.metrics['smape']
        self.metrics['mrse'] = self.metrics['mrse'] / other.metrics['mrse']
        self.metrics['lpips'] = self.metrics['lpips'] / other.metrics['lpips']

        self.metrics['psnr_ldr'] = self.metrics['psnr_ldr'] / other.metrics['psnr_ldr']
        self.metrics['ssim_ldr'] = self.metrics['ssim_ldr'] / other.metrics['ssim_ldr']
        return self
