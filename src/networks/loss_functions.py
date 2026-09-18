"""Losses and PSNR evaluation used by NelifDecoder."""

from abc import abstractmethod

import torch
import torch.nn as nn

from . import pytorch_ssim


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
    img_tensor = torch.clamp(img_tensor, min=0.0, max=1.0)
    img_ldr = torch.pow(img_tensor + 1e-8, 1.0 / 2.2)
    return img_ldr


def calculate_psnr_ldr_torch(gt, pred):
    """
    完全在 GPU 上运行的极速 PSNR 计算
    """
    pred_first = pred[0]
    gt_first = gt[0]

    pred_ldr = HDR2LDR_torch(pred_first)
    gt_ldr = HDR2LDR_torch(gt_first)

    mse = torch.mean((gt_ldr - pred_ldr) ** 2)

    if mse < 1e-10:
        return torch.tensor(float('inf'), device=pred.device)

    data_range = pred_ldr.max() - pred_ldr.min()

    psnr_val = 10.0 * torch.log10((data_range ** 2) / mse)

    return psnr_val


class L1Loss(LossFunction):
    def forward(self, pred, gt):
        return torch.abs(pred-gt)


class SSIMLoss(LossFunction):
    def forward(self, pred, gt):
        return 1.0 - pytorch_ssim.ssim(pred, gt) # It seems like ssim already means along the dimensions of rgb images


class Loss(torch.nn.Module):
    def __init__(self, configs):
        super().__init__()
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
            if cfg['loss'] == 'L1Mask':
                loss = loss_func(pred, gt, mask=mask.bool())
            else:
                loss = loss_func(pred, gt)

            if cfg['reweigh']:
                liangdu = data["log1p_diffuse_direct_shading"].permute(0,3,1,2)
                liangdu = liangdu + data["log1p_specular_direct_shading"].permute(0,3,1,2)
                loss *= liangdu * 0.3

            if "special_weight" in cfg.keys():
                pixel_wise_weight = data["log1p_direct_shading"]
                pixel_wise_weight = pixel_wise_weight.permute(0,3,1,2)
                loss = loss * pixel_wise_weight

            if cfg["relative"]:
                liangdu = (data["log1p_diffuse_direct_shading"].permute(0,3,1,2)  + data["log1p_specular_direct_shading"].permute(0,3,1,2)).mean(dim=(1,2,3), keepdim=True) + 1e-3
            if torch.any(torch.isinf(loss)) or torch.any(torch.isnan(loss)):
                print('NaN/Inf in {} Loss '.format(l_name))

            loss_map['final_loss'] += cfg['weight'] * loss.mean()
            if cfg['visualize']:
                loss_map[l_name] = loss
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
