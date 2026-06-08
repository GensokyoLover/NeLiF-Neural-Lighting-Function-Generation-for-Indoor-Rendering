import numpy as np
import cmapy

def gamma(img):
    return img ** (1/2.2)

def gamma_inv(img):
    return img ** 2.2

def float2uint8(img, clip=True, clip_min=0., clip_max=1.):
    if clip:
        img = img.clip(min=clip_min, max=clip_max)
    return (img * 255).astype(np.uint8)

def depth2uint8(img):
    s = img.max() - img.min()
    img = (img - img.min()) / s
    return float2uint8(img, clip=False)

def feature2uint8(img):
    for i in range(3):
        s = img[..., i].max() - img[..., i].min()
        img[..., i] = (img[..., i] - img[..., i].min()) / s
    return float2uint8(img, clip=False)

def colormap(img, normalize=True):
    if img.shape[-1] != 1:
        print(img.shape)
        raise ValueError
    return cmapy.colorize((depth2uint8(img) if normalize else float2uint8(img)), 'viridis')

def HDR2LDR(img, clip=True, clip_min=0., clip_max=None):
    if clip:
        img = img.clip(min=clip_min, max=clip_max)
    return float2uint8(gamma(img))

def normal2LDR(img):
    img = img * 0.5 + 0.5
    return float2uint8(img)