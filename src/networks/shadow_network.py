"""Shadow reconstruction components used by NelifDecoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .partition_pyramid import PartitioningPyramidNew


def reversedl(it):
    return list(reversed(list(it)))


class MidConvUNet(nn.Module):
    """Predict filter and upsampling weights at each pyramid level."""

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

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode="bilinear"), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))

        return reversedl(outputs)


class NelifShadowNetwork(nn.Module):
    """Reconstruct RGB shadows through the decoder's step(data) interface."""

    def __init__(self,input_dim,dims_and_depths=[
                    (20, 20),
                    (36, 36),
                    (54, 54),
                    (76, 76),
                    (96, 96),
                    (128, 128, 96)
                ]):

        super().__init__()
        self.filter = PartitioningPyramidNew()

        self.weight_predictor = MidConvUNet(
            input_dim,
            self.filter.inputs,
            dims_and_depths=dims_and_depths
        )

    def step(self, data):
        # Light features are arranged as B * 3 independent color channels.
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
