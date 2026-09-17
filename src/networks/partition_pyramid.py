"""Progressive partition filtering used by NelifShadowNetwork."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_splat(img, kernel, size):
    h = img.shape[2]
    w = img.shape[3]

    total = torch.zeros_like(img)

    img = F.pad(img, [(size - 1) // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i + h, j:j + w] * kernel[:, i * size + j, None, :, :]
    return total


def upscale_quadrant(img, kernel, indices):
    quad = torch.zeros(
        [img.shape[0], img.shape[1], img.shape[2] * 2, img.shape[3] * 2],
        dtype=img.dtype, device=img.device
    )
    quad[:, :, 0::2, 0::2] = img * kernel[:, indices[0], None, :, :]
    quad[:, :, 0::2, 1::2] = img * kernel[:, indices[1], None, :, :]
    quad[:, :, 1::2, 0::2] = img * kernel[:, indices[2], None, :, :]
    quad[:, :, 1::2, 1::2] = img * kernel[:, indices[3], None, :, :]
    return quad


def upscale(img, kernel):
    img = F.pad(img, (1, 1, 1, 1))

    kernel = F.pad(kernel, (1, 1, 1, 1))
    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    bl = upscale_quadrant(img, kernel, [8, 9, 12, 13])
    br = upscale_quadrant(img, kernel, [10, 11, 14, 15])

    return tl[:, :, 3:-1, 3:-1] + tr[:, :, 3:-1, 1:-3] + bl[:, :, 1:-3, 3:-1] + br[:, :, 1:-3, 1:-3]


class PartitioningPyramidNew:
    """Partition shadows by scale, then filter and reconstruct coarse to fine."""

    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + K + 1] + [41 for i in range(K - 1)]

        self.final_activate = nn.LeakyReLU(0.3)

    def __call__(self, weights, shadow):

        part_weights = F.softmax(
            weights[0][:, 25:25 + self.K],
            dim=1
        )

        partitions = part_weights[:, :, None] * shadow[:, None]

        pyramid = [
            F.avg_pool2d(
                partitions[:, i],
                kernel_size=2 ** i,
                stride=2 ** i
            )
            for i in range(self.K)
        ]

        i = self.K - 1

        filter_kernel = F.softmax(
            weights[i][:, 0:25],
            dim=1
        )

        denoised = conv_splat(
            pyramid[i],
            filter_kernel,
            5
        )

        # Fuse each finer partition before applying that level's filter.
        for i in reversed(range(self.K - 1)):

            up_kernel = (
                F.softmax(
                    weights[i + 1][:, 25:41],
                    dim=1
                )
                * 4.0
            )

            denoised = upscale(
                denoised,
                up_kernel
            )

            denoised = denoised + pyramid[i]

            filter_kernel = F.softmax(
                weights[i][:, 0:25],
                dim=1
            )

            denoised = conv_splat(
                denoised,
                filter_kernel,
                5
            )

        three_weight = weights[0][:, -1, ...].unsqueeze(dim=1)

        denoised = (
            denoised
            * self.final_activate(three_weight)
        )

        return denoised
