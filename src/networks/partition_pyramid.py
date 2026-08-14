import torch
import torch.nn as nn
import torch.nn.functional as F


def splat(img, kernel, size):
    h = img.shape[2]
    w = img.shape[3]
    total = torch.zeros_like(img)

    img = F.pad(img, [(size - 1) // 2] * 4)
    kernel = F.pad(kernel, [(size - 1) // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i + h, j:j + w] * kernel[:, i * size + j, None, i:i + h, j:j + w]

    return total


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





class PartitioningPyramid():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + K + 1] + [41 for i in range(K - 1)]

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final_activate = nn.LeakyReLU(0.3)
        self.relu = nn.ReLU(inplace=False)

    def __call__(self, weights, shadow):


        part_weights = F.softmax(
            weights[0][:, 25:25 + self.K],
            dim=1
        )

        # [B, K, C, H, W]
        partitions = part_weights[:, :, None] * shadow[:, None]


        # level 0 : H
        # level 1 : H/2
        # level 2 : H/4
        # level 3 : H/8
        # level 4 : H/16
        # --------------------------------------------------
        pyramid = [
            F.avg_pool2d(
                partitions[:, i],
                kernel_size=2 ** i,
                stride=2 ** i
            )
            for i in range(self.K)
        ]

        # Level 4:
        # raw shadow -> filter
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 4. Progressive reconstruction
        #
        # Filter
        #   ↓
        # Upsample
        #   ↓
        # + current level shadow
        #   ↓
        # Filter
        #   ↓
        # Upsample
        # ...
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 5. final modulation
        # --------------------------------------------------
        three_weight = weights[0][:, -1, ...].unsqueeze(dim=1)

        denoised = (
            denoised
            * self.final_activate(three_weight)
        )

        return denoised

class TemporalPartitioningPyramid():
    def __init__(self, K = 5):
        self.K = K
        self.inputs = [25 + 25 + 1 + 1 + 1 + K] + [41 for i in range(K-1)]
        self.t_lambda_index = 51
        self.final_activate = nn.LeakyReLU(0.3)

    def __call__(self, weights, rendered, previous):   

        part_weights = F.softmax(weights[0][:, 52:], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        denoised_shadow_list = []
        for i in range(5):
            dd = denoised_levels[i]
            for j in reversed(range(i)):
                dd =  upscale(dd, F.softmax(weights[j+1][:, 25:41], 1) * 4)
            denoised_shadow_list.append(dd.permute(1,2,3,0))
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i + 1][:, 25:41], 1) * 4)
        three_weight = weights[0][:, -1, ...].unsqueeze(dim=1)
        denoised = denoised * self.final_activate(three_weight)
        for i in range(len(denoised_shadow_list)):
            denoised_shadow_list[i] = denoised_shadow_list[i] * self.final_activate(three_weight).permute(1,2,3,0)
        previous = splat(previous, F.softmax(weights[0][:, 25:50], 1), 5)
        t_mu = torch.sigmoid(weights[0][:, 50, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output


class TemporalPartitioningPyramid():
    def __init__(self, K = 5):
        self.K = K
        self.inputs = [25 + 25 + 1 + 1  + K + 1] + [41 for i in range(K-1)]
        self.t_lambda_index = 51
    
    def __call__(self, weights, rendered, previous):   

        part_weights = F.softmax(weights[0][:, 52:57], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i+1][:, 25:41], 1) * 4)

        previous = splat(previous, F.softmax(weights[0][:, 25:50], 1), 5)
        t_mu = torch.sigmoid(weights[0][:, 50, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output

class OnlyKernelPyramid():
    def __init__(self, K=4):
        self.K = K
        self.inputs = [25 + K + 1] + [25 for i in range(K - 1)]
        self.t_lambda_index = 51

    def __call__(self, weights, shadow, train_level):
        for i in range(4):
            print(weights[i].shape)
        part_weights = F.softmax(weights[0][:, 25:25 + 4 + 1], 1)[:, :4, ...]
        # starter,ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        # starter.record()
        partitions = part_weights[:, :, None] * shadow[:4, ].unsqueeze(0)
        denoised_levels = [
            conv_splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(4)
        ]
        # weights = [weight.to(torch.float32) for weight in weights]
        # ender.record()
        # torch.cuda.synchronize()
        # print("soft time: ",starter.elapsed_time(ender))
        # starter.record()
        final_result = denoised_levels[0]
        for i in range(1, 4):
            denoised_levels[i] = F.upsample(denoised_levels[i], scale_factor=2 ** i, mode="bilinear")
            final_result = final_result + denoised_levels[i]

        # ender.record()
        # torch.cuda.synchronize()
        # print("upsample time: ",starter.elapsed_time(ender))

        return final_result


class NoSoftMaxPartitioningPyramid():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + K + 1] + [41 for i in range(K - 1)]
        self.t_lambda_index = 51

    def __call__(self, weights, shadow):
        # part_weights = F.softmax(weights[0][:, 25:30], 1)
        part_weights = weights[0][:, 25:30]
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        partitions = part_weights[:, :, None] * shadow[:, None]

        denoised_levels = [
            conv_splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                weights[i][:, 0:25],
                5
            )
            for i in range(self.K)
        ]
        # weights = [weight.to(torch.float32) for weight in weights]
        ender.record()
        torch.cuda.synchronize()
        #print("soft time: ", starter.elapsed_time(ender))
        starter.record()
        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            # print("denoised shadow {} shape".format(i),denoised.shape)
            denoised = denoised_levels[i] + upscale(denoised, weights[i + 1][:, 25:41] * 4)
        ender.record()
        torch.cuda.synchronize()
        #print("upsample time: ", starter.elapsed_time(ender))

        return denoised


# no upsampling

class PartitioningPyramidSmall():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + K + 1] + [25 for i in range(K - 1)]
        self.t_lambda_index = 51

    def __call__(self, weights, shadow):
        part_weights = F.softmax(weights[0][:, 25:30], 1)
        partitions = part_weights[:, :, None] * shadow[:, None]

        denoised_levels = [
            conv_splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i + 1][:, 25:41], 1) * 4)

        return denoised


class PartitioningPyramidNoUpsample():
    def __init__(self, K=5):
        self.K = K
        self.inputs = [25 + K + 1] + [25 for i in range(K - 1)]
        self.t_lambda_index = 51

    def __call__(self, weights, shadow):
        part_weights = F.softmax(weights[0][:, 25:30], 1)
        partitions = part_weights[:, :, None] * shadow[:, None]

        denoised_levels = [
            conv_splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i + 1][:, 25:41], 1) * 4)
        return denoised