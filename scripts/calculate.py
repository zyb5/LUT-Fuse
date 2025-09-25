import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import time
from data.simple_dataset import RandomCropPair
from data.simple_dataset import SimpleDataSet
import torch.nn as nn
import transforms as T
import os


def rgb_to_ycbcr(img):
    return torch.stack(
        (0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
               128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
               128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
               dim=1)


def ycbcr_to_rgb(img):
    return torch.stack(
         (img[:, 0, :, :] + (img[:, 2, :, :] - 0.5) * 1.402,
                img[:, 0, :, :] - (img[:, 1, :, :] - 0.5) * 0.344136 - (img[:, 2, :, :] - 0.5) * 0.714136,
                img[:, 0, :, :] + (img[:, 1, :, :] - 0.5) * 1.772),
                dim=1)


def load_lookup_table(filepath):
    try:
        lut = np.load(filepath).astype(np.float32)
        lut = torch.tensor(lut, device="cuda")  # 将查找表移到 GPU 上
        return lut
    except Exception as e:
        print(f"加载查找表时出错: {e}")
        return None


def generator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))
    return layers


class Generator_for_info(nn.Module):
    def __init__(self, in_channels=4):
        super(Generator_for_info, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),)

        self.mid_layer = nn.Sequential(
            *generator_block(16, 16, normalization=True),
            *generator_block(16, 16, normalization=True),
            *generator_block(16, 16, normalization=True),)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, img_input):
        x = self.input_layer(img_input)
        identity = x
        out = self.mid_layer(x)
        out += identity
        out = self.output_layer(out)
        return out


def apply_fusion_4d_with_interpolation(visible_img, infrared_img, lut, get_context):

    image_cat = torch.cat((visible_img, infrared_img), dim=1)  # [0, 255]
    context = get_context(image_cat)

    context_scaled = (context *255./ 16.0).squeeze(1)  # [0, 16]
    infrared_scaled = infrared_img / 16.0  # [0, 16]

    ycbcr_vis = rgb_to_ycbcr(visible_img / 255.)  # [0, 1]
    ycbcr_vis_scaled = ycbcr_vis * 255.0 / 16.0  # [0, 16]

    y_vi_scaled = ycbcr_vis_scaled[:, 0, :, :]  # [b, 1, h, w]   # [0, 16]
    cb_cr = ycbcr_vis[:, 1:, :, :]  # [0, 1]
    ir_scaled = infrared_scaled[:, 0, :, :]  # [0, 16]

    # 获取floor和ceil索引
    ir_floor = torch.floor(ir_scaled).long()
    ir_ceil = torch.clamp(ir_floor + 1, 0, lut.shape[3] - 1)
    ir_alpha = ir_scaled - ir_floor

    y_vi_floor = torch.floor(y_vi_scaled).long()
    y_vi_ceil = torch.clamp(y_vi_floor + 1, 0, lut.shape[0] - 1)
    y_vi_alpha = y_vi_scaled - y_vi_floor

    c_floor = torch.floor(context_scaled).long()
    c_ceil = torch.clamp(c_floor + 1, 0, lut.shape[2] - 1)
    c_alpha = context_scaled - c_floor

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        visible_img.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
        visible_img.device)

    # 计算 x 和 y 方向的梯度
    grad_x = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(ycbcr_vis[:, :1, :, :], sobel_y, padding=1)

    # 计算梯度幅值
    gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # [b, 1, h, w]
    min_val = gradient.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_val = gradient.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values

    gradient_normalized = (gradient - min_val) / (max_val - min_val + 1e-8)

    gradient_scaled = (gradient_normalized * 255.)
    gradient_scaled = (gradient_scaled / 16.0).squeeze(1)

    g_floor = torch.floor(gradient_scaled).long()
    g_ceil = torch.clamp(g_floor + 1, 0, lut.shape[1] - 1)
    g_alpha = gradient_scaled - g_floor

    ir_alpha = ir_alpha.unsqueeze(-1)
    y_vi_alpha = y_vi_alpha.unsqueeze(-1)
    g_alpha = g_alpha.unsqueeze(-1)
    c_alpha = c_alpha.unsqueeze(-1)

    def lerp(v1, v2, alpha):
        out = v1 * (1 - alpha) + v2 * alpha
        return out

    fusion_result = (
        lerp(
            lerp(
                lerp(
                    lerp(lut[y_vi_floor, g_floor, c_floor, ir_floor], lut[y_vi_floor, g_floor, c_floor, ir_ceil],
                         ir_alpha),
                    lerp(lut[y_vi_floor, g_floor, c_ceil, ir_floor], lut[y_vi_floor, g_floor, c_ceil, ir_ceil],
                         ir_alpha),
                    c_alpha,
                ),
                lerp(
                    lerp(lut[y_vi_floor, g_ceil, c_floor, ir_floor], lut[y_vi_floor, g_ceil, c_floor, ir_ceil],
                         ir_alpha),
                    lerp(lut[y_vi_floor, g_ceil, c_ceil, ir_floor], lut[y_vi_floor, g_ceil, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                g_alpha,
            ),
            lerp(
                lerp(
                    lerp(lut[y_vi_ceil, g_floor, c_floor, ir_floor], lut[y_vi_ceil, g_floor, c_floor, ir_ceil],
                         ir_alpha),
                    lerp(lut[y_vi_ceil, g_floor, c_ceil, ir_floor], lut[y_vi_ceil, g_floor, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                lerp(
                    lerp(lut[y_vi_ceil, g_ceil, c_floor, ir_floor], lut[y_vi_ceil, g_ceil, c_floor, ir_ceil], ir_alpha),
                    lerp(lut[y_vi_ceil, g_ceil, c_ceil, ir_floor], lut[y_vi_ceil, g_ceil, c_ceil, ir_ceil], ir_alpha),
                    c_alpha,
                ),
                g_alpha,
            ),
            y_vi_alpha,
        )
    )

    fusion_y = fusion_result.permute(0, 3, 1, 2)

    fusion_ycbcr = torch.cat([fusion_y, cb_cr], dim=1)
    fusion_rgb = ycbcr_to_rgb(fusion_ycbcr)  # fusion_rgb = fusion_ycbcr.permute(0, 3, 1, 2)

    return fusion_rgb

