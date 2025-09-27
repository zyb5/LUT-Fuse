from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from glob import glob
import transforms as T
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F

# class SimpleDataSet(Dataset):
#     def __init__(self, visible_path, visible_gt_path, infrared_path, phase="train", transform=None):
#         self.phase = phase
#         self.visible_files = sorted(glob(os.path.join(visible_path, "*")))
#         self.visible_gt_files = sorted(glob(os.path.join(visible_gt_path, "*")))
#         self.infrared_files = sorted(glob(os.path.join(infrared_path, "*")))
#         self.transform = transform
#
#     def __len__(self):
#         l = len(self.infrared_files)
#         return l
#
#     def __getitem__(self, item):
#         image_A_path = self.visible_files[item]
#         image_A_gt_path = self.visible_gt_files[item]
#         image_B_path = self.infrared_files[item]
#         image_A = Image.open(image_A_path).convert(mode='RGB')
#         image_A_gt = Image.open(image_A_gt_path).convert(mode='RGB')
#         image_B = Image.open(image_B_path).convert(mode='L')   ##########
#
#         image_A = self.transform(image_A)
#         image_A_gt = self.transform(image_A_gt)
#         image_B = self.transform(image_B)
#
#         name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]
#
#         return image_A, image_A_gt, image_B, name
#
#     @staticmethod
#     def collate_fn(batch):
#         images_A, image_A_gt, images_B, name = zip(*batch)
#         images_A = torch.stack(images_A, dim=0)
#         image_A_gt = torch.stack(image_A_gt, dim=0)
#         images_B = torch.stack(images_B, dim=0)
#         return images_A, image_A_gt, images_B, name

# class RandomCropPair:
#     def __init__(self, size):
#         self.size = size  # 裁剪尺寸 (h, w)
#
#     def __call__(self, vis_img, ir_img):
#         # 获取随机裁剪参数
#         i, j, h, w = RandomCrop.get_params(vis_img, output_size=self.size)
#         # 对可见光和红外图像使用相同的裁剪参数
#         vis_img = F.crop(vis_img, i, j, h, w)
#         ir_img = F.crop(ir_img, i, j, h, w)
#
#         vis_img = F.to_tensor(vis_img)
#         ir_img = F.to_tensor(ir_img)
#         return vis_img, ir_img
#
# class SimpleDataSet(Dataset):
#     def __init__(self, visible_path, infrared_path, phase="train", transform=None):
#         self.phase = phase
#         self.visible_files = sorted(glob(os.path.join(visible_path, "*.*")))
#         self.infrared_files = sorted(glob(os.path.join(infrared_path, "*.*")))
#         self.transform = transform
#
#     def __len__(self):
#         l = len(self.infrared_files)
#         return l
#
#     def __getitem__(self, item):
#         image_A_path = self.visible_files[item]
#         image_B_path = self.infrared_files[item]
#         image_A = Image.open(image_A_path).convert(mode='RGB')
#         image_B = Image.open(image_B_path).convert(mode='L')   ##########
#
#         # image_A = self.transform(image_A)
#         # image_B = self.transform(image_B)
#
#         if self.transform is not None:
#             if isinstance(self.transform, RandomCropPair):
#                 image_A, image_B = self.transform(image_A, image_B)
#             else:
#                 image_A = self.transform(image_A)
#                 image_B = self.transform(image_B)
#
#         name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]
#
#         return image_A, image_B, name
#
#     @staticmethod
#     def collate_fn(batch):
#         images_A, images_B, name = zip(*batch)
#         images_A = torch.stack(images_A, dim=0)
#         images_B = torch.stack(images_B, dim=0)
#         return images_A, images_B, name
#
#
# class RandomCropPair:
#     def __init__(self, size):
#         self.size = size  # 裁剪尺寸 (h, w)
#
#     def __call__(self, vis_blur_img, ir_blur_img, vis_gt_img, ir_gt_img):
#         # 获取随机裁剪参数
#         i, j, h, w = RandomCrop.get_params(vis_blur_img, output_size=self.size)
#         # 对可见光和红外图像使用相同的裁剪参数
#         vis_blur_img = F.crop(vis_blur_img, i, j, h, w)
#         ir_blur_img = F.crop(ir_blur_img, i, j, h, w)
#         vis_gt_img = F.crop(vis_gt_img, i, j, h, w)
#         ir_gt_img = F.crop(ir_gt_img, i, j, h, w)
#
#         vis_blur_img = F.to_tensor(vis_blur_img)
#         ir_blur_img = F.to_tensor(ir_blur_img)
#         vis_gt_img = F.to_tensor(vis_gt_img)
#         ir_gt_img = F.to_tensor(ir_gt_img)
#         return vis_blur_img, ir_blur_img, vis_gt_img, ir_gt_img
#
# class SimpleDataSet(Dataset):
#     def __init__(self, visible_blur_path, infrared_blur_path, visible_gt_path, infrared_gt_path, phase="train", transform=None):
#         self.phase = phase
#         self.visible_blur_files = sorted(glob(os.path.join(visible_blur_path, "*.*")))
#         self.infrared_blur_files = sorted(glob(os.path.join(infrared_blur_path, "*.*")))
#         self.visible_gt_files = sorted(glob(os.path.join(visible_gt_path, "*.*")))
#         self.infrared_gt_files = sorted(glob(os.path.join(infrared_gt_path, "*.*")))
#         self.transform = transform
#
#     def __len__(self):
#         l = len(self.infrared_gt_files)
#         return l
#
#     def __getitem__(self, item):
#         image_A_blur_path = self.visible_blur_files[item]
#         image_B_blur_path = self.infrared_blur_files[item]
#         image_A_gt_path = self.visible_gt_files[item]
#         image_B_gt_path = self.infrared_gt_files[item]
#         image_A_blur = Image.open(image_A_blur_path).convert(mode='RGB')
#         image_B_blur = Image.open(image_B_blur_path).convert(mode='L')   ##########
#         image_A_gt = Image.open(image_A_gt_path).convert(mode='RGB')
#         image_B_gt = Image.open(image_B_gt_path).convert(mode='L')  ##########
#
#         if self.transform is not None:
#             if isinstance(self.transform, RandomCropPair):
#                 image_A_blur, image_B_blur, image_A_gt, image_B_gt = self.transform(image_A_blur, image_B_blur, image_A_gt, image_B_gt)
#             else:
#                 image_A_blur = self.transform(image_A_blur)
#                 image_B_blur = self.transform(image_B_blur)
#                 image_A_gt = self.transform(image_A_gt)
#                 image_B_gt = self.transform(image_B_gt)
#
#         name = image_A_blur_path.replace("\\", "/").split("/")[-1].split(".")[0]
#
#         return image_A_blur, image_B_blur, image_A_gt, image_B_gt, name
#
#     @staticmethod
#     def collate_fn(batch):
#         image_A_blur, image_B_blur, image_A_gt, image_B_gt, name = zip(*batch)
#         image_A_blur = torch.stack(image_A_blur, dim=0)
#         image_B_blur = torch.stack(image_B_blur, dim=0)
#         image_A_gt = torch.stack(image_A_gt, dim=0)
#         image_B_gt = torch.stack(image_B_gt, dim=0)
#         return image_A_blur, image_B_blur, image_A_gt, image_B_gt, name


class RandomCropPair:
    def __init__(self, size):
        self.size = size  # 裁剪尺寸 (h, w)

    def __call__(self, vis_img, ir_img):
        # 获取随机裁剪参数
        i, j, h, w = RandomCrop.get_params(vis_img, output_size=self.size)
        # 对可见光和红外图像使用相同的裁剪参数
        vis_img = F.crop(vis_img, i, j, h, w)
        ir_img = F.crop(ir_img, i, j, h, w)

        vis_img = F.to_tensor(vis_img)
        ir_img = F.to_tensor(ir_img)
        return vis_img, ir_img

class SimpleDataSet(Dataset):
    def __init__(self, visible_path, infrared_path, phase="train", transform=None):
        self.phase = phase
        self.visible_files = sorted(glob(os.path.join(visible_path, "*.*")))
        self.infrared_files = sorted(glob(os.path.join(infrared_path, "*.*")))
        self.transform = transform

    def __len__(self):
        l = len(self.infrared_files)
        return l

    def __getitem__(self, item):
        image_A_path = self.visible_files[item]
        image_B_path = self.infrared_files[item]
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='L')   ##########

        # image_A = self.transform(image_A)
        # image_B = self.transform(image_B)

        if self.transform is not None:
            if isinstance(self.transform, RandomCropPair):
                image_A, image_B = self.transform(image_A, image_B)
            else:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        return images_A, images_B, name


