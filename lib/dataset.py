import os
import random
import sys
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps, ImageEnhance

path = os.getcwd()
sys.path.append(path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')#Image.open()打开的图片是PIL类型，默认RG
    # y, _, _ = img.split()
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1#1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((ty, tx, ty + tp, tx + tp))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    # img_bic = img_bic.crop((ty, tx, ty + tp, tx + tp))

    # info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        # img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            # img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            # img_bic = img_bic.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(DatasetFromFolder, self).__init__()
        self.hr_image_filenames = [join(HR_dir, x) for x in  sorted(listdir(HR_dir)) if is_image_file(x)]
        self.lr_image_filenames = [join(LR_dir, x) for x in  sorted(listdir(LR_dir)) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):

        target = load_img(self.hr_image_filenames[index])
        # name = self.hr_image_filenames[index]
        # lr_name = name[:25]+'LR/'+name[28:-4]+'x4.png'
        # lr_name = name[:18] + 'LR_4x/' + name[21:]
        input = load_img(self.lr_image_filenames[index])

        # target = ImageOps.equalize(target)
        # input_eq = ImageOps.equalize(input)
        # target = ImageOps.equalize(target)
        #         # input = ImageOps.equalize(input)
        input, target, = get_patch(input, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            img_in, img_tar, _ = augment(input, target)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.hr_image_filenames)


class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, upscale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            # input = self.transform(input)
            bicubic = self.transform(bicubic)

        return bicubic, file

    def __len__(self):
        return len(self.image_filenames)


class Lowlight_DatasetFromVOC(data.Dataset):
    def __init__(self, patch_size, upscale_factor, data_augmentation,
                 transform=None):
        super(Lowlight_DatasetFromVOC, self).__init__()
        self.imgFolder = "datasets/VOC2007/JPEGImages"
        self.image_filenames = [join(self.imgFolder, x) for x in listdir(self.imgFolder) if is_image_file(x)]#连接两个或更多的路径名组件
        #1.如果各组件名的首字母不包含‘/’，则函数会自动加上  2.如果有一个组件是绝对路径，则它之前的所有组件均会被舍弃 3.如果最后一个组件为空，则它之前的所有组件均会被丢弃
        #3.如果最后一个组件为空，则生成的路径以一个‘/’分隔符结尾
        #os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表

        self.patch_size = patch_size#128
        self.upscale_factor = upscale_factor#1
        self.transform = transform
        self.data_augmentation = data_augmentation#true

    def __getitem__(self, index):

        ori_img = load_img(self.image_filenames[index])  # PIL image
        width, height = ori_img.size
        ratio = min(width, height) / 384
        newWidth = int(width / ratio)
        newHeight = int(height / ratio)
        ori_img = ori_img.resize((newWidth, newHeight), Image.ANTIALIAS)#Image.ANTIALIAS(高质量)
        high_image = ori_img#random.random用于生成0-1的随机浮点数
        ## color and contrast *dim*
        color_dim_factor = 0.3 * random.random() + 0.7
        contrast_dim_factor = 0.3 * random.random() + 0.7#变化范围为0.7-1
        ori_img = ImageEnhance.Color(ori_img).enhance(color_dim_factor)#ImageEnhance类专门用于图像的增强处理，不仅可以增强(减弱)图像的亮度，对比度，色度，还可以用于增强图像的锐度
        #当下的所有数据增强都是在没有OpenCV库的情况下，使用Numpy完成的，其中factor为1为将返回原始图像的拷贝factor越小,颜色越少(亮度，对比度等)
        #Color类用于调整图像的颜色均衡，在某种程度上类似控制彩色电视机 0.1-0.5-0.8-2.0饱和度依次增大
        ori_img = ImageEnhance.Contrast(ori_img).enhance(contrast_dim_factor)

        ori_img = cv2.cvtColor((np.asarray(ori_img)), cv2.COLOR_RGB2BGR)  # cv2 image
        ori_img = (ori_img.clip(0, 255)).astype("uint8")
        low_img = ori_img.astype('double') / 255.0

        # generate low-light image
        beta = 0.5 * random.random() + 0.5
        alpha = 0.1 * random.random() + 0.9
        gamma = 3.5 * random.random() + 1.5
        low_img = beta * np.power(alpha * low_img, gamma)

        low_img = low_img * 255.0
        low_img = (low_img.clip(0, 255)).astype("uint8")
        low_img = Image.fromarray(cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB))

        img_in, img_tar = get_patch(low_img, high_image, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            img_in, img_tar, _ = augment(img_in, img_tar)

        if self.transform:
            img_in = self.transform(img_in)
            img_tar = self.transform(img_tar)

        return img_in, img_tar

    def __len__(self):
        return len(self.image_filenames)
