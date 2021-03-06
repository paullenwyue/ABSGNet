from os.path import join

from torchvision.transforms import Compose, ToTensor

from lib.dataset import DatasetFromFolderEval, DatasetFromFolder, Lowlight_DatasetFromVOC


def transform():
    return Compose([
        ToTensor(),#transforms.ToTensor()的操作对象为PIL格式的图像及numpy(cv2读取数据)，转为tensor格式，HWC->CHnO
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#归一化，把需要处理的数据经过处理后限制在你需要的一定范围内
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, 'high')
    lr_dir = join(data_dir, 'low')
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                             transform=transform())

def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir, upscale_factor,
                             transform=transform())

def get_Low_light_training_set(upscale_factor, patch_size, data_augmentation):
    return Lowlight_DatasetFromVOC(patch_size, upscale_factor, data_augmentation,
                             transform=transform())

