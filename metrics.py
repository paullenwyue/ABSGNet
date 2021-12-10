
import numpy as np
from skimage.measure import compare_ssim


def psnr(img_rec, img_ori):
    img_rec = img_rec.astype(np.float32)
    img_ori = img_ori.astype(np.float32)

    max_gray = 255.
    mse = np.mean(np.power(img_rec - img_ori, 2))
    if mse == 0:
        return 100
    return 10. * np.log10(max_gray ** 2 / mse)


def ssim(img_rec, img_ori):
    return compare_ssim(img_rec, img_ori, data_range=255, multichannel=True)
