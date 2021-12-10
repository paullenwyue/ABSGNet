from __future__ import print_function
import argparse
import torch
from model import BSWN
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir
import lpips
import os
import cv2
from joblib import load
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from  niqe import niqe
from IQA_pytorch import FSIM
from brisque import brisque
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=32, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default="./datasets/LOL/test/low/")
parser.add_argument('--model_type', type=str, default='FG')
parser.add_argument('--output', default='./output/', help='Location to save checkpoint models')
parser.add_argument('--modelfile', default='./checkpoint/FGNet_params_best.pkl', help='sr pretrained base model')
parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
parser.add_argument('--chop', type=bool, default=False)

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
print('===> Building model ', opt.model_type)
os.environ['CUDA_VISIBLE_DEVICES']='0'
model = BSWN()
#model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(
    opt.modelfile))
if cuda:
    model = model.cuda(gpus_list[0])
model1=FSIM(channels=3).cuda()
def eval():
    model.eval()
    LL_filename = os.path.join(opt.image_dataset)
    test_NL_folder = "./datasets/LOL/test/high/"
    est_filename = os.path.join(opt.output)
    tStart = time.time()
    try:
        os.stat(est_filename)
    except:
        os.mkdir(est_filename)
    LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    print(LL_filename)
    Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    print(Est_img)
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_NL_folder)) if is_image_file(x)]
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    time_ave = 0
    lpips_score = 0.0
    fsim_score=0.0
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    for i in range(LL_image.__len__()):
        with torch.no_grad():
            LL_in = Image.open(LL_image[i]).convert('RGB')
            LL = trans(LL_in)
            LL = LL.unsqueeze(0)
            LL = LL.cuda()
            NL_in = Image.open(test_NL_list[i]).convert('RGB')
            NL = trans(NL_in)
            NL = NL.unsqueeze(0)
            NL = NL.cuda()
            t0 = time.time()
            prediction = model(LL)
            lpips_val = loss_fn_alex(prediction,NL)
            lpips_score = lpips_score + lpips_val
            score=model1(prediction,NL,as_loss=False)
            fsim_score=fsim_score+score.item()
            t1 = time.time()
            time_ave += (t1 - t0)
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(Est_img[i])
            print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))
    psnr_score = 0.0
    ssim_score = 0.0
    UQI_score = 0.0
    lpips_score = 0.0
    # niqe_score=0.0
    brisque_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(Est_img[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        UQI_val= UQI(est,gt)
        # niqe_val=niqe(est)
        brisque_val=brisque(est)
        brisque_val=brisque_val.reshape(1,-1)
        clf=load('svr_brisque.joblib')
        score1=clf.predict(brisque_val)[0]
        gt1 = torch.from_numpy(gt).permute(2, 0, 1).float()
        est1 = torch.from_numpy(est).permute(2, 0, 1).float()
        lpips_val = loss_fn_alex(est1.unsqueeze(0).cuda(), gt1.unsqueeze(0).cuda())
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
        lpips_score=lpips_score+lpips_val
        brisque_score=brisque_score+score1
        print(test_NL_list[i])
        print("PSNR: %.4f " % (psnr_val))
        print("SSIM: %.4f " % (ssim_val))
        UQI_score+=UQI_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    lpips_score = lpips_score / (test_NL_list.__len__())
    UQI_score=UQI_score/(test_NL_list.__len__())
    fsim_score=fsim_score/15
    # niqe_score=niqe_score/15
    brisque_score=brisque_score/15
    print("LPIPS: %.4f " % (lpips_score))
    print("PSNR: %.4f " % (psnr_score))
    print("SSIM: %.4f " %(ssim_score))
    print("FSIM: %.4f " % (fsim_score))
    print("UQI: %.4f " %(UQI_score))
    # print("NIQE: %.4f " % (niqe_score))
    print("brisque_score: %.4f " % (brisque_score))
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    print("===> Processing Time: %.4f ms." % (time_ave / LL_image.__len__() * 1000))
def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    return img
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        float32, [0, 255]
        float32, [0, 255]
    '''
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()

    return rlt


def PSNR(pred, gt, shave_border):
    pred = pred[shave_border:-shave_border, shave_border:-shave_border]
    gt = gt[shave_border:-shave_border, shave_border:-shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)


def image_mean(image):
    mean = np.mean(image)
    return mean


def image_var(image, mean):
    # print(np.shape(image))
    m, n, _ = np.shape(image)
    var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
    return var


def images_cov(image1, image2, mean1, mean2):
    m, n, _ = np.shape(image1)
    cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
    return cov


def UQI(O1, F1):
    meanO = image_mean(O1)
    meanF = image_mean(F1)
    varO = image_var(O1, meanO)
    varF = image_var(F1, meanF)
    covOF = images_cov(O1, F1, meanO, meanF)
    UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
    return UQI
##Eval Start!!!!
eval()
