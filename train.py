import argparse
import itertools
import os
import time
from os import listdir
from os.path import join
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
import lib.pytorch_ssim as pytorch_ssim
from lib.data import get_training_set, is_image_file, get_Low_light_training_set
from lib.utils import TVLoss, print_network,MSSSIM
from model import BSWN
import lpips
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Name_Exp = 'BSWN'
exp = Experiment(Name_Exp)
exp.add_source_file("train.py")
exp.add_source_file("model.py")
exp.add_source_file("lib/dataset.py")
exp.captured_out_filter = apply_backspaces_and_linefeeds
@exp.config
def cfg():
    parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
    parser.add_argument('--batchSize', type=int, default=5, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped LR image')
    parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
    parser.add_argument('--isdimColor', default=True, help='synthesis at HSV color space')
    parser.add_argument('--isaddNoise', default=True, help='synthesis with noise')
    parser.add_argument('--in_channels', type=int, default=3, help='input channels for generator')
    parser.add_argument('--out_channels', type=int, default=3, help='output channels for generator')
    parser.add_argument('--start_channels', type=int, default=32, help='start channels for generator')
    parser.add_argument('--pad', type=str, default='zero', help='pad type of networks')
    parser.add_argument('--norm', type=str, default='none', help='normalization type of networks')
    parser.add_argument('--m_block', type=int, default=2, help='the additional blocks used in mainstream')
    opt = parser.parse_args()
def checkpoint(model, epoch, opt):
    try:
        os.stat(opt.save_folder)
    except:
        os.mkdir(opt.save_folder)
    model_out_path = opt.save_folder + "{}_{}.pth".format(Name_Exp, epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
def log_metrics(_run, logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        _run.log_scalar(key, float(value), iter)
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)
def eval(model, epoch):
    print("==> Start testing")
    tStart = time.time()
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    model.eval()#在非训练的时候是需要加上这句代码的，若没有，一些网络层的值会发生变动，不会固定，神经网络的每一次生成的结果也是不固定的，如dropout层和BN层，生成的质量可能好也可能不好
    test_LL_folder = "datasets/LOL/test/low/"
    test_NL_folder = "datasets/LOL/test/high/"
    test_est_folder = "outputs/eopch_%04d/" % (epoch)
    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)
    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    for i in range(test_LL_list.__len__()):
        with torch.no_grad():#with torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
            LL = trans(Image.open(test_LL_list[i]).convert('RGB')).unsqueeze(0).cuda()
            prediction = model(LL)
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    lpips_score = 0.0
    loss_fn_alex = lpips.LPIPS(net='alex')
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        gt1=torch.from_numpy(gt).permute(2,0,1).float()
        est1=torch.from_numpy(est).permute(2,0,1).float()
        lpips_val=loss_fn_alex(est1.unsqueeze(0).cpu(),gt1.unsqueeze(0).cpu())
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
        lpips_score = lpips_score + lpips_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    lpips_score = lpips_score / (test_NL_list.__len__())
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    return psnr_score, ssim_score,lpips_score
@exp.automain
def main(opt, _run):
    cuda = opt.gpu_mode#default=true
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)#设置CPU生成的随机种子，方便下次复现实验结果
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.benchmark = True#可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
        #在保证输入数据的维度或类型变化不大的情况下，设置这个flag可以增加运行效率
    best_psnr = 0
    best_ssim =0
    best_lpips =100.0
    best_epoch =0
    # =============================#
    #   Prepare training data     #
    # =============================#
    print('===> Prepare training data')
    train_set = get_training_set("datasets/LOL/train", 1, opt.patch_size, True) # uncomment it to do the fine tuning
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True, drop_last=False)
    print('===> Build model')
    lighten =BSWN()
    lighten = torch.nn.DataParallel(lighten)#DataParallel会自动帮我们将数据切分load到相应的GPU中，将模型复制到相应的GPU中，进行正向传播计算梯度并汇总
    #lighten.load_state_dict(torch.load("DLN_journal.pth", map_location=lambda storage, loc: storage), strict=True)
#torch.nn.Module类提供了将模型参数的作为字典映射保存和加载的方法，strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合
    print('---------- Networks architecture -------------')
    print_network(lighten)
    print('----------------------------------------------')
    if cuda:
        lighten = lighten.cuda(0)
    # =============================#nvi
    #         Loss function       #
    # =============================#
    L1_criterion = nn.L1Loss()#nn.L1loss取预测值和真实值的绝对误差的平均数
    ssim = pytorch_ssim.SSIM()
    if cuda:
        gpus_list = range(opt.gpus)
        L1_criterion = L1_criterion.cuda()
        ssim=ssim.cuda(gpus_list[0])
    # =============================#
    #         Optimizer            #
    # =============================#
    parameters = [lighten.parameters()]
    optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    print(len(training_data_loader))
    # =============================#
    #         Training             #
    # =============================#
    psnr_score, ssim_score,niqe_score = eval(lighten, 0)
    print(psnr_score)
    for epoch in range(opt.start_iter, opt.nEpochs + 1):#0-500
        print('===> training epoch %d' % epoch)
        epoch_loss = 0
        lighten.train()
        tStart_epoch = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):#返回两个值 一个索引，一个是数据
            over_Iter = epoch * len(training_data_loader) + iteration

            optimizer.zero_grad()

            LL_t, NL_t = batch[0], batch[1]
            if cuda:
                LL_t = LL_t.cuda()
                NL_t = NL_t.cuda()

            t0 = time.time()
            pred_t = lighten(LL_t)
            ssim_loss = 1 - ssim(pred_t, NL_t)
            l1_loss= L1_criterion(pred_t,NL_t)
            loss = 0.16*ssim_loss + 0.84 * l1_loss

            loss.backward()
            optimizer.step()
            t1 = time.time()

            epoch_loss += loss

            if iteration % 10 == 0:
                print("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)),
                      end=" ==> ")
                logs = {
                    "loss": loss.data,
                    "ssim_loss": ssim_loss.data,
                    "l1_loss":  l1_loss.data,
                }
                log_metrics(_run, logs, over_Iter)
                print("time: {:.4f} s".format(t1 - t0))

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}; ==> {:.2f} seconds".format(epoch, epoch_loss / len(
            training_data_loader), time.time() - tStart_epoch))
        _run.log_scalar("epoch_loss", float(epoch_loss / len(training_data_loader)), epoch)

        if epoch % (opt.snapshots) == 0:
            file_checkpoint = checkpoint(lighten, epoch, opt)
            exp.add_artifact(file_checkpoint)

            psnr_score, ssim_score,lpips_score = eval(lighten, epoch)
            if best_psnr < psnr_score:
                best_psnr=psnr_score
                best_epoch=epoch
            if best_ssim < ssim_score:
                best_ssim = ssim_score
            if best_lpips > lpips_score:
                best_lpips = lpips_score
            logs = {
                "psnr": psnr_score,
                "ssim": ssim_score,
                "lpips": lpips_score,
                "best_psnr":best_psnr,
                "best_ssim": best_ssim,
                "best_lpips": best_lpips,
                "best_epoch":best_epoch,
            }
            log_metrics(_run, logs, epoch, end_str="\n")
        #调整学习率lr的策略
        if (epoch + 1) % (opt.nEpochs * 2 / 3) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
