import argparse
import os
import sys
import pdb
import cv2
from niqe import niqe
parser = argparse.ArgumentParser('Test an image')
parser.add_argument(
    '--mode', choices=['brisque', 'niqe', 'piqe'], help='iqa algorithoms,brisque or niqe or piqe')
parser.add_argument('--path', required=True, help='image path')
args = parser.parse_args()

if __name__ == "__main__":
    '''
    test conventional blindly image quality assessment methods(brisque/niqe/piqe)
    '''
    mode = args.mode
    path = args.path
    im = cv2.imread(path)
    if im is None:
        print("please input correct image path!")
        sys.exit(0)
    score = niqe(im)
    print("{}-----{} score:{}".format(path, mode, score))