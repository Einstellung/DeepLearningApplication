import argparse
import os

import torch
import torchvision.transforms as transforms
from glob import glob

from utils import load_img, save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='test_picture',help='root directory of the dataset')
parser.add_argument('--model', type=str, default='output/netG_model_epoch_200.pth', help='generator checkpoint file')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = opt.model
image_dir = opt.dataset

net_g = torch.load(model_path).to(device)



image_filenames = sorted(glob(image_dir + '/*.*'))

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for i, batch in enumerate(image_filenames):
    img = load_img(batch)
    
    # 转成黑白图片
    white_black = img.convert('L')
    img = transform(white_black)

    input_picture = torch.cat((img, img, img), dim=0)

    input_picture = input_picture.unsqueeze(0).to(device)

    

    out = net_g(input_picture)
    out_img = out.detach().squeeze(0).cpu()


    save_img(out_img, "output/generate/%04d.png" % (i+1))
    white_black.save("output/blank/%04d.png" % (i+1))
