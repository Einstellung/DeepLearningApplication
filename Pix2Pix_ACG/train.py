import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from dataset import DatasetFromFolder
from utils import Logger

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataroot', type=str, default='datasets/anime_resized/', help='root directory of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--white_to_color', type=bool, default=True, help='change white to color mode or not')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率
# 前提是网络输入数据类型和维度变化不大
cudnn.benchmark = True


print('===> Loading datasets')

# 加载数据集
data_loader = DataLoader(dataset=DatasetFromFolder(opt.dataroot, opt.direction, opt.white_to_color), 
                        num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

# 查看是否使用GPU
device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
# 加载生成器和判别器
net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

# 设置损失函数
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)


# 设置优化方法
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# 设置学习率调整策略
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

# 打印日志，初始化
logger = Logger(opt.niter + opt.niter_decay, len(data_loader))



###### 训练 ######
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    for iteration, batch in enumerate(data_loader, 1):

        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) 更新判别器网络
        ######################

        optimizer_d.zero_grad()

        """
        在使用conditional GAN的时woso
        需要将真实图片和生成图片一woso
        """

        # 判别器对虚假数据进行训练woso
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # 判别器对真实数据进行训练
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # 判别器损失
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) 更新生成器网络
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        
        # 生成器损失
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizer_g.step()



        # 记录计算的日志 (http://localhost:8097)
        logger.log(losses={'loss_d': loss_d, 'loss_g_gan': loss_g_gan, 'loss_g_l1': loss_g_l1,
                    'loss_g': loss_g}, 
                    images={'real_A': real_a, 'real_B': real_b, 'fake_image': fake_b})

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    #checkpoint
    if epoch % 10 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        dataroot = opt.dataroot.split(os.sep)[1]
        if not os.path.exists(os.path.join("checkpoint", dataroot)):
            os.mkdir(os.path.join("checkpoint", dataroot))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataroot, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(dataroot, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + dataroot))
