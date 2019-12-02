import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs    # 总的epoch数量
        self.batches_epoch = batches_epoch     # 一个epoch中batch的数量，这里为dataloader中最大文件夹中文件的值
        self.epoch = 1              # 表示现在迭代到了第几个epoch
        self.batch = 1              # 用来记录迭代到第几个batch
        self.prev_time = time.time()    # 用来记录迭代开始时刻，便于做差记录一次迭代花了多少时间
        self.mean_period = 0        # 用来记录累计已经花了多少时间
        self.losses = {}            # 新建立一个字典，用来保存计算的loss
        self.loss_windows = {}      
        self.image_windows = {}     # 新建一个字典，用来保存图片：real_A | real_B | fake_A | fake_B


    def log(self, losses=None, images=None):
        # 记录累计已经花了多少时间
        self.mean_period += (time.time() - self.prev_time)
        # 用来记录迭代开始时刻时间，便于以后做差记录记录一次迭代花了多少时间
        self.prev_time = time.time()

        # 向控制台输出信息
        # 基本和print是等价，没有print'\n'自动换行的功能，使得每一次迭代计算信息看起来更加直观
        # 输出形式举例：Epoch 001/200 [0083/1074]
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            # self.losses字典中没有loss_name的时候，对其进行相应的赋值
            if loss_name not in self.losses:

                # 元素张量可以通过item得到元素值image_windows
                # 例如从tensor(0.4279)  变成    0.4278833866119385
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            #  当self.losses全部赋值完毕之后，进行相应的打印
            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            # 打印对应的loss_name的loss，除以batch计算的是平均loss
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        # 总的迭代完成的batch的累计，而不是一个batches_epoch中迭代完成的batch的累计
        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        # 还剩下多少个batch没有迭代
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        # 计算全部训练完成剩余时间
        # mean_period是已经花了多少时间，batches_done是已经迭代了多少次，相除表示一次迭代花的时间
        # 输出形式举例：ETA: 2 days, 17:26:54.824266
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # 画图
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                # 在visdom窗口中，对图片进行显示和更新
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # 每个batch结束的时候(batch默认就是1)
        if (self.batch % self.batches_epoch) == 0:
            # 打印损失
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # 为下一次epoch迭代重置损失
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            # 每个batch计算结束的时候，向控制台输出信息换行
            sys.stdout.write('\n')
        else:
            # 一个batch只有一个数据，迭代到下一个batch就+1
            self.batch += 1

        

class ReplayBuffer():
    """
    定义缓冲区，存取之前生成的图片

    这个缓冲区允许我们使用生成的历史图片来更新判别器
    而不只是利用最近生成的图片
    """
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        返回缓冲区中的图像

        最后返回的形式是：
        有50%的数据是输入缓冲区的图片，然后返回
        有另外50%的数据是缓冲区之前已经存储的数据，然后向之前缓冲区中提取数据的位置插入新的数据
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)

            # 如果缓冲区未满，继续添加
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                # 有50%的几率，缓冲区将会返回之前存储的图片，并且向缓冲区中插入最新图片
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element

                # 有另外50%的几率，缓冲区将会返回现在的图片
                else:
                    to_return.append(element)
        # 收集所有图片并且返回
        return Variable(torch.cat(to_return))

class LambdaLR():
    """
    定义学习率调整策略
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """
        学习率调整方法参数初始化

        参数：
            n_epochs                    --总epoch数量
            offset                      --补偿
            decay_start_epoch           --从哪一个epoch开始，学习开始衰减
        """
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# 定义生成器和判别器网络初始化的形式
def weights_init_normal(m):
    classname = m.__class__.__name__  # 获取类名称
    # 如果网络有卷积层，进行对应初始化
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)   # 对weight参数使用normal初始化方法进行初始化，mean=0, std=0.02
    # 如果网络有BatchNorm2d，进行对应初始化
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)         # 对bias使用constant初始化方法，全部初始化为0

