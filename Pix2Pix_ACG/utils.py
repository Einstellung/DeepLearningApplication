import numpy as np
from PIL import Image
from visdom import Visdom
import time
import datetime
import sys


def load_img(filepath):
    img = Image.open(filepath)
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))



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
        # 迭代200个batch之后画一次图
        if batches_done % 200 == 0:
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