import torch.nn as nn
import torch.nn.functional as F

# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        # 残差模块不改变shape
        conv_block = [  nn.ReflectionPad2d(1),  # 构建残差模块的时候使用映射填充的形式
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),     # 不使用BatchNorm而是使用InstanceNorm
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        """
        定义生成网络

        参数：
            input_nc                    --输入通道数
            output_nc                   --输出通道数
            n_residual_blocks           --残差模块数量
        """
        super(Generator, self).__init__()

        # 初始化卷积模块
        # 因为使用ReflectionPad扩充
        # 所以输入是3*256*256
        # 输出是64*256*256
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # 进行下采样
        # 第一个range：输入是64*256*256，输出是128*128*128
        # 第二个range：输入是128*128*128，输出是256*64*64

        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # 使用残差模块
        # 输入输出都是256*64*64
        for _ in range(n_residual_blocks): # 默认添加9个残差模块
            model += [ResidualBlock(in_features)]

        # 进行上采样
        # 第一个range：输入是256*64*64，输出是128*128*128
        # 第二个range：输入是128*128*128，输出是64*256*256       
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # 最后输出层
        # 输入是64*256*256
        # 输出是3*256*256
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # 构建卷积分类器
        # 输入为3*256*256
        # 输出为64*128*128
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # 输入为64*128*128
        # 输出为128*64*64
        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]
 
        # 输入为128*64*64
        # 输出为256*32*32
        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # 输入为256*32*32
        # 输出为512*31*31
        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # 全卷积分类层
        # 输入为输出为512*31*31
        # 输出为1*30*30
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # 使用平均池化的办法输出预测值
        # avg_pool2d(input,kernel_size），这里kernel_size为30
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)