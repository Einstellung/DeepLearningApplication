from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from glob import glob

class DatasetFromFolder(data.Dataset):
    """
    定义从文件夹中如何导入数据
        数据格式为：
        ├── datasets                   
        |   ├── <dataset_name>         
        |   |   ├── train              # Training
        |   |   |   ├── A              # Contains domain A images 
        |   |   |   └── B              # Contains domain B images 
        |   |   └── test               # Testing
        |   |   |   ├── A              # Contains domain A images 
        |   |   |   └── B              # Contains domain B images
    注意，如果是给图片上色的话（黑白图片变成彩色）不需要B文件夹， 即我们定义的B文件是转成的黑白图像
    A文件夹默认为彩色图像
    """
    def __init__(self, image_dir, direction, white_to_color=False):
        """
        对如何进行数据处理进行初始化

        参数：
            image_dir               --数据文件根目录位置（即train文件夹）所在目录位置
            direction               --转换方向，是A ——> B 还是 B ——> A
            white_to_color          --是否使用给黑白图片上色模式，默认为False
        """
        # super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.white_to_color = white_to_color
        # 获取A文件夹中的文件列表
        self.files_A = sorted(glob(join(image_dir, '%s/A' % "train") + '/*.*'))

        # 如果不使用给黑白图片上色模式，获取B文件夹中的文件列表
        # 如果采用给黑白图片上色的模式，因为没有B文件夹，所以不需要获取文件夹中的文件内容了
        if not self.white_to_color:
            self.files_B = sorted(glob(join(image_dir, '%s/B' % "train") + '/*.*'))


    def __getitem__(self, index):
        """
        返回索引为index的图片数据

        obj[index]等价于obj.__getitem__(index)is_image_file
        """
        a = Image.open(self.files_A[index])

        # 查看是否是黑白图片转彩色图片任务
        # 如果不是的化，正常加载图片
        # 如果是的话，将A文件夹中的彩色图片转为灰度图
        if not self.white_to_color:
            b = Image.open(self.files_B[index])
        else:
            b = a.convert('L')
        
        # pix2pix论文中，对输入图片先扩展为286*286
        # 然后再随机裁剪为256*256
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)

        # 生成器输入维度为3，convert转换之后的灰度图像是1维
        # 所以使用concatenate进行维度扩展，复制3个相同的b，扩展为3维，依旧是灰度图像
        if self.white_to_color:
            b = torch.cat((b, b, b), dim=0)

        w_offset = random.randint(0, max(0, 286 - 256 - 1))     # 宽度方向的随机裁剪初始值
        h_offset = random.randint(0, max(0, 286 - 256 - 1))     # 长度方向的随机裁剪初始值
    
        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]  # 随机裁剪为256*256
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # 转换方向
        # a2b表示的是A ——> B
        # b2a表示的是B ——> A
        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        """
        用来表示A文件夹有多少文件
        """
        return len(self.files_A)
