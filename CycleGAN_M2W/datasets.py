import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    """
    数据预处理，用于Dataloader加载数据集
    数据格式为：
        ├── datasets                   
        |   ├── <dataset_name>         
        |   |   ├── train              # Training
        |   |   |   ├── A              # Contains domain A images 
        |   |   |   └── B              # Contains domain B images 
        |   |   └── test               # Testing
        |   |   |   ├── A              # Contains domain A images 
        |   |   |   └── B              # Contains domain B images 
    """
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        """
        对如何数据预处理进行初始化
        参数：
            root                    --数据文件根目录
            transforms_             --定义如何对数据进行转换
            unaligned               --是否采取无序随机提取图片的方式，当为False时，采取有序模式，按顺序提取图片
            mode                    --模式选择，是train还是test
        """
        # 定义数据转换形式
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        """
        返回索引为index的图片数据

        obj[index]等价于obj.__getitem__(index)
        """
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        # 是否对B文件夹的数据采取unaligned模式
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        # 以字典的形式返回采集到的数据
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))