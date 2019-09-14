## 项目介绍

本项目使用DCGAN来生成二次元人物头像，使用的是PyTorch，版本为0.4.1，除此之外，不需要其他依赖环境。另外，本项目使用fire工具来输入命令参数，因此要运行本项目的时候，如果没有安装fire，可以通过

```python
pip install fire
```

来安装。

二次元人物头像数据集，来自[知乎](https://zhuanlan.zhihu.com/p/24767059)大佬做好和爬取的数据集，文章上面有下载链接。下载好数据之后，请保存到faces文件夹下面，目录结构如下所示

```
data/
└── faces/
    ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
    ├── 0001a0fca4e9d2193afea712421693be-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-0.jpg
    ├── 0001d9ed32d932d298e1ff9cc5b7a2ab-1.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-0.jpg
    ├── 00028d3882ec183e0f55ff29827527d3-1.jpg
    ├── 000333906d04217408bb0d501f298448-0.jpg
    ├── 0005027ac1dcc32835a37be806f226cb-0.jpg
```

DCGAN的原理网上有很多，这里就不再展开说了。

## 文件夹介绍

data文件夹下，用来存储数据集。checkpoints文件夹用来存储D和G的模型。每训练一段时间就保存一下。便于回溯训练效果最好的模型。picture文件夹是保存我已经训练过生成的图片。

该项目实际训练主要由两个文件组成，`model.py` 用来定义DCGAN网络，`main.py`用来设置模型的训练过程，以及配置config参数等。

## 模型训练

```bash
python main.py train --gpu --vis=False
```

当训练第10次的时候，效果如下：

![](https://github.com/Einstellung/DeepLearningApplication/blob/master/DCGAN_ACG/pciture/imgs9.png)

当训练到第200次的时候

![](https://github.com/Einstellung/DeepLearningApplication/blob/master/DCGAN_ACG/pciture/imgs199.png)

看起来效果要明显好很多了。

本代码参考自[GitHub](https://github.com/chenyuntc/pytorch-book/tree/master/chapter7-GAN%E7%94%9F%E6%88%90%E5%8A%A8%E6%BC%AB%E5%A4%B4%E5%83%8F)的实现，我在学习研究这个代码过程中，最大的疑惑是在模型定义部分，为什么在判别器部分要有detach，而在生成器部分没有写detach，后来经过思考和查阅资料解决了这个问题，如果你也有类似的疑惑，可以点击这个[链接](https://github.com/chenyuntc/pytorch-book/issues/173)，我有提供一个解答思路。