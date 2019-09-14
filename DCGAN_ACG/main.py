import torch as t 
import torchvision as tv
from torchnet.meter import AverageValueMeter
from model import NetG, NetD
import tqdm

class Config(object):
    data_path = r"./data" # 数据集存放路径
    num_workers = 4 # 多进程加载数据所用的进程数
    image_size = 96 # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4 # 生成器学习率
    lr2 = 2e-4 # 判别器学习率
    beta1 = 0.5 # Adam优化器的beta1参数
    gpu = True
    nz = 100 # 噪声维度
    ngf = 64 # 生成器feature map数
    ndf = 64 # 判别器feature map数

    save_path = r"./imgs"

    plot_every = 20 # 每间隔20 batch， visdom画图一次

    debug_file = r"./tmp"  # 存储debug的数据
    d_every = 1 # 每1个batch训练一次判别器
    g_every = 5 # 每5个batch训练一次生成器
    save_every = 10 # 每10个batch保存一次模型
    netd_path = None # "checkpoints/netd._pth" # 预训练模型
    netg_path = None

    # 只测试不训练
    gen_img = "result.png"
    # 从512张生成图片中，保存最好的64张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0 # 噪声的均值
    gen_std = 1 # 噪声的方差

opt = Config()

def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device("cuda") if opt.gpu else t.device("cpu")

    # 数据处理，输出规范为-1~1
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset, 
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        drop_last=True)

    # 网络
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss()

    # 真图片label为1，假图片label为0， noise为生成网络的输入
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # 用来结果的均值和标准差
    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能把真图片判别为正
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## 尽可能把假图片判断为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                # 使用detach来关闭G求梯度，加速训练
                fake_img = netg(noises).detach()
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())


            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                # 尽可能把假的图片也判别为1
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            # 可视化

        
        # 保存模型、图片
        if (epoch+1) % opt.save_every == 0:
            fix_fake_imgs = netg(fix_noises)
            tv.utils.save_image(fix_fake_imgs.data[:64], "%s%s.png" % (opt.save_path, epoch), normalize=True, range=(-1, 1))
            t.save(netd.state_dict(), r"./checkpoints/netd_%s.pth" % epoch)
            t.save(netg.state_dict(), r"./checkpoints/netg_%s.pth" % epoch)
            errord_meter.reset()
            errorg_meter.reset()

if __name__ == "__main__":
    import fire
    fire.Fire()