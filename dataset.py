import os
from glob import glob

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(400, 400)):
        super(StegaData, self).__init__()
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        '''
        # glob.glob(data_path/*.jpg)获取data_path路径下的所有jpg文件并形成一个列表
        # os.path.join(path1,path2...)->从倒数第一个，以‘/’开头的参数开始拼接，之前的参数全部丢弃。   
        eg: os.path.join('/11/','/11/22.jpg')
            Out[18]: '/11/22.jpg'
            os.path.join('/11/','11/22.jpg')
            Out[19]: '/11/11/22.jpg'
        '''
        self.files_list = glob(os.path.join(self.data_path, '*'))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]
        try:
            img_cover = Image.open(img_cover_path).convert('RGB')
            # ImageOps.fit()图像被裁剪到指定的宽高比和尺寸
            img_cover = ImageOps.fit(img_cover, self.size)
            img_cover = self.to_tensor(img_cover)
            # transforms.ToTensor()可以将PIL图片转为c*h*w格式并从[0,255]->[0,1]
            # img_cover = np.array(img_cover, dtype=np.float32) / 255.
        except:
            img_cover = np.zeros((3, self.size[0], self.size[1]), dtype=np.float32)
            img_cover = torch.from_numpy(img_cover)
        # np.random.binomial()二项式分布函数: n=1时，重复伯努利试验。
        # 相当于一次抛一枚硬币试验,正面朝上发生的概率为0.5,做100次实验,求每次试验发生正面朝上的硬币个数：
        secret = np.random.binomial(1, 0.5, self.secret_size)
        secret = torch.from_numpy(secret).float()

        return img_cover, secret

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':
    # dataset = StegaData(data_path='F:\\VOCdevkit\\VOC2012\\JPEGImages')
    # print(len(dataset))
    # img_cover, secret = dataset[10]
    # print(type(img_cover), type(secret))
    # print(img_cover.shape, secret.shape)

    dataset = StegaData(data_path='examples/asserts', secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))
    print(type(image_input), type(secret_input))
    print( secret_input)
