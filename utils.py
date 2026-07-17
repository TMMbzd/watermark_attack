import itertools
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:1')


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur  # 7
    '''
    np.meshgrid()——生成网格点坐标矩阵
    
    a = np.array([1,2,3])
    b = np.array([7,8])
    # 从坐标向量中返回坐标矩阵
    # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
    res = np.meshgrid(a,b)
    # 返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]
    '''
    coords = torch.from_numpy(np.stack(np.meshgrid(range(N_blur), range(N_blur), indexing='ij'), axis=-1)) - (
            0.5 * (N - 1))  # （7,7,2)    [0,6]->[-3,3]
    manhat = torch.sum(torch.abs(coords), dim=-1)  # (7, 7)
    # 计算平均值的时候，中心点就是原点，所以μ等于0
    """ manhat:
    tensor([[6., 5., 4., 3., 4., 5., 6.],
            [5., 4., 3., 2., 3., 4., 5.],
            [4., 3., 2., 1., 2., 3., 4.],
            [3., 2., 1., 0., 1., 2., 3.],
            [4., 3., 2., 1., 2., 3., 4.],
            [5., 4., 3., 2., 3., 4., 5.],
            [6., 5., 4., 3., 4., 5., 6.]])
    """
    # nothing, default
    vals_nothing = (manhat < 0.5).float()  # (7, 7)
    """ vals_nothing:
     ([[0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.]])
    """

    # gauss
    # 标准差在1到3个之间随机采样 sigrange_gauss=[1., 3.]
    sig_gauss = torch.rand(1)[0] * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords ** 2, dim=-1) / 2. / sig_gauss ** 2)

    # line
    theta = torch.rand(1)[0] * 2. * np.pi  # 对一个随机角度进行采样，2Π即360度
    v = torch.FloatTensor([torch.cos(theta), torch.sin(theta)])  # 2维标量,如tensor([ 0.1974, -0.9803])
    dists = torch.sum(coords * v, dim=-1)  # (7, 7) --> xcosθ+ysinθ

    # sigrange_line=[.25, 1.]-->   sig_line∈（0.25，1）
    sig_line = torch.rand(1)[0] * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    # wmin_line=3 --> w_line∈(3,3.1)
    w_line = torch.rand(1)[0] * (0.5 * (N - 1) + 0.1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists ** 2 / 2. / sig_line ** 2) * (manhat < w_line)  # (7, 7)

    t = torch.rand(1)[0]
    vals = vals_nothing
    if t < (probs[0] + probs[1]):  # probs=[.25, .25]
        vals = vals_line
    else:
        vals = vals
    if t < probs[0]:
        vals = vals_gauss
    else:
        vals = vals

    v = vals / torch.sum(vals)  # 归一化 (7, 7)
    z = torch.zeros_like(v)
    f = torch.stack([v, z, z, z, v, z, z, z, v], dim=0).reshape([3, 3, N, N])
    return f


def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 3, 3))
    for i in range(batch_size):
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right
        # 均匀地随机扰动标记的四个角点位置
        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size]], dtype="float32")
        # 目标图像
        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype="float32")
        # 获得投射变化后的H矩阵
        # 参数说明：rect表示原始的位置左上，左下，右下，右下，
        #         dst表示变换后四个角的位置
        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)  # 矩阵求逆
        Ms[i, 0, :, :] = M_inv
        Ms[i, 1, :, :] = M
    Ms = torch.from_numpy(Ms).float()

    return Ms


def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    # 为从[−0.1，0.1]均匀采样的每个rgb通道添加随机颜色偏移量。
    rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)

    rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)  # [−0.3，0.3]均匀采样
    return rnd_hue + rnd_brightness


# reference: https://github.com/mlomnitz/DiffJPEG.git
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))

if torch.cuda.is_available():
    y_table = y_table.to(device)
    c_table = c_table.to(device)


# 1. RGB -> YCbCr
class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)  # 转为batch*h*w*c的格式
        # a = torch.tensor([[1,2],[3,4]])
        # b = torch.tensor([[5,6],[7,8]])
        # c = torch.tensordot(a, b, dims=1)-->tensor([[19, 22],[43, 50]]) 矩阵乘法
        # 想表示内积，直接令dims = 1即可
        # 如果dims=0则按照逐元素挨个相乘,相当于求外积,即1*5,6,7,8，2*5,6,7,8 组成（2，2，2，2)维的矩阵
        # dims可以为二维数组，(dims_a, dims_b)，指定两个张量任意维度相乘。
        # 如dims = ([0],[1])，结果是tensor([[23, 31],[34, 46]])
        # 如果不写dims则是两个矩阵的点积求和，结果是tensor(70)
        # 图片是b*400*400*3的,按dims=1正好是400*3,即原图一行对应的RGB值
        """
        dims出现在哪几维,对应维取 ：
        dims是int
        1.dims=1,说明a取最后一维(即a[0,:])与b取第一维(即b[:,0])二者点积后求和
          也就是[1,2]与[5,7]点积得[5,14]再求和,得到c[0,0]的结果19
        2.同理dims=2相乘指的是将a的最后两个维度与b的前两个维度矩阵相乘然后将结果累加求和，消除（收缩）这四个维度，矩阵a，b剩下的维度concat，就是所求矩阵维度
        3.dims=0,即a[0,0]标量
        a = tf.ones(shape=[2,3,3])
        b = tf.ones(shape=[3,2,6])
        c = tf.tensordot(a,b, axes=1)
        c的形状为[2,3,2,6],这里axes=1,说明取a的后1维即[3]和b的前1维即[3]进行矩阵相乘,其他维不变,那么根据矩阵乘法,自然得到c的大小为[2,3,2,6]
        
        dims是list或tuple
        A = np.random.randint(0,9,(3,4,5))
        B = np.random.randint(0,9,(4,5,2))
        np.tensordot(A, B, [(1,2), (0,1)])
        (1,2) 是对A而言，不是取第1，2轴，而是除去1,2 轴，所以要取的是第0轴  A[0,:,:]
        (0,1) 是对B而言，不是取第0，1轴，而是除去0,1 轴，所以要取的是第2轴  B[:,:,0]
        A的形状是(3,4,5)，第0轴上有3个元素；B的形状(4,5,2)，第2轴上有2个元素，所以结果形状是(3,2)
        """

        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        result.view(image.shape)
        return result  # b*h*w*3


# 2. Chroma subsampling 色度采样 4：1：1
class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()  # 转为b*3*h*w
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        # image_2[:, 1, :, :]后是[b,400,400]需要unsqueeze(1)进行维度扩充成b*1*h*w
        # 对cbcr进行压缩采样
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))  # b*1*h*w
        cb = cb.permute(0, 2, 3, 1)  # b*h*w*1
        cr = cr.permute(0, 2, 3, 1)  # b*h*w*1
        # image本来是b*3*h*w的,经过image[:, :, :, 0]后只取通道是0的那一层,变成(b*h*w)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)  # 返回y(b*h*w),cb与cr大小都是(b*h/2*w/2)


# 3. Block splitting
class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]  # 假设输入是y分量,即b*400*400大小
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)  # 大小是[1, 50, 8, 50, 8]
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)  # 大小是[1, 50, 50, 8, 8]
        # contiguous：view只能用在contiguous的variable上。
        # 如果在view之前用了transpose,permute等,需要用contiguous()来返回一个contiguous copy
        # 假设是b*400*400的y分量图，能分成2500个8*8大小的块
        # 若是b*200*200的cb和cr分量，能分成625个8*8大小的块
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)  # 大小[1, 2500, 8, 8]


# 4. DCT
class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width   (我觉得是batch x h*w/64 x 8 x 8四维的)
    Output:
        dcp(tensor): batch x height x width
    """

    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        # product 用于求多个可迭代对象的笛卡尔积(Cartesian Product)，它跟嵌套的 for 循环等价.即:
        # product(A, B) 和 ((x,y) for x in A for y in B)的效果是一样的。
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)

        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        '''
        result = numpy.outer(a, b) 用来求外积的，非常直观，比矩阵相乘简单
        1. a，b是两个数组，如果a，b是高维数组，函数会自动将其flatten成1维
        2. a的长度是m，b的长度是n，外积的结果result是 m * n的数组,数组元素为 result[i,j] = a[i] * b[j]
        np.outer([1,2,3],[4,5,6])
        np.outer([[1],[2],[3]],[4,5,6])
        上面两句运行结果都是
        array([[ 4,  5,  6],
               [ 8, 10, 12],
               [12, 15, 18]])
        '''
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        # 由于DCT变化需要定义域对称,在做DCT变化之前要先将矩阵中的数值左移 128,使得值域落在-128~127
        image = image - 128
        # 输入image的后两维8*8的f(x,y)正好与DCT矩阵的x,y相对应
        # 实际上就是割出来的8*8的patch与对应位置的dct矩阵点乘后求和
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


# 5. Quantization
class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding=None, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding=None, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image * 255)  # 从[0,1]-->[0,255]
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


# -5. Dequantization
class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """

    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width
    """

    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


# -4. Inverse DCT
class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


# -3. Block joining
class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


# -2. Chroma upsampling
class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input: 
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


# -1: YCbCr -> RGB
class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)  # batch*c*h*w


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """

    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()

        self.height, self.width = height, width

    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height / 2), int(self.width / 2)
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)
        # 保证图片的灰度级在[0,255]之间
        image = torch.min(255 * torch.ones_like(image), torch.max(torch.zeros_like(image), image))
        return image / 255  # [0,255]-->[0,1]


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


def round_only_at_0(x):
    cond = (torch.abs(x) < 0.5).float()
    return cond * (x ** 3) + (1 - cond) * x


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """

    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


def jpeg_compress_decompress(image,
                             #  downsample_c=True,
                             rounding=round_only_at_0,
                             quality=80,
                             args=None):
    # image_r = image * 255
    height, width = image.shape[2:4]
    # orig_height, orig_width = height, width
    # if height % 16 != 0 or width % 16 != 0:
    #     # Round up to next multiple of 16
    #     height = ((height - 1) // 16 + 1) * 16
    #     width = ((width - 1) // 16 + 1) * 16

    #     vpad = height - orig_height
    #     wpad = width - orig_width
    #     top = vpad // 2
    #     bottom = vpad - top
    #     left = wpad // 2
    #     right = wpad - left
    # #image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
    # image = torch.pad(image, [[0, 0], [0, vpad], [0, wpad], [0, 0]], 'reflect')

    if torch.cuda.is_available():
        rgb_to_ycbcr_jpeg().to(device)
        chroma_subsampling().to(device)
        block_merging().to(device)
        dct_8x8().to(device)
        y_quantize().to(device)
        c_quantize().to(device)
        y_dequantize().to(device)
        c_dequantize().to(device)
        idct_8x8().to(device)
        block_splitting().to(device)
        chroma_upsampling().to(device)
        ycbcr_to_rgb_jpeg().to(device)

    factor = quality_to_factor(quality)

    compress = compress_jpeg(rounding=rounding, factor=factor).to(device)
    decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor).to(device)

    y, cb, cr = compress(image)
    recovered = decompress(y, cb, cr)

    return recovered


if __name__ == '__main__':
    ''' test JPEG compress and decompress'''
    img = Image.open('examples/asserts/1.png')
    img = np.array(img) / 255.
    img_r = np.transpose(img, [2, 0, 1])
    img_tensor = torch.from_numpy(img_r).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    recover = jpeg_compress_decompress(img_tensor).cpu()

    recover_arr = recover.detach().squeeze(0).numpy()
    recover_arr = np.transpose(recover_arr, [1, 2, 0])

    print(img_tensor.shape, recover.shape)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(recover_arr)
    plt.show()

    """test blur"""
    # blur
    # import torch.nn.functional as F
    #
    # image = Image.open('examples/asserts/1.png')
    # image = np.array(image) / 255.
    # img_r = np.transpose(image, [2, 0, 1])
    # img_tensor = torch.from_numpy(img_r).unsqueeze(0).float()
    # print(img_tensor.shape)
    #
    # N_blur = 7
    # f = random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
    #                        wmin_line=3)
    # # print(f.shape)
    # # print(type(f))
    # encoded_image = F.conv2d(img_tensor, f, bias=None, padding=int((N_blur - 1) / 2))
    #
    # print(encoded_image.shape)
    #
    # encoded_image = encoded_image.detach().squeeze(0).numpy()
    # encoded_image = np.transpose(encoded_image, [1, 2, 0])
    #
    # plt.subplot(121)
    # plt.imshow(image)
    # plt.subplot(122)
    # plt.imshow(encoded_image)
    # plt.show()
    #
    # import torch
