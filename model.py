import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import lpips
    import torchgeometry
    from kornia import color
    import tools
    import utils
except ImportError:
    # Encoder/decoder inference only needs torch. The optional packages above are
    # required by the StegaStamp training loss and augmentation pipeline.
    lpips = None
    torchgeometry = None
    color = None
    tools = None
    utils = None

# import sys
# sys.path.append("PerceptualSimilarity\\")
# from PerceptualSimilarity.lpips_torch import lpips
"""
import lpips
loss_fn_alex = lpips.LPIPS(net='alex')  —> best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') —> closer to "traditional" perceptual loss, when used for optimization

import torch
img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)
"""


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            # pytorch默认使用kaiming均匀分布初始化神经网络参数,此处用了正态分布。
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class StegaStampEncoder(nn.Module):
    def __init__(self,args):
        super(StegaStampEncoder, self).__init__()
        self.secret_dense = Dense(args.secret_size, 7500, activation='relu', kernel_initializer='he_normal')  # 秘密消息全连接层

        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, inputs):
        secrect, image = inputs
        secrect = secrect - .5
        image = image - .5
        secrect = self.secret_dense(secrect)  # [100,]-->[7500,]
        secrect = secrect.reshape(-1, 3, 50, 50)  # [b,3,50,50]
        # 上采样成3*400*400的tensor
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect)
        # 秘密消息和载体图像在通道维拼接成6*400*400的tensor
        inputs = torch.cat([secrect_enlarged, image], dim=1)
        # U-Net下采样   输入为6*400*400
        conv1 = self.conv1(inputs)  # 输出为32*400*400
        conv2 = self.conv2(conv1)  # 输出为32*200*200
        conv3 = self.conv3(conv2)  # 输出为64*100*100
        conv4 = self.conv4(conv3)  # 输出为128*50*50
        conv5 = self.conv5(conv4)  # 输出为256*25*25
        # U-Net上采样
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5))  # 先上采样再卷积为128*50*50
        merge6 = torch.cat([conv4, up6], dim=1)  # 与conv4 concat为256*50*50
        conv6 = self.conv6(merge6)  # 卷积输出为128*50*50

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))  # 先上采样再卷积为64*100*100
        merge7 = torch.cat([conv3, up7], dim=1)  # 与conv3 concat为128*100*100
        conv7 = self.conv7(merge7)  # 卷积输出为64*100*100

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7))  # 先上采样再卷积为32*200*200
        merge8 = torch.cat([conv2, up8], dim=1)  # 与conv2 concat为64*200*200
        conv8 = self.conv8(merge8)  # 卷积输出为32*200*200

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))  # 先上采样再卷积为32*400*400
        merge9 = torch.cat([conv1, up9, inputs], dim=1)  # 与conv1和inputs concat为70*400*400
        conv9 = self.conv9(merge9)  # 卷积输出为32*400*400

        residual = self.residual(conv9)  # 1*1卷积输出为3*400*400,得到三通道的残差图
        return residual


# class Flatten(nn.Module):
#     def __init__(self):
#         super(Flatten, self).__init__()
#
#     def forward(self, input):
#         return input.view(input.size(0), -1)


class StegaStampDecoder(nn.Module):
    def __init__(self, args):
        super(StegaStampDecoder, self).__init__()
        self.secret_size = args.secret_size

        self.decoder = nn.Sequential(
            # # 输入400*400大小的图片
            Conv2D(3, 32, 3, strides=2, activation='relu'),  # 200* 200
            Conv2D(32, 32, 3, activation='relu'),  # 200
            Conv2D(32, 64, 3, strides=2, activation='relu'),  # 100 * 100
            Conv2D(64, 64, 3, activation='relu'),  # 100 * 100
            Conv2D(64, 64, 3, strides=2, activation='relu'),  # 50 * 50
            Conv2D(64, 128, 3, strides=2, activation='relu'),  # 25 * 25
            Conv2D(128, 128, 3, strides=2, activation='relu'),  # 13 * 13
            nn.Flatten(),  # 128*13*13 = 21632
            Dense(21632, 512, activation='relu'),
            Dense(512, args.secret_size, activation=None))

        # Spatial transformer localization-network
        # 从输入图像中提取特征
        # 输入图片的shape为(-1,3,400,400)
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 200* 200
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 100*100
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 50*50
            nn.ReLU(True),
            nn.Flatten()  # 128*50*50 = 320,000
        )
        # Regressor for the 3 * 2 affine matrix
        # 利用全连接层回归theta参数
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 50 * 50, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

        # 将权重初始化为 identity 变换
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # affine_grid里的theta应该是3维的张量，size应该是四维张量
        # 根据变换矩阵来计算变换后图片的对应位置
        grid = F.affine_grid(theta, x.size())
        # 默认使用双向性插值，可以通过mode参数设置
        x = F.grid_sample(x, grid)
        return x

    def forward(self, image):
        image = image - .5
        # transform the input
        image = self.stn(image)
        return torch.sigmoid(self.decoder(image))


class Discriminator(nn.Module):  # Critic网络
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2D(3, 8, 3, strides=2, activation='relu'),  # 8*200*200
            Conv2D(8, 16, 3, strides=2, activation='relu'),  # 16*100*100
            Conv2D(16, 32, 3, strides=2, activation='relu'),  # 32*50*50
            Conv2D(32, 64, 3, strides=2, activation='relu'),  # 64*25*25
            Conv2D(64, 1, 3, activation=None))  # 1*25*25

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        # torch.mean(input, dim, keepdim=False, *, out=None) → Tensor
        output = torch.mean(x)
        return output, x


def transform_net(encoded_image, args, global_step, writer=None):
    if utils is None:
        raise ImportError("transform_net requires the optional StegaStamp training utilities.")
    gpu_choiced = 1
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])  # [1/1000，1]

    # 在[50,100]范围内均匀采样JPEG质量
    jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)

    # 采用高斯噪声模型（对标准偏差σ〜U [0,0.2]进行采样）来解决成像噪声。
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise  # rnd_noise:0.02

    # 亮度和对比度: 用m∼U[0.5，1.5]和b∼U[−0.3，0.3]对mx+b进行仿射直方图重新缩放。
    # 色度偏移：为从[−0.1，0.1]均匀采样的每个rgb通道添加随机颜色偏移量。
    # 亮度和色度：
    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri  # rnd_bri:0.3  亮度
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue  # rnd_hue:0.1  色调/色度
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)  # [batch_size, 3, 1, 1]
    # 对比度:
    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)  # contrast_low:0.5
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)  # contrast_high:1.5
    contrast_params = [contrast_low, contrast_high]  # [0.5,1.5]

    # 去饱和度：在完整的RGB图像和其灰度等效图像之间随机线性内插.此处是获得比例因子，从[0,1]均匀采样
    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat  # rnd_sat: (0,1.0)

    # blur
    N_blur = 7
    f = utils.random_blur_kernel(probs=[.25, .25], N_blur=7, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                 wmin_line=3)
    if args.cuda:
        f = f.cuda(gpu_choiced)
    # nn.Conved是2D卷积层，而F.conv2d是2D卷积操作
    # print(encoded_image.device)
    # print(f.device)
    encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

    # noise：采用高斯噪声模型（对标准偏差σ〜U [0,0.2]进行采样）来解决成像噪声。
    noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32)
    if args.cuda:
        noise = noise.cuda(gpu_choiced)
    encoded_image = encoded_image + noise  # 实际上也就是原图+噪音
    """
    torch.clamp(input, min, max, out=None) → Tensor
    将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量。
    也即大于max的用max代替,小于min的用min代替
    """
    encoded_image = torch.clamp(encoded_image, 0, 1)  # 为了保持pytorch处理图像所需要的[0,1]区间

    # contrast & brightness：用m∼U[0.5，1.5]和b∼U[−0.3，0.3]对mx+b进行仿射直方图重新缩放。
    # 此处是缩放因子的获取
    contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0],
                                                                    contrast_params[1])  # 从[0.5,1.5]均匀采样
    contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    if args.cuda:
        contrast_scale = contrast_scale.cuda(gpu_choiced)
        rnd_brightness = rnd_brightness.cuda(gpu_choiced)
    # 对mx+b进行仿射直方图重新缩放。
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    # 去饱和度：在完整的RGB图像和其灰度等效图像之间随机线性内插。rnd_sat：（0，1）
    # 在颜色通道上进行加权求平均得到完整RGB图像对应的灰度等效图像,因求平均会降维,故在颜色通道增维
    sat_weight = torch.FloatTensor([0.3, 0.6, 0.1]).reshape(1, 3, 1, 1)
    if args.cuda:
        sat_weight = sat_weight.cuda(gpu_choiced)
    encoded_image_lum = torch.sum(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    encoded_image = encoded_image.reshape([-1, 3, 400, 400])
    if not args.no_jpeg:
        if torch.cuda.is_available():
            jpeg_quality = jpeg_quality.cuda(gpu_choiced)
            utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0,
                                           quality=jpeg_quality).cuda(gpu_choiced)
        encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0,
                                                       quality=jpeg_quality)
    if writer is not None:
        if global_step % 50 == 0:
            writer.add_scalar('transformer/rnd_bri', rnd_bri, global_step)
            writer.add_scalar('transformer/rnd_sat', rnd_sat, global_step)
            writer.add_scalar('transformer/rnd_hue', rnd_hue, global_step)
            writer.add_scalar('transformer/rnd_noise', rnd_noise, global_step)
            writer.add_scalar('transformer/contrast_low', contrast_low, global_step)
            writer.add_scalar('transformer/contrast_high', contrast_high, global_step)
            writer.add_scalar('transformer/jpeg_quality', jpeg_quality, global_step)

    return encoded_image  # 获得扭曲后的编码图


def get_secret_acc(secret_true, secret_pred):
    # 比特精确度,和秘密消息正确率
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)  # 四舍五入
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()  # numel()函数：返回数组中元素的个数
    return bit_acc, str_acc


def build_model(encoder, decoder, discriminator, secret_input, image_input, l2_edge_gain,
                borders, secret_size, M, loss_scales, yuv_scales, args, global_step, writer):
    if any(dep is None for dep in (lpips, torchgeometry, color, tools)):
        raise ImportError("build_model requires lpips, torchgeometry, kornia, and tools.")
    gpu_choiced = 1
    test_transform = transform_net(image_input, args, global_step)

    # M[:, 0, :, :]扭曲到正常的单应性矩阵H,M[:, 0, :, :] H的逆矩阵
    # 实际上原图乘以H还是H的逆都会发生扭曲
    input_warped = torchgeometry.warp_perspective(image_input, M[:, 1, :, :], dsize=(400, 400), flags='bilinear')
    mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), M[:, 1, :, :], dsize=(400, 400),
                                                 flags='bilinear')
    # 扭曲的载体图,边缘用原图填充
    input_warped += (1 - mask_warped) * image_input

    residual_warped = encoder((secret_input, input_warped))
    encoded_warped = residual_warped + input_warped
    residual = torchgeometry.warp_perspective(residual_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')

    if borders == 'no_edge':
        encoded_image = image_input + residual
    elif borders == 'black':
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        input_unwarped = torchgeometry.warp_perspective(image_input, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    elif borders.startswith('random'):
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400),
                                              flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        input_unwarped = torchgeometry.warp_perspective(input_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        ch = 3 if borders.endswith('rgb') else 1
        # torch.rand([1, ch, 1, 1])此处要进行维度扩展也即二者维度都是4才能运用广播机制
        encoded_image += (1 - mask) * torch.ones_like(residual) * torch.rand([1, ch, 1, 1])
    elif borders == 'white':
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400),
                                              flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        input_unwarped = torchgeometry.warp_perspective(input_warped, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        encoded_image += (1 - mask) * torch.ones_like(residual)
    elif borders == 'image':
        mask = torchgeometry.warp_perspective(torch.ones_like(residual), M[:, 0, :, :], dsize=(400, 400),
                                              flags='bilinear')
        encoded_image = residual_warped + input_warped
        encoded_image = torchgeometry.warp_perspective(encoded_image, M[:, 0, :, :], dsize=(400, 400), flags='bilinear')
        """
        torch.roll(input, shifts, dims=None) → Tensor
        eg:
        x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        >>> x
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])
        >>> torch.roll(x, 1, 0)
        tensor([[7, 8],
                [1, 2],
                [3, 4],
                [5, 6]])
        >>> torch.roll(x, shifts=(2, 1), dims=(0, 1))
        tensor([[6, 5],
                [8, 7],
                [2, 1],
                [4, 3]])
        """
        # 选择当前batch_size中的下一张图片进行边缘填充
        encoded_image += (1 - mask) * torch.roll(image_input, 1, 0)

    if borders == 'no_edge':
        D_output_real, _ = discriminator(image_input)
        D_output_fake, D_heatmap = discriminator(encoded_image)
    else:
        D_output_real, _ = discriminator(input_warped)
        D_output_fake, D_heatmap = discriminator(encoded_warped)

    transformed_image = transform_net(encoded_image, args, global_step, writer=writer)

    decoded_secret = decoder(transformed_image)

    bit_acc, str_acc = get_secret_acc(secret_input, decoded_secret)

    # lpips_loss
    loss_fn_alex = lpips.LPIPS(net='alex')
    if args.cuda:
        loss_fn_alex = loss_fn_alex.cuda(gpu_choiced)
    lpips_loss = torch.mean(loss_fn_alex(image_input, encoded_image))

    # 二分类交叉熵用来监督解码准确度
    cross_entropy = nn.BCELoss()  # 二分类
    if args.cuda:
        cross_entropy = cross_entropy.cuda(gpu_choiced)
    secret_loss = cross_entropy(decoded_secret, secret_input)

    size = (int(image_input.shape[2]), int(image_input.shape[3]))  # 图片大小(H,W)
    gain = 10
    # 该模型学习在图像的边缘添加分散的图案(可能是为了帮助定位)。通过增加边缘处余弦衰减的L2损失的权重来减轻这种影响？
    falloff_speed = 4  # Cos dropoff that reaches 0 at distance 1/x into image
    falloff_im = np.ones(size)
    for i in range(int(falloff_im.shape[0] / falloff_speed)):  # for i in range 100
        falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
        falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2  # [cos[(4*pi*i/400)+pi] + 1]/2
    for j in range(int(falloff_im.shape[1] / falloff_speed)):
        falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
        falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
    falloff_im = 1 - falloff_im
    falloff_im = torch.from_numpy(falloff_im).float()
    if args.cuda:
        falloff_im = falloff_im.cuda(gpu_choiced)
    falloff_im *= l2_edge_gain  # l2_edge_gain:10

    encoded_image_yuv = color.rgb_to_yuv(encoded_image)
    image_input_yuv = color.rgb_to_yuv(image_input)
    im_diff = encoded_image_yuv - image_input_yuv
    im_diff += im_diff * falloff_im.unsqueeze_(0)
    yuv_loss_op = torch.mean((im_diff) ** 2, axis=[0, 2, 3])  # 计算yuv三通道的均值
    yuv_scales = torch.Tensor(yuv_scales)
    if args.cuda:
        yuv_scales = yuv_scales.cuda(gpu_choiced)
    image_loss_op = torch.dot(yuv_loss_op, yuv_scales)

    D_loss = -(D_output_real - D_output_fake)
    G_loss = -D_output_fake
    # loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
    loss_op = loss_scales[0] * image_loss_op + loss_scales[1] * lpips_loss + loss_scales[2] * secret_loss
    if not args.no_gan:
        loss_op += loss_scales[3] * G_loss

    if global_step % 50 == 0:
        result_encoded_image, result_residual = prepare_deployment_hiding_graph(encoder, secret_input, image_input)
        psnr = tools.get_psnr(result_encoded_image, image_input)
        ssim = tools.get_ssim(result_encoded_image, image_input)

        writer.add_scalar('loss/image_loss', image_loss_op, global_step)
        writer.add_scalar('loss/lpips_loss', lpips_loss, global_step)
        writer.add_scalar('loss/secret_loss', secret_loss, global_step)
        writer.add_scalar('loss/G_loss', G_loss, global_step)
        writer.add_scalar('loss/dis_loss', D_loss, global_step)
        writer.add_scalar('loss/loss', loss_op, global_step)
        writer.add_scalar('color_loss/Y_loss', yuv_loss_op[0], global_step)
        writer.add_scalar('color_loss/U_loss', yuv_loss_op[1], global_step)
        writer.add_scalar('color_loss/V_loss', yuv_loss_op[2], global_step)

        writer.add_scalar('metric/bit_acc', bit_acc, global_step)
        writer.add_scalar('metric/str_acc', str_acc, global_step)
        writer.add_scalar('metric/PSNR', psnr, global_step)
        writer.add_scalar('metric/SSIM', ssim, global_step)

        writer.add_image('input/image_input', image_input[0], global_step)
        writer.add_image('input/image_warped', input_warped[0], global_step)
        writer.add_image('encoded/encoded_warped', encoded_warped[0], global_step)
        writer.add_image('encoded/residual_warped', residual_warped[0] + 0.5, global_step)
        writer.add_image('encoded/encoded_image', encoded_image[0], global_step)
        writer.add_image('transformed/transformed_image', transformed_image[0], global_step)
        writer.add_image('result/result_encoded_image', result_encoded_image[0], global_step)
        writer.add_image('result/result_residual', result_residual[0], global_step)
        writer.add_image('discriminator', D_heatmap[0], global_step)

    # print('image_loss = %.3f,image_loss_scale = %.3f' % (image_loss_op, loss_scales[0]))
    return loss_op, secret_loss, D_loss, bit_acc, str_acc


def prepare_deployment_hiding_graph(encoder, secret_input, image_input):
    residual = encoder((secret_input, image_input))
    encoded_image = residual + image_input
    # tf.clip_by_value()将encoded_image的值弄到区间[0,1]之间
    # 小于0的用0替代，大于1的用1替代
    encoded_image = torch.clamp(encoded_image, 0, 1)

    return encoded_image, residual + .5


def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoder(image_input)

    return torch.round(decoded_secret)
