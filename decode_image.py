import glob
import os
import bchlib
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import sys
from torchvision import transforms
from model import StegaStampDecoder
BCH_POLYNOMIAL = 137
BCH_BITS = 5

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
def evaluate_watermark_accuracy(image_path, model_path='checkpoints/stegastamp.pth', secret_size=100, device=None):
    """Decode the default 100-bit secret from an image and return bit accuracy."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"StegaStamp checkpoint not found: {model_path}. "
            "Place the trained checkpoint at checkpoints/stegastamp.pth "
            "or pass the model_path argument."
        )

    class Args:
        pass

    args = Args()
    args.model = model_path
    args.image = image_path
    args.secret_size = secret_size
    # args.images_dir=args.images_dir+'result_{}'.format(ti)
    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    decoder = StegaStampDecoder(args)
    checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
    decoder.load_state_dict(checkpoint['model_decoder'])
    decoder.to(device)
    decoder.eval()
    orisecret = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                        0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                        0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                        0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                        1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                        0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                        0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                        0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                        1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                        ]])
    orisecret = torch.from_numpy(orisecret).float()
    orisecret = orisecret.to(device)
    # decoder = torch.load(args.model).cuda()

    # bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    sum_bit=0
    number=0
    print('==========================================')
    for filename in files_list:
        number+=1
        # print(filename)
        image = cv2.imread(filename, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR -> RGB
        image = cv2.resize(image, (400,400))
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device)
        # image = Image.open(filename).convert("RGB")
        # image = np.array(ImageOps.fit(image, (400, 400)), dtype=np.float32)
        # image /= 255.
        save_name = filename.split('/')[-1].split('.')[0]
        secret = torch.round(decoder(image))
        # print(secret)
        # print(secret.shape)
        bit,_=get_secret_acc(secret,orisecret)
        # print('{} bit:{}'.format(save_name,bit))
        # sum_bit+=bit
    return bit


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="examples/asserts/1.png")
    parser.add_argument("--model", type=str, default="checkpoints/stegastamp.pth")
    parser.add_argument("--secret_size", type=int, default=100)
    args = parser.parse_args()
    print(evaluate_watermark_accuracy(args.image, args.model, args.secret_size))

