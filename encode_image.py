import os
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
from model import StegaStampEncoder


class StegaStampWatermarker:
    def __init__(self, model_path='checkpoints/stegastamp.pth', device='cuda:0', width=400, height=400):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.height = height

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"StegaStamp checkpoint not found: {model_path}. "
                "Place the trained checkpoint at checkpoints/stegastamp.pth "
                "or pass --stegastamp_model."
            )

        # Encoder only needs args.secret_size; this keeps inference independent
        # from the original training argparse configuration.
        class DummyArgs:
            secret_size = 100

        self.encoder = StegaStampEncoder(DummyArgs())
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['model_encoder'])
        self.encoder.eval()
        self.encoder.to(self.device)

    def add_watermark(self, image_path, secret_bits=None, save_dir=None, output_size=(512, 512)):
        """
        为单张图片添加水印
        :param image_path: 输入图片路径
        :param secret_bits: 1x100 的 torch.Tensor 或 np.array。如果为 None，则使用默认值
        :param save_dir: 保存目录
        :param output_size: 最终输出的分辨率
        """
        # 1. 准备秘密信息 (默认 100 bits)
        if secret_bits is None:
            secret_bits = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                                     0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                                     0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                                     0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                                     1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
                                     0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                                     0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
                                     1, 0, 1, 0, 0, 0, 0, 0, 1, 0]])

        if isinstance(secret_bits, np.ndarray):
            secret_bits = torch.from_numpy(secret_bits).float()
        secret_bits = secret_bits.to(self.device)

        # 2. 读取并预处理图片
        image_cv = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        image_resized = cv2.resize(image_cv, (self.width, self.height))
        image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0).to(self.device)

        # 3. 模型推断
        with torch.no_grad():
            residual = self.encoder((secret_bits, image_tensor))
            encoded_image = torch.clamp(residual + image_tensor, 0, 1)

        # 4. 后处理与尺寸调整
        # 转换回 PIL Image 以便缩放和保存
        res_img = transforms.ToPILImage()(encoded_image.squeeze(0).cpu())
        res_img = res_img.resize(output_size, Image.LANCZOS)

        # 计算残留图 (可视化用)
        res_residual = transforms.ToPILImage()((residual.squeeze(0) + 0.5).cpu())
        res_residual = res_residual.resize(output_size, Image.LANCZOS)

        # 5. 保存结果
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            base_name = os.path.basename(image_path).split('.')[0]
            save_path_img = os.path.join(save_dir, f"{base_name}_encoded.png")
            save_path_res = os.path.join(save_dir, f"{base_name}_residual.png")

            res_img.save(save_path_img)
            res_residual.save(save_path_res)
            print(f"Saved: {save_path_img}")

        return res_img, res_residual


# --- 使用示例 ---
if __name__ == "__main__":
    MODEL_PATH = 'checkpoints/stegastamp.pth'
    IMG_PATH = 'examples/asserts/1.png'
    SAVE_DIR = 'results/watermark_demo'

    # 初始化一次
    watermarker = StegaStampWatermarker(MODEL_PATH)

    # 调用函数添加水印
    watermarked_image, _ = watermarker.add_watermark(IMG_PATH, save_dir=SAVE_DIR)
