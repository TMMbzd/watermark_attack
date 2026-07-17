import cv2
import numpy as np


def calculate_psnr(img1, img2, crop_border=0, input_order="HWC", test_y_channel=False):
    """Calculate PSNR for uint8 images in [0, 255]."""
    img1 = _prepare_image(img1, crop_border, input_order, test_y_channel)
    img2 = _prepare_image(img2, crop_border, input_order, test_y_channel)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2, crop_border=0, input_order="HWC", test_y_channel=False):
    """Calculate SSIM, averaged over channels for RGB images."""
    img1 = _prepare_image(img1, crop_border, input_order, test_y_channel)
    img2 = _prepare_image(img2, crop_border, input_order, test_y_channel)
    if img1.ndim == 2:
        return _ssim_channel(img1, img2)
    return float(np.mean([_ssim_channel(img1[..., i], img2[..., i]) for i in range(img1.shape[2])]))


def _prepare_image(img, crop_border, input_order, test_y_channel):
    if input_order not in {"HWC", "CHW"}:
        raise ValueError("input_order must be 'HWC' or 'CHW'")
    img = np.asarray(img)
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    if crop_border:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    img = img.astype(np.float64)
    if test_y_channel and img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float64)
    return img


def _ssim_channel(img1, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())
