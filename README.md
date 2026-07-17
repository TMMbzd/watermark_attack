# Watermark Attack

Watermark Attack 是一个基于 CRoSS 扩散隐写流程和 StegaStamp 水印模块整理出的实验项目。项目聚焦于图像水印嵌入、扩散模型语义替换、背景替换攻击，以及攻击后水印恢复效果评估。

## Project Structure

- `demo.py`: 基础 CRoSS 隐藏与恢复流程。
- `dataset.py`: StegaStamp 训练/验证数据读取模块。
- `encode_image.py`: 使用 StegaStamp encoder 给图像嵌入水印。
- `decode_image.py`: 使用 StegaStamp decoder 提取水印并计算 bit accuracy。
- `stega_cross.py`: 扩散隐写和水印模块结合的全局替换流程。
- `gauss_test14.py`: 背景替换、局部拼接和水印残差处理流程。
- `train_stega_cross.py`: 使用贝叶斯优化搜索 `lamda_text` 和 `lamda_wm`。
- `model.py`, `tools.py`, `utils.py`: StegaStamp 模型结构和辅助函数。
- `metrics.py`: 轻量 PSNR/SSIM 指标实现。
- `examples/`: 整理后的示例输入和中间结果图。
- `checkpoints/`: 保留必要的轻量模型文件；大权重建议本地放置，不直接提交。
- `results/`: 默认输出目录。

## Environment

```bash
pip install -r requirements.txt
```

Stable Diffusion v1.5 可以使用 HuggingFace 模型名，也可以指定本地目录：

```bash
export CROSS_SD_MODEL=/path/to/stable-diffusion-v1-5
export HF_TOKEN=your_huggingface_token
```

水印相关流程需要 StegaStamp checkpoint。请把包含 `model_encoder` 和 `model_decoder` 的权重放到：

```text
checkpoints/stegastamp.pth
```

也可以在运行时用 `--stegastamp_model /path/to/model.pth` 指定。

## Examples

基础隐藏与恢复：

```bash
python demo.py \
  --image_path examples/asserts/1.png \
  --private_key "a gold fish in the aquarium looks around" \
  --public_key "a gold fish in the hot pot" \
  --save_path results/demo \
  --num_steps 50
```

全局替换与水印恢复：

```bash
python stega_cross.py \
  --gt_path examples/01_gt.png \
  --stegastamp_model checkpoints/stegastamp.pth \
  --private_key "a cat" \
  --public_key "a tiger" \
  --save_path results/stega_cross
```

贝叶斯优化搜索最佳参数：

```bash
python train_stega_cross.py \
  --gt_path examples/01_gt.png \
  --stegastamp_model checkpoints/stegastamp.pth \
  --opt_calls 10 \
  --save_path results/train_stega_cross
```

背景替换攻击流程：

```bash
python gauss_test14.py \
  --gt_path examples/01_gt.png \
  --target_bg_path examples/backgrounds/target_bg.png \
  --mask_path_step5 examples/masks/mask_002.png \
  --mask_path_step6 examples/masks/mask_003.png examples/masks/mask_024.png \
  --pre_generated_reveal examples/07_revealed.png \
  --stegastamp_model checkpoints/stegastamp.pth \
  --save_path results/gauss_test14
```



```bibtex
@article{yu2023cross,
  title={CRoSS: Diffusion Model Makes Controllable, Robust and Secure Image Steganography},
  author={Yu, Jiwen and Zhang, Xuanyu and Xu, Youmin and Zhang, Jian},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
