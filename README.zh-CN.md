# DKB³

本目录只包含 DKB³ 自身的实现（针对《Medical Image Analysis》投稿版本）。完整说明见 [英文 README](README.md)。

## 保留内容

- 双向 Brownian Bridge、共享先验编码器、五通道皮肤镜先验和结构损失。
- 双桥训练、掩模锚定的合成增强、KCS 候选选择。
- ISIC2016 / ISIC2018 配置、数据清单工具和必要测试。
- DKB³ 运行必需的底层桥模块；它不是额外的对比方法。

分割基线、其他方法的训练/验证适配、论文统计绘图和生成质量分析脚本均不再放在本目录。原始 `dual-bbdm` 工程不变。

## 安装与训练

先安装适合设备的 PyTorch/torchvision，再从本目录运行：

```bash
python -m pip install -r requirements.txt
python train_dual_bridge.py --config configs/isic2018.yaml
python export_bridge_artifacts.py --config configs/isic2018.yaml --use-ground-truth-candidate --no-visualizations
```

先按 [数据说明](data/README.md) 准备训练清单；不要直接运行时把验证或测试病例混入训练。ISIC2018 划分为 2594/100/1000，ISIC2016 为 720/180/379。合成图保存在 `artifacts/isic2018/synthetic_images/`，下游训练时使用原训练 mask 作标签，分割算法由你自行提供。

## 在线细化

KCS 不需要捆绑分割网络：提供自己的分割概率图即可。输入为与缩放后 RGB 图像对齐的二维浮点 `.npy`，数值 [0,1]、前景为病灶；默认分辨率 256×256。它必须是模型预测，不能用真实标签代替。

```bash
python refine_masks.py --config configs/isic2018.yaml --checkpoint outputs/bridge/isic2018/last.pt --image examples/image.jpg --primary-probability examples/primary_probability.npy --output outputs/refined/case.png
```

示例输入是占位路径，需要替换成自己的文件。输出包括细化 mask、概率图和候选评分。默认生成两次桥采样及其均值，使用图像证据进行 KCS，不读取真实标签。外部分割器的 TTA 候选可以通过重复 `--candidate 名称=概率图.npy` 传入，但必须先恢复到原图坐标。

也可以在自己的代码中调用 `dual_bbdm.refine_masks(system, image, probability)`。本次只调整模型接口和发布范围，没有重训模型或改变双桥训练、采样和评分公式。
