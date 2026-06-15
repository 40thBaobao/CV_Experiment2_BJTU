# 计算机视觉基础实验二：交大视觉印象

本项目为《计算机视觉基础》课程阶段二实验，主要完成图像检索和文字检测两个任务。

## 项目文件

```text
.
├── CVfinal_version.ipynb   # 实验 notebook
├── CVfinal_version.py      # notebook 导出的 python 文件
├── README.md               # 项目说明
└── demo_video.mp4          # 实验演示视频
```

## 数据集

本实验使用课程提供的“交大视觉印象数据集2026”。由于数据集文件较大，本仓库不上传原始数据集。

运行代码前需要在 notebook 中修改 `DATA_ROOT` 为本地数据集路径。

数据集目录结构如下：

```text
交大视觉印象数据集2026/
├── image_retrieval/
│   ├── base/
│   └── query/
└── object_detection/
    └── data/
```

## 运行环境

主要使用的 Python 库包括：

```text
torch
torchvision
numpy
pandas
matplotlib
pillow
easyocr
```

## 实验方法

图像检索部分使用 ImageNet 预训练 ResNet50 提取图像特征，通过余弦相似度计算 query 图像与 base 图像之间的相似度，并使用 Precision@K 作为评价指标。

文字检测部分使用 EasyOCR 的 detect-only 模式，对校园地标图片进行文字区域检测，并生成可视化检测结果。每类 landmark 选取 2 组结果，共 24 组，用于人工核验。

## 运行方式

打开并运行：

```text
CVfinal_version.ipynb
```

主要流程包括：

1. 读取图像检索和文字检测数据。
2. 使用 ResNet50 提取图像特征。
3. 计算 Top-K 检索结果和 P@K 指标。
4. 使用 EasyOCR 生成文字检测可视化结果。
5. 保存并展示实验结果。

## 说明

完整实验结果、P@K 曲线、文字检测可视化结果和人工核验分析见实验报告。

演示视频为：

```text
demo_video.mp4
```
