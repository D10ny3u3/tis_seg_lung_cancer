本仓库提供了对 https://github.com/cpath-ukk/lung_cancer 代码测试的过程。
测试内容主要在组织分割部分（02、04）。
根据当前环境、注释、文件系统等变异因素会对代码进行调整。详细步骤见tutorial目录，分别为环境配置、训练、分割三个环节，供参考。

原readme部分中译：

**所提供代码的参考说明**

注意！我们提供了用于开发和实现的工作脚本。
本代码库按“原样”提供，不予支持。

注意！大多数实验和脚本都是在 Anaconda 环境中使用 Ubuntu 22.04 运行的。我们建议使用该环境设置。

## 01_qupath_patch_extraction
这是一个 Groovy 脚本，用于 QuPath v. > 0.3，用于提取训练或验证/测试图像块（patches）以及相应的分割掩膜（来自标注的 ground truth）。

## 02_train_pixel_wise_segmentation
train_script.py：这是一个训练脚本。
它使用了 Python v.3.9 和 pytorch v.1.10。
使用了 segmentation-models-pytorch v.0.3.1 库（或任何其他版本）进行模型构建。
pytorch_toolbelt 用于构建部分损失函数（v.0.6.3）。

dataset_v2.py：用于数据集构建和数据增强的辅助脚本。
使用了 Albumentations 包（v1.3.0）进行数据增强。

## 03_Patch_level_validation_testing
这是一个 Python 脚本，用于使用图像块级别的分割准确度指标（Dice 分数和 IoU）对训练好的检查点进行额外的验证或测试。

## 04_WSI_inference
这是一个推理脚本，用于处理全切片图像（WSI）。

提供三个版本：1) 用于主要分割算法（使用切片和组织检测掩膜作为输入），2) 用于亚型分类/监督式像素级分割算法（使用来自主要算法的多类组织分割掩膜 + 切片作为输入），3) 与 2) 相同，但适用于基于 UNI 和 Prov-GigaPath 特征提取器的分类器（需要来自 Hugging Face 的原始模型检查点以及基于这些特征训练的分类器）。

注意！我们提供用于审查的脚本，该脚本基于 openslide（v.3.4.1 或更高版本）库。

- wsi_tis_detect.py：用于运行独立的组织与背景分割脚本（基于 GrandQC 工具的分割算法——手稿在审中，该工具将在发表后开源）。对于自定义组织检测器，可以忽略此脚本，并可在 main.py 中使用来自其他来源的组织检测图。
- main.py：使用训练好的像素级分割模型启动推理的脚本。需要组织掩膜作为输入，该掩膜应存储在输出文件夹中。
- run.sh：用于串联组织分割和伪影检测步骤的 bash 脚本（可被自定义组织检测流程忽略）。
- wsi_colors.py：定义所用颜色方案的脚本。
- wsi_maps.py：包含在原始 WSI 上制作分割掩膜叠加图的函数的脚本。
- wsi_process.py：WSI 处理流程脚本，输出：分割掩膜（类别编码）、分割掩膜（RGB，对应 wsi_colors.py 中定义的颜色方案）。
- wsi_slide_info.py：用于检索推理所需 WSI 元数据的流程。
- wsi_tis_detect_helper_fx.py：组织分割脚本（wsi_tis_detect.py）的辅助函数。

处理过的切片的输出 txt 文件将被保存。
对于亚型分类版本，它将包含亚型面积和最终切片分类结果。

## 05_foundation_model_classifier_training
用于训练基于基础模型提取特征的肺癌亚型分类监督分类器的脚本。

- prepare_ds_prov_gigapath_gpu_monodir.py（UNI 基础模型）
- prepare_ds_uni_gpu_multidir.py（Prov-GigaPath 基础模型）
用于准备训练/验证/测试数据集的脚本，实际上是将图像块转换为特征向量，这些特征向量将作为训练脚本的输入。

- train_validation.py
训练脚本，在提取的特征之上训练一个全连接层和一个分类层。

## 06_foundation_model_classifier_test_patch_level
用于评估基于基础模型（来自上述 5 步骤）的分类器在图像块级别准确度的测试脚本。
输出：图像块级别的亚型“概率”。
