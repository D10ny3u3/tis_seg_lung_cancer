为了调适训练代码，新增了train_script_adj.py。主要变化：

# `train_script_adj.py` 相对 `train_script.py` 的变化

## 对比说明

- 基准文件：`train_script.py`
- 对比文件：`train_script_adj.py`
- 差异规模：新增 71 行，删除 12 行
- 总体来看，修改集中在训练参数、数据路径、类别权重、评估指标和训练前调试信息；网络结构、损失函数选择、优化器及训练主循环未发生实质性变化。

## 变化概览

| 项目 | `train_script.py` | `train_script_adj.py` | 主要影响 |
|---|---|---|---|
| `ACTIVATION` | `'softmax2d'` | `None` | 模型改为输出 logits，与 `CrossEntropyLoss` 及新 IoU 指标的 `argmax` 处理方式一致 |
| `OVERSAMPLE_FLAG` | `True` | `False` | 使用普通平衡批次构建，不再走 oversampling 分支 |
| `DATA_DIR` | `'/path/to/dataset'` | `'/home/[User]/projects/kludt_reproduction/data/kludt_manual'` | 改为本机数据集绝对路径 |
| `CLASSES` | `9999` | `13` | 明确为 13 类分割任务（包含背景/未标注类 0） |
| `B_S_TRAIN` | `9999` | `4` | 训练 batch size 改为 4 |
| `CLASS_WEIGHTS` | 11 个元素 | 13 个元素 | 与新类别数匹配：`[0.0] + [1.0] * 12` |
| `MODEL_SAVE_NAME` | `'/data/v01/v01_E'` | `'/home/[User]/projects/kludt_reproduction/checkpoints/kludt_'` | 检查点改存到本机指定目录 |
| 训练指标 | 二分类式 `IoU(threshold=0.5)` | 自定义 `MulticlassIoU` | 支持多类别逐一计算 IoU |
| 验证指标 | `IoU`、`Fscore`、`Accuracy` | 仅自定义 `MulticlassIoU` | 验证阶段不再报告 Fscore 和 Accuracy |
| 训练前调试 | 无 | 新增一个 batch 的形状及前向输出检查 | 便于确认输入、标签和模型输出是否正确 |

## 详细变化

### 1. 训练参数调整

原文件第 31 行将模型激活函数设为 `softmax2d`：

```python
ACTIVATION = 'softmax2d'
```

调整后第 31 行改为：

```python
ACTIVATION = None
```

因此模型直接输出 logits。当前代码使用的 `CrossEntropyLoss` 和新版 `MulticlassIoU` 都按 logits 处理，其中 IoU 指标通过 `y_pr.argmax(dim=1)` 得到预测类别。

### 2. 关闭 oversampling

`OVERSAMPLE_FLAG` 由 `True` 改为 `False`。因此初始化和每个 epoch 重新构建批次时，都会进入 `construct_balanced_batches(...)` 分支，而不再使用 `get_oversampled_batches(...)`。

### 3. 数据集路径和模型保存路径本地化

- `DATA_DIR` 从占位路径 `'/path/to/dataset'` 改为：

  ```python
  '/home/[User]/projects/kludt_reproduction/data/kludt_manual'
  ```

- `MODEL_SAVE_NAME` 从 `'/data/v01/v01_E'` 改为：

  ```python
  '/home/[User]/projects/kludt_reproduction/checkpoints/kludt_'
  ```

这两项修改使脚本依赖当前运行环境中的具体目录。运行前需要确认数据集目录和检查点保存目录存在。

### 4. 类别数与 batch size 调整

- `CLASSES` 从 `9999` 改为 `13`。
- `B_S_TRAIN` 从 `9999` 改为 `4`。
- `B_S_VAL` 保持为 `1`。

`CLASSES = 13` 与注释中“包含类别 0”的设定一致，并会直接影响模型输出通道数、交叉熵类别权重和 IoU 类别遍历范围。

### 5. 类别权重扩展

原代码将 `CLASS_WEIGHTS` 定义为 11 个元素：

```python
CLASS_WEIGHTS = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

调整后原定义被注释，并改为 13 个元素：

```python
CLASS_WEIGHTS = [0.0] + [1.0] * 12
```

即类别 0 的权重为 0，其余 12 个类别权重均为 1。这与 `CLASSES = 13` 相匹配。由于 `IGNORE_INDEX = 0`，类别 0 在损失计算中仍会被忽略。

### 6. 禁用一行无实际作用的 oversampled 权重访问

原文件第 165 行直接执行：

```python
train_balanced_dataset.oversampled_class_weights
```

调整后该行被注释：

```python
# train_balanced_dataset.oversampled_class_weights
```

该语句原本只是读取属性，没有赋值或打印，因此注释后不改变训练逻辑。

### 7. 用多类别 IoU 替换原有指标

调整文件在损失函数配置之后新增了 `MulticlassIoU` 类（第 248-287 行），主要逻辑为：

1. 对模型输出执行 `argmax(dim=1)`，得到每个像素的预测类别。
2. 展平预测结果和真实标签。
3. 排除真实标签为 `IGNORE_INDEX`（即 0）的像素。
4. 对每个类别分别计算：

   ```text
   IoU = intersection / union
   ```

5. 只统计当前 batch 中 `union > 0` 的类别，并对这些类别的 IoU 取平均。
6. 如果所有类别都被排除，则返回值为 0 的 tensor。

该类还实现了 `to(device)`，用于兼容 `segmentation_models_pytorch` 的 `TrainEpoch`/`ValidEpoch` 初始化流程。

指标配置相应改为：

```python
metrics = [
    MulticlassIoU(num_classes=CLASSES, ignore_index=IGNORE_INDEX)
]

metrics_val = [
    MulticlassIoU(num_classes=CLASSES, ignore_index=IGNORE_INDEX)
]
```

与原代码相比：

- 训练指标从 `smp.utils.metrics.IoU(threshold=0.5)` 改为多类别 IoU。
- 验证指标从 `IoU`、`Fscore`、`Accuracy` 三项缩减为仅 `MulticlassIoU`。
- 新指标采用“仅对当前 batch 中出现的类别求平均”的策略，因此与固定类别集合上的宏平均 IoU 不完全相同。

### 8. 新增训练前输入及前向检查

在 `torch.backends.cudnn.benchmark = True` 之后、正式训练循环之前，新增了以下调试流程：

1. 从 `train_loader` 取出一个 batch。
2. 打印图像形状、掩码形状、掩码数据类型和部分唯一标签值。
3. 将模型切换到 `eval()` 模式。
4. 在 `torch.no_grad()` 下将图像移动到 `DEVICE` 并执行一次前向传播。
5. 打印模型输出形状。
6. 将模型切回 `train()` 模式。

该检查不执行反向传播，也不会更新模型参数，主要用于在正式训练前验证输入、标签和输出张量的形状。

### 9. 仅格式变化

训练主循环中新增了若干空行，文件末尾也增加了一个空行。这些变化不影响程序行为。

## 保持不变的部分

以下核心配置和流程在调整文件中没有实质性修改：

- 网络类型仍为 `unet++`，编码器仍为 `timm-efficientnet-b0`，使用 ImageNet 预训练权重。
- 损失函数仍选择 `CE`。
- patch size、epoch 数、学习率及学习率衰减策略保持不变。
- 优化器仍为 Adam，学习率和衰减 epoch 配置未变。
- 数据加载、批次重排、模型保存及训练/验证循环的主体结构保持不变。
