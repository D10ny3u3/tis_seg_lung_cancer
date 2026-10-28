# `wsi_process.py` 相对 `wsi_process_0.py` 的调整记录

## 结论

新版 `wsi_process.py` 在原有 WSI 分块推理流程上增加了**逐 patch 数据导出**，用于保存每个网格位置的 H&E 图像、组织检测 mask、类别 mask 和彩色 mask；同时修正了 mask 数据类型、加入 `torch.no_grad()`，并替换了 Pillow 中已移除的 `Image.ANTIALIAS`。

差异规模：新增 82 行、删除 13 行，文件由 120 行增加到 189 行。核心分块网格、组织检测判断、模型推理和整图 mask 拼接逻辑未发生结构性改变。

## 主要调整

| 位置 | 原版 `wsi_process_0.py` | 新版 `wsi_process.py` | 影响 |
|---|---|---|---|
| 导入 | 无 `os` | 新增 `import os` | 支持创建逐 patch 输出目录和构造文件路径。 |
| 输出目录 | 无 patch 级输出 | 在函数内创建固定根目录及子目录 | 新增 patch 图像和 mask 的落盘功能。 |
| patch 命名 | 无 | 使用 `patch_y{he:04d}_x{wi:04d}.png` | 同一网格位置的各类输出共用同一文件名。 |
| 组织检测 mask | 仅内存中使用 | 每个 patch 都保存到 `tissue_detection_masks/` | 新增 padding 后组织 mask 的可视化/核对数据。 |
| 有组织 patch | 读取、resize、推理，不保存 patch | 读取、resize、保存 H&E patch，再推理并保存 mask | 增加磁盘 I/O；推理条件不变。 |
| 无组织 patch | 仅生成内存中的空白 mask | 仍读取并保存 H&E patch，同时生成并保存空白 mask | 虽然不推理，但所有网格位置都有成对的图像和 mask；会增加 I/O 和存储占用。 |
| 推理上下文 | 直接调用 `model.predict()` | 使用 `with torch.no_grad():` 包裹推理 | 明确关闭梯度计算，降低推理时的显存/内存开销。 |
| mask 数据类型 | `mask_raw` 为 `int8`；空白 mask 默认 `float64` | 统一使用 `uint8`，空白 mask 使用 `(m_p_s, m_p_s)` | 类型更符合类别 ID/PIL 保存需求，避免不同分支拼接时隐式类型提升。 |
| 空白 mask 尺寸 | 固定为 `(512, 512)` | 使用 `(m_p_s, m_p_s)` | 与模型输入尺寸保持一致，不再依赖硬编码的 512。 |
| 彩色 mask | 仅在最终整图生成时调用 `make_1class_map_thr()` | 每个 patch 额外生成并保存彩色 mask | 新增 `segmentation_masks_color/` 输出。 |
| 图像缩放枚举 | `Image.ANTIALIAS` | `Image.Resampling.LANCZOS` | 兼容新版 Pillow；`ANTIALIAS` 已在新版本中移除。 |

## 新增输出目录

新版函数使用以下固定根目录：

```text
/home/suxin/projects/kludt_reproduction/outputs/test_one_wsi_output/patches/
├── images/
├── segmentation_masks/
├── tissue_detection_masks/
└── segmentation_masks_color/
```

每个网格位置最多产生四类文件：

- `images/patch_y####_x####.png`：缩放至模型输入尺寸的 H&E patch。
- `segmentation_masks/patch_y####_x####.png`：类别 ID mask，供定量分析使用。
- `tissue_detection_masks/patch_y####_x####.png`：组织检测 mask。
- `segmentation_masks_color/patch_y####_x####.png`：便于目视检查的彩色 mask。

## 行为变化

1. **推理逻辑不变**：是否执行模型推理仍由 `np.count_nonzero(td_patch == 0) > 50` 决定。
2. **所有网格位置都会保存 H&E patch 和 mask**，包括未执行推理的背景 patch。
3. **整图拼接结果不变**：仍将各 patch 的类别 mask 拼接，并补齐右侧和底部 buffer。
4. **最终整图彩色 mask 的生成和缩放逻辑保留**，只是缩放方式改为 LANCZOS。
5. **磁盘占用和运行时间会增加**，尤其是整张 WSI 网格较大或背景 patch 较多时。

## 需要注意的问题

1. **输出路径硬编码**：`patch_root` 固定为上述绝对路径，没有通过参数传入，也没有跟随 `main.py` 的 `OUTPUT_DIR`。
2. **多切片会相互覆盖**：文件名只包含网格坐标，不包含切片名。如果 `main.py` 一次处理多张 WSI，后一张切片会覆盖前一张切片的同名 patch 文件。
3. **彩色 mask 目录重复创建**：`os.makedirs(color_mask_dir, exist_ok=True)` 位于最内层 patch 循环中，功能正确，但可以在循环外创建以减少重复系统调用。
4. **新增 I/O 可能显著变慢**：每张切片都要为完整网格写出图像和 mask，建议仅在调试、数据集构建或需要人工核验时启用。

## 未改变的部分

- `slide_process_single()` 的函数签名和调用方式。
- WSI 分块循环、patch 坐标计算和组织检测 patch 切片方式。
- 模型预处理、预测结果 `argmax` 和背景类别覆盖逻辑。
- 整图 mask 拼接、右侧/底部 padding 和返回值结构。
