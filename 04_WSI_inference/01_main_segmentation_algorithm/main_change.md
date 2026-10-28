# `main.py` 与 `main_adj.py` 差异记录

## 结论

`main_adj.py` 是在 `main.py` 基础上制作的诊断/调试调整版。核心分割流程没有改变，主要增加了 WSI 分辨率诊断输出，并将静默吞掉异常改为打印完整堆栈后重新抛出。

代码规模变化：新增 24 行、删除 4 行；文件由 202 行增加到 222 行。虽然改动量不算大，但错误处理行为变化明显，因此记录如下。

## 具体调整

| 位置 | `main.py` | `main_adj.py` | 影响 |
|---|---|---|---|
| 导入部分 | `from wsi_process import slide_process_single, slide_process_double` | `from wsi_process import slide_process_single` | 删除未使用的 `slide_process_double` 导入，接口本身未改变。 |
| `slide_info()` 调用之后 | 直接进入 `ONLY_SLIDE_INFO_FLAG` 判断 | 新增 `WSI resolution diagnostics` 诊断输出 | 会额外打印 `p_s`、模型 patch 大小、目标 MPP、`slide_info` 返回的 MPP、WSI 尺寸、level downsample、原始 MPP X/Y，以及估算的 patch MPP。 |
| MPP 校验 | 无 | 新增 `mpp > 0` 判断 | MPP 无效时打印警告；有效时打印 `mpp * p_s / M_P_S_MODEL_1` 的估算值。 |
| 主循环异常处理 | `except:` 后仅打印 “There was some problem with the slide.”，并继续处理下一张切片 | `except Exception:` 后调用 `traceback.print_exc()` 并 `raise` | 不再静默跳过失败切片；会输出完整异常堆栈，并立即终止整个批处理，便于定位问题。 |
| 空白字符 | 无 | 组织图 resize 后存在一行尾随空格变化 | 无功能影响。 |

## 未改变的部分

- 命令行参数、模型路径、模型加载方式和推理设备。
- 组织检测图加载与 resize 逻辑。
- `slide_process_single()` 的调用参数和处理流程。
- map、mask、overlay 的保存方式。
- 每张切片的统计报告内容和输出目录结构。

## 运行行为差异

1. `main_adj.py` 每张切片都会输出更多分辨率诊断信息，适合检查 MPP、patch 尺寸和保存分辨率是否符合预期。
2. 由于异常会被重新抛出，`main_adj.py` 遇到坏切片时会停止当前批次；`main.py` 则会打印一条简短提示后继续处理后续切片。
3. 两者的模型推理和输出结果生成逻辑相同，上述改动不会改变正常切片的分割计算本身。

## 注意事项

- 诊断中的 “Estimated saved patch MPP” 使用 `mpp * p_s / M_P_S_MODEL_1` 计算。由于 `p_s` 在 `slide_info()` 中已经取整，该值是估算值，不一定完全等于模型目标 MPP。
- `slide.properties.get("openslide.mpp-x")` 和 `openslide.mpp-y` 的值可能为字符串或 `None`；当前代码只打印原始属性，不会因此直接报错。
