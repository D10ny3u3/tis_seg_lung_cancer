- 下载 https://zenodo.org/records/14507273 到 `~/projects/kludt_reproduction/code/lung_cancer/04_WSI_inference/01_main_segmentation_algorithm/models/td`。
参考：https://github.com/cpath-ukk/grandqc/issues/3

- 修改 wsi_tis_detect.py
```bash
MODEL_TD_DIR = './models/td/'
MODEL_TD_NAME = 'Tissue_Detection_MPP10.pth'
MPP_MODEL_TD = 10
M_P_S_MODEL_TD = 512
ENCODER_MODEL_TD = 'timm-efficientnet-b0'
ENCODER_MODEL_TD_WEIGHTS = 'imagenet'
```

- 创建链接：
```bash
SOURCE_DIR="/media/suxin/EXOS/PATHO_LUNG/TCGA_PATHO_DIAG_SLIDES/svs"
TEST_DIR="/home/suxin/projects/kludt_reproduction/outputs/test_one_wsi"

mkdir -p "$TEST_DIR"
SLIDE=$(find "$SOURCE_DIR" -maxdepth 1 -type f \
    -name 'TCGA-75-7030-01Z-00-DX1*.svs' | head -n 1)

if [ -z "$SLIDE" ]; then
    echo "错误：未找到目标 TCGA 切片，请检查文件名。"
else
    ln -sfn "$SLIDE" "$TEST_DIR/$(basename "$SLIDE")"
    echo "已建立软链接："
    ls -lh "$TEST_DIR"
fi
```

- 运行组织分割

```bash
cd /home/suxin/projects/kludt_reproduction/code/lung_cancer/04_WSI_inference/01_main_segmentation_algorithm

python wsi_tis_detect.py \
  --slide_folder /home/suxin/projects/kludt_reproduction/outputs/test_one_wsi \
  --output_dir /home/suxin/projects/kludt_reproduction/outputs/test_one_wsi_output
```
