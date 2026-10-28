创建工作目录。

```bash
mkdir -p ~/projects/kludt_reproduction/{code,data,models,outputs,notes}
cd ~/projects/kludt_reproduction
git clone https://github.com/cpath-ukk/lung_cancer.git code/lung_cancer
#### Cloning into 'code/lung_cancer'...
#### /usr/lib/git-core/git-remote-https: symbol lookup error: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
unset LD_LIBRARY_PATH
git clone https://github.com/cpath-ukk/lung_cancer.git code/lung_cancer
```

作者提供了3个checkpoints。可以先下载下来，不过对于组织类型分割任务可能没有太大意义。这些模型主要的目的是肺癌亚型诊断。

```bash
cd ~/projects/kludt_reproduction/models
aria2c -c -s 4 -x 4 https://zenodo.org/api/records/12818437/files/CLAM_UNI_SUBTYPING_CHECKPOINTS.zip/content
aria2c -c -s 4 -x 4 https://zenodo.org/api/records/12818437/files/PROV_GIGAPATH_CLASSIFIER_CHECKPOINTS.zip/content
aria2c -c -s 4 -x 4 https://zenodo.org/api/records/12818437/files/UNI_CLASSIFIER_CHECKPOINTS.zip/content
```

设置conda环境

```bash
conda create -n kludt python=3.9 pip -y
dpkg -l openslide-tools libopenslide-dev
sudo apt-get update
env | grep -i proxy
unset http_proxy https_proxy all_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide-dev

conda activate kludt
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  openslide-python pillow numpy
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
#### 降级numpy
python -m pip install "numpy<2" -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install segmentation-models-pytorch==0.3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install timm==0.9.12 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install albumentations==1.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorch-toolbelt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
#### albumentations触发了numpy升级
python -m pip install --force-reinstall "numpy==1.26.4" -i https://pypi.tuna.tsinghua.edu.cn/simple

```

从https://zenodo.org/records/12818382下载训练文件。放在```~/projects/kludt_reproduction/data/kludt_manual```目录下。

新建一个python文件split_dataset.py在02_train_pixel_wise_segmentation目录下，对训练文件进行train和val的拆分。

```
from pathlib import Path
import random
import shutil

root = Path("/home/suxin/projects/kludt_reproduction/data/kludt_manual")
img_dir = root / "train" / "image"
mask_dir = root / "train" / "mask"
val_img_dir = root / "val" / "image"
val_mask_dir = root / "val" / "mask"

assert img_dir.is_dir(), f"Missing: {img_dir}"
assert mask_dir.is_dir(), f"Missing: {mask_dir}"
assert not (val_img_dir.exists() or val_mask_dir.exists()), \
    "Validation directories already exist; stop to avoid overwriting data."

image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
images = {p.stem: p for p in img_dir.iterdir()
          if p.is_file() and p.suffix.lower() in image_exts}
masks = {p.stem: p for p in mask_dir.iterdir()
         if p.is_file() and p.suffix.lower() in image_exts}

if images.keys() != masks.keys():
    print("Images without matching masks:", sorted(images.keys() - masks.keys())[:10])
    print("Masks without matching images:", sorted(masks.keys() - images.keys())[:10])
    raise SystemExit("Filename mismatch detected; no files were moved.")

pairs = sorted(images.keys())
if len(pairs) < 2:
    raise SystemExit("Not enough image-mask pairs to split.")

random.Random(42).shuffle(pairs)
n_val = max(1, round(len(pairs) * 0.2))
val_ids = pairs[:n_val]  # 修正：取前n_val个作为验证集

val_img_dir.mkdir(parents=True)
val_mask_dir.mkdir(parents=True)

for stem in val_ids:
    shutil.move(str(images[stem]), str(val_img_dir / images[stem].name))
    shutil.move(str(masks[stem]), str(val_mask_dir / masks[stem].name))

print("Total pairs before split:", len(pairs))
print("Moved to validation:", n_val)
print("Remaining training pairs:", len(pairs) - n_val)
print("Validation directories created successfully.")
```


