from pathlib import Path
import random
import shutil

root = Path("~/projects/kludt_reproduction/data/kludt_manual")
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
