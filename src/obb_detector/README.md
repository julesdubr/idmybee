# Bombus forewing OBB detector

Pipeline based on a small Ultralytics YOLO OBB model trained with PyTorch.

## 1. Environment

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -U pip
pip install torch torchvision ultralytics pandas numpy pillow opencv-python
```

Check the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Build the dataset

Only rows with `status == OK` are used. The split is by `specimen_id`, not by image.

```bash
python prepare_dataset.py \
  --images-csv /path/to/images.csv \
  --crops-csv /path/to/crops.csv \
  --out /path/to/bombus_obb_dataset \
  --seed 42
```

If source images are on the same disk and hardlinks are available:

```bash
python prepare_dataset.py ... --copy-mode hardlink
```

The script produces:

```text
bombus_obb_dataset/
  dataset.yaml
  manifest.csv
  split_stats.csv
  images/{train,val,test}/...
  labels/{train,val,test}/...
```

The labels are in the YOLO OBB format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

with normalized corner coordinates.

## 3. Train

First experiment:

```bash
python train.py \
  --data /path/to/bombus_obb_dataset/dataset.yaml \
  --model yolo26n-obb.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch -1 \
  --device 0
```

For a quick baseline use `--imgsz 640`. For better small-wing localization try `1024` and `1280`.

The first serious comparison should be:

```text
imgsz = 640 / 1024 / 1280
```

with identical specimen-level splits.

## 4. Evaluate

```bash
python evaluate.py \
  --model runs/bombus_obb/yolo26n_1024/weights/best.pt \
  --data /path/to/bombus_obb_dataset/dataset.yaml \
  --split test \
  --imgsz 1024 \
  --device 0
```

## 5. Inference and crop

```bash
python predict.py \
  --model runs/bombus_obb/yolo26n_1024/weights/best.pt \
  --image /path/to/new_photo.jpg \
  --out /path/to/predictions \
  --imgsz 1024 \
  --conf 0.45
```

The predictor takes the highest-confidence OBB, rectifies it and writes a normalized wing crop.

## 6. Suggested experiment matrix

Do not tune dozens of hyperparameters first. Start with:

1. YOLO26n-OBB, 640 px
2. YOLO26n-OBB, 1024 px
3. YOLO26n-OBB, 1280 px

Then evaluate precision, recall, mAP50-95, OBB localization quality and, most importantly, downstream landmark/GPA classification performance.

## 7. Important scientific split

Never put different photographs of the same `specimen_id` into different train/val/test sets. The supplied `prepare_dataset.py` prevents this leakage.

## 8. Licensing

Ultralytics currently offers AGPL-3.0 and an Enterprise license. Check the applicable license before integrating the trained model into a closed or commercial mobile/web product.
