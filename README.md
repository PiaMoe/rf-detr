# RF-DETR with distance and heading estimation

This branch is for training an rf-detr detection model including distance and heading predictions. Original model without modifications: branch [detection only](https://github.com/PiaMoe/rf-detr/tree/detection_only)

### Dataset structure

RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: `train`, `val`, and `test`. Each subdirectory should contain its own `.json` file that holds the annotations for that particular split, along with the corresponding image files. For training, only the `train` and `val` split are used. Below is an example of the directory structure:

```
dataset/
├── train/
│   ├── train.json
│   ├── images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ... (other image files)
├── val/
│   ├── val.json
│   ├── images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ... (other image files)
```

To transform YOLO-style labels into COCO-style labels, use the function `yolo_to_coco`in `datasets/custom_data.py`.

## Training
use `config.py`for changing Hyperparameters, model architecture etc. Per default, the base model is used.

**Single GPU training**
``` shell
python3 rfdetr/train.py --data_path "path/to/dataset/" --name "rfetr_extended" --epochs 100
```

**Multi GPU training**
CAUTION: Multi GPU training might not work.
``` shell
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env rfdetr/train.py --data_path "path/to/dataset" --name "rfdetr_extended" --epochs 100
```

Losses and metrics on validation set will be logged with W&B automatically.

## Inference
The trained model can be used as described in [RF-DETR](https://github.com/PiaMoe/rf-detr/tree/detection_only) 
`rf-detr/inference.py` performs inference on a directory containing images or a video file (adjust path to weights and files in main). Detections (incl. distance and headings) are saved.

## Testing
Evaluating the model's performance w.r.t. object detection, distance estimation and heading estimation with `rf-detr/evaluate`: 
- use the saved detection results that are output by `rf-detr/inference.py`
- adjust paths to ground truth labels and predictions in main
- run `rf-detr/evaluate`  

Outputs are the same as for [YOLOv7](https://github.com/PiaMoe/YOLOv7_distance/tree/distance_scHeading) (incl. plots for distance & heading errors and correlations)

Example output:

```

Total Samples: 1493
Overall weighted_rel_dist_err_boat = 0.7404274408521122

Overall abs_mean_dist_err_boat = 83.10846732197568

Distance bins:
  Distance bin (0.0, 100.0):
    samples = 566
    weighted_rel_dist_err_boat = 1.484
    abs_mean_dist_err_boat = 47.347
  Distance bin (100.0, 200.0):
    samples = 337
    weighted_rel_dist_err_boat = 0.342
    abs_mean_dist_err_boat = 71.190
  Distance bin (200.0, 300.0):
    samples = 212
    weighted_rel_dist_err_boat = 0.232
    abs_mean_dist_err_boat = 92.243
  Distance bin (300.0, 400.0):
    samples = 98
    weighted_rel_dist_err_boat = 0.161
    abs_mean_dist_err_boat = 90.195
  Distance bin (400.0, 500.0):
    samples = 82
    weighted_rel_dist_err_boat = 0.155
    abs_mean_dist_err_boat = 111.445
  Distance bin (500.0, 600.0):
    samples = 49
    weighted_rel_dist_err_boat = 0.178
    abs_mean_dist_err_boat = 179.960
  Distance bin (600.0, 700.0):
    samples = 27
    weighted_rel_dist_err_boat = 0.160
    abs_mean_dist_err_boat = 189.963
  Distance bin (700.0, 800.0):
    samples = 37
    weighted_rel_dist_err_boat = 0.160
    abs_mean_dist_err_boat = 202.292
  Distance bin (800.0, 900.0):
    samples = 20
    weighted_rel_dist_err_boat = 0.315
    abs_mean_dist_err_boat = 375.158
  Distance bin (900.0, 1000.0):
    samples = 3
    weighted_rel_dist_err_boat = 0.479
    abs_mean_dist_err_boat = 556.883
    
 Distance bin (900.0, 1000.0):
    samples = 3
    weighted_rel_dist_err_boat = 0.479
    abs_mean_dist_err_boat = 556.883

Mean heading error: 74.1 degrees
Heading precision: 0.308
Correct headings: 345, Incorrect headings: 774

Correlation Matrix:
                confidence  distance error  heading error       IoU
confidence        1.000000        0.004826       0.012272  0.594948
distance error    0.004826        1.000000       0.034124 -0.028553
heading error     0.012272        0.034124       1.000000  0.013214
IoU               0.594948       -0.028553       0.013214  1.000000
```
