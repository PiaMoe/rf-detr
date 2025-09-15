import json
from runpy import run_path
from pathlib import Path
import os
import matplotlib.pyplot as plt
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def plot_training_logs(log_file_path: str, output_path: str = None, use_ema: bool = False):
    epoch_logs = defaultdict(list)

    # --- Einlesen aller Zeilen ---
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                log = json.loads(line)
                epoch = log.get("epoch")
                if epoch is not None:
                    epoch_logs[epoch].append(log)
            except json.JSONDecodeError:
                continue

    # --- Für jede Epoche den letzten Eintrag verwenden ---
    sorted_epochs = sorted(epoch_logs.keys())
    train_loss, train_loss_bb, train_loss_ce, train_loss_giou = [], [], [], []
    test_loss, test_loss_bb, test_loss_ce, test_loss_giou = [], [], [], []
    map_50, map_50_95, recall, precision = [], [], [], []

    for ep in sorted_epochs:
        log = epoch_logs[ep][-1]  # letzter Eintrag der Epoche

        train_loss.append(log.get("train_loss", None))
        train_loss_bb.append(log.get("train_loss_bbox", None))
        train_loss_ce.append(log.get("train_loss_ce", None))
        train_loss_giou.append(log.get("train_loss_giou", None))

        test_loss.append(log.get("ema_test_loss" if use_ema else "test_loss", None))
        test_loss_bb.append(log.get("ema_test_loss_bbox" if use_ema else "test_loss_bbox", None))
        test_loss_ce.append(log.get("ema_test_loss_ce" if use_ema else "test_loss_ce", None))
        test_loss_giou.append(log.get("ema_test_loss_giou" if use_ema else "test_loss_giou", None))

        results = log.get("ema_test_results_json" if use_ema else "test_results_json", {})
        map_50.append(results.get("map", None))
        recall.append(results.get("recall", None))
        precision.append(results.get("precision", None))
        class_results = results.get("class_map", {})
        class_all = class_results[-1]
        map_50_95.append(class_all.get("map@50:95", None))

    # --- Plot ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(sorted_epochs, train_loss, label="Train Loss")
    axs[0].plot(sorted_epochs, train_loss_bb, label="Train Loss BBox")
    axs[0].plot(sorted_epochs, train_loss_ce, label="Train Loss CE")
    axs[0].plot(sorted_epochs, train_loss_giou, label="Train Loss GIoU")
    axs[0].set_ylabel("Train Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sorted_epochs, test_loss, label="Test Loss" + (" (EMA)" if use_ema else ""))
    axs[1].plot(sorted_epochs, test_loss_bb, label="Test Loss BBox" + (" (EMA)" if use_ema else ""))
    axs[1].plot(sorted_epochs, test_loss_ce, label="Test Loss CE" + (" (EMA)" if use_ema else ""))
    axs[1].plot(sorted_epochs, test_loss_giou, label="Test Loss GIoU" + (" (EMA)" if use_ema else ""))
    axs[1].set_ylabel("Test Loss")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(sorted_epochs, map_50, label="mAP@50")
    axs[2].plot(sorted_epochs, map_50_95, label="mAP@50:95")
    axs[2].plot(sorted_epochs, precision, label="Precision")
    axs[2].plot(sorted_epochs, recall, label="Recall")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Test Metrics")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Plot gespeichert unter: {output_path}")
    else:
        plt.show()


def predictions_to_coco(pred_dir, gt_file):
    out_file = pred_dir / "predictions.json"

    cocoGt = COCO(gt_file)

    # Mapping file_name -> image_id
    file_to_id = {img["file_name"]: img["id"] for img in cocoGt.dataset["images"]}
    class_map = {
        "boat": 0
    }

    coco_preds = []

    for pred_file in pred_dir.glob("*.json"):
        with open(pred_file, "r") as f:
            preds = json.load(f)

        if isinstance(preds, dict):
            preds = [preds]

        # image_id from file_name
        file_name = pred_file.name
        file_name = file_name.rsplit(".", 1)[0] + ".jpg"  # .json -> .jpg
        if file_name not in file_to_id:
            print(f"⚠️ Kein Mapping für {file_name} gefunden – überspringe.")
            continue
        image_id = file_to_id[file_name]

        for p in preds:
            x1, y1, x2, y2 = p["bbox"]
            w, h = x2 - x1, y2 - y1
            coco_preds.append({
                "image_id": image_id,
                "file_name": file_name,
                "category_id": class_map[p["class_name"]],
                "bbox": [x1, y1, w, h],
                "score": float(p["confidence"])
            })

    # all predictions in one file
    with open(out_file, "w") as f:
        json.dump(coco_preds, f)

    print(f"COCO predictions saved in {out_file}, {len(coco_preds)} detections over all.")

def evaluate_det(gt, pred):

    cocoGt = COCO(gt)
    cocoDt = cocoGt.loadRes(pred)

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    #run_path = "../runs/train/BOArDING_Det/"
    #plot_training_logs(log_file_path= run_path + "log.txt", output_path=run_path+ "training_plot.png", use_ema=True)
    pred_dir = Path("../runs/detect/B3_Det/labels")
    gt_path = "../../../data/BOArDING_3/Det/val/val.json"
    predictions_to_coco(pred_dir, gt_path)
    evaluate_det(gt_path, str(pred_dir) + "/predictions.json")

