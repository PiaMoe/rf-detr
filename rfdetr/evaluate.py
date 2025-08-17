import os
import json
from tkinter import Image

import cv2
from rfdetr import RFDETRBase
import matplotlib.pyplot as plt
from collections import defaultdict
from glob import glob
import numpy as np
from tueplots.constants.color import rgb
from scipy.stats import gaussian_kde


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
    train_loss, train_loss_bb, train_loss_ce, train_loss_giou, train_loss_dist, train_loss_head = [], [], [], [], [], []
    test_loss, test_loss_bb, test_loss_ce, test_loss_giou, test_loss_dist, test_loss_head = [], [], [], [], [], []
    map_50, map_50_95, recall, precision, = [], [], [], []
    abs_distance_error, abs_heading_diff = [], []

    for ep in sorted_epochs:
        log = epoch_logs[ep][-1]  # letzter Eintrag der Epoche

        train_loss.append(log.get("train_loss", None))
        train_loss_bb.append(log.get("train_loss_bbox", None))
        train_loss_ce.append(log.get("train_loss_ce", None))
        train_loss_giou.append(log.get("train_loss_giou", None))
        train_loss_dist.append(log.get("train_loss_distance", None))
        train_loss_head.append(log.get("train_loss_heading", None))

        test_loss.append(log.get("ema_test_loss" if use_ema else "test_loss", None))
        test_loss_bb.append(log.get("ema_test_loss_bbox" if use_ema else "test_loss_bbox", None))
        test_loss_ce.append(log.get("ema_test_loss_ce" if use_ema else "test_loss_ce", None))
        test_loss_giou.append(log.get("ema_test_loss_giou" if use_ema else "test_loss_giou", None))
        test_loss_dist.append(log.get("ema_test_loss_distance" if use_ema else "test_loss_distance", None))
        test_loss_head.append(log.get("ema_test_loss_heading" if use_ema else "test_loss_heading", None))

        results = log.get("ema_test_results_json" if use_ema else "test_results_json", {})
        map_50.append(results.get("map", None))
        recall.append(results.get("recall", None))
        precision.append(results.get("precision", None))
        abs_distance_error.append(log.get("test_abs_distance_error", None))
        abs_heading_diff.append(log.get("test_abs_heading_diff", None))
        class_results = results.get("class_map", {})
        class_all = class_results[-1]
        map_50_95.append(class_all.get("map@50:95", None))

    # --- Plot ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axs[0][0].plot(sorted_epochs, train_loss, label="Train Loss", c="blue")
    axs[0][0].plot(sorted_epochs, train_loss_bb, label="Train Loss BBox", c="orange")
    axs[0][0].plot(sorted_epochs, train_loss_ce, label="Train Loss CE", c="green")
    axs[0][0].plot(sorted_epochs, train_loss_giou, label="Train Loss GIoU", c="red")
    axs[0][0].plot(sorted_epochs, train_loss_dist, label="Train Loss Distance", c="purple")
    axs[0][0].plot(sorted_epochs, train_loss_head, label="Train Loss Heading", c="brown")
    axs[0][0].plot(sorted_epochs, test_loss, label="Test Loss" + (" (EMA)" if use_ema else ""), c="blue", linestyle='--')
    axs[0][0].plot(sorted_epochs, test_loss_bb, label="Test Loss BBox" + (" (EMA)" if use_ema else ""), c="orange", linestyle='--')
    axs[0][0].plot(sorted_epochs, test_loss_ce, label="Test Loss CE" + (" (EMA)" if use_ema else ""), c="green", linestyle='--')
    axs[0][0].plot(sorted_epochs, test_loss_giou, label="Test Loss GIoU" + (" (EMA)" if use_ema else ""), c="red", linestyle='--')
    axs[0][0].plot(sorted_epochs, test_loss_dist, label="Test Loss Distance" + (" (EMA)" if use_ema else ""), c="purple", linestyle='--')
    axs[0][0].plot(sorted_epochs, test_loss_head, label="Test Loss Heading" + (" (EMA)" if use_ema else ""), c="brown", linestyle='--')
    axs[0][0].set_xlabel("Epoch")
    axs[0][0].set_ylabel("Train & Test Loss")
    axs[0][0].legend(loc='upper right', fontsize='small')
    axs[0][0].grid(True)

    axs[0][1].plot(sorted_epochs, map_50, label="mAP@50")
    axs[0][1].plot(sorted_epochs, map_50_95, label="mAP@50:95")
    axs[0][1].plot(sorted_epochs, precision, label="Precision")
    axs[0][1].plot(sorted_epochs, recall, label="Recall")
    axs[0][1].set_xlabel("Epoch")
    axs[0][1].set_ylabel("Test Metrics")
    axs[0][1].legend()
    axs[0][1].grid(True)

    axs[1][0].plot(sorted_epochs, abs_distance_error, label="Abs Distance Error")
    axs[1][0].set_xlabel("Epoch")
    axs[1][0].set_ylabel("Abs Distance Error")

    axs[1][1].plot(sorted_epochs, abs_heading_diff, label="Abs Heading Diff")
    axs[1][1].set_xlabel("Epoch")
    axs[1][1].set_ylabel("Abs Heading Diff")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Plot gespeichert unter: {output_path}")
    else:
        plt.show()


def plot_dist_err(err_results, num_samples=None, labelX='GT - Distance [m]', labelY=r'$\varepsilon$',
                  path='err_plot.png', color='blue'):
    fig, ax = plt.subplots()
    x = [(x[0] + x[1]) / 2 for x in err_results]
    y = list(err_results.values())
    bars = ax.bar(x, y, width=30, color=color)
    # Annotate each bar with the corresponding sample count
    if num_samples is not None:
        for bar, key in zip(bars, num_samples):
            yval = bar.get_height()  # Get the height of the bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval, num_samples[key], ha='center', va='bottom')

    ax.set_ylabel(labelY)
    ax.set_xlabel(labelX)
    plt.savefig(path)


def plot_errors(errors, bins, max_dist, path):
    # group errors into bins
    delta = max_dist / (2 * bins)
    binIdx = [max_dist / bins * x - delta for x in range(1, bins + 1)]
    grouped_data = {k: [] for k in binIdx}
    for gt, error in errors:
        if gt <= max_dist:
            i = np.argmin(list(map(lambda x: abs(gt - x), binIdx)))
            grouped_data[binIdx[i]].append(error)  # append error to bin

    Nmax = np.max([len(grouped_data[k]) for k in grouped_data])
    np.random.seed(1)
    u = np.random.rand(Nmax)

    plotted_errors = np.concatenate(list(grouped_data.values()))

    fig, ax = plt.subplots()

    colorarr = [rgb.tue_blue, rgb.tue_red]

    for count, k in enumerate(grouped_data):
        ax.plot(grouped_data[k], 0.8 * u[:len(grouped_data[k])] + 0.1 + count, 'o', alpha=0.5,
                color=colorarr[count % 2], ms=2, mec='none')
        # perform kernel density estimation
        if len(grouped_data[k]) > 1:
            kde = gaussian_kde(grouped_data[k],
                               bw_method='scott')  # 'scott' or 'silverman' are common choices for bandwidth
            x_values = np.linspace(np.min(plotted_errors), np.max(plotted_errors), 1000)
            kde_values = kde(x_values)
            # normalize kde between 0-1
            kde_values = 0.8 * kde_values / np.max(kde_values) + 0.1 + count
            ax.plot(x_values, kde_values, color=rgb.tue_dark, alpha=0.9, linewidth=0.7, linestyle='dashed')

    ax.set_yticks([x + 0.5 for x in range(0, count + 1)])
    ax.set_yticklabels([f"[{int(k - delta)}, {int(k + delta)})" for k in grouped_data], rotation=90, va="center",
                       fontsize=7)

    for y, n in zip([0 + x for x in range(0, count + 1)], [len(grouped_data[k]) for k in grouped_data]):
        ax.axhline(y, color=rgb.tue_dark, alpha=0.5)
        t = ax.text(np.max(plotted_errors), y + 0.5, str(n).rjust(3, " ") + " samples", color=rgb.tue_dark, va='center',
                    ha='right', fontsize="x-small")
        t.set_bbox(dict(facecolor='white', alpha=1.0, linewidth=0, pad=1.0))

    ax.vlines(x=0, ymin=0, ymax=count + 1, colors=rgb.tue_darkgreen, linestyles='dashed', alpha=1, linewidth=1)
    ax.set_ylabel("Distance Bins")
    ax.set_xlabel("pred - target [m]")
    ax.set_ylim(0, count + 1)

    ax.set_title("Distance Prediction Errors")

    plt.savefig(path)


def plot_dist_pred(data, path):
    fig, ax = plt.subplots()
    for x in data:
        if x[0] < x[1]:
            ax.plot(x[0], x[1], 'o', alpha=0.6, color=rgb.tue_blue, markersize=3, mec='none')
        else:
            ax.plot(x[0], x[1], 'o', alpha=0.6, color=rgb.tue_red, markersize=3, mec='none')
    data = np.asarray(data)
    ax.plot([0, np.max(data[:, 0])], [0, np.max(data[:, 0])], linestyle='--', color=rgb.tue_darkgreen,
            alpha=1)
    ax.set_ylabel("Prediction [m]")
    ax.set_xlabel("Ground Truth Distance [m]")
    plt.savefig(path)


def plot_heading_pred(data, path):
    fig, ax = plt.subplots()
    for x in data:
        # Compute angular error (in degrees)
        gt = x[0]
        pred = x[1]
        diff_deg = min(abs(gt - pred), 360 - abs(gt - pred))

        # Color based on error
        color = rgb.tue_blue if diff_deg < 30 else rgb.tue_red
        ax.plot(gt, pred, 'o', alpha=0.6, color=color, markersize=3, mec='none')

    data = np.asarray(data)
    ax.plot([0, 360], [0, 360], linestyle='--', color=rgb.tue_darkgreen, alpha=1)

    ax.set_ylabel("Predicted Heading")
    ax.set_xlabel("Ground Truth Heading")

    plt.savefig(path)


def plot_heading_err(data, path):
    data = np.asarray(data)  # shape (N, 2): [gt_heading, pred_heading], both in [0, 1)
    gt = data[:, 0]
    pred = data[:, 1]

    # Compute angular error
    diff = np.abs(gt - pred)
    angular_error = np.minimum(diff, 360 - diff)

    # Define bins (8 bins = 45° each)
    bins = np.linspace(0, 360, 9)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_errors = []
    sample_counts = []

    for i in range(8):
        mask = (gt >= bins[i]) & (gt < bins[i + 1])
        count = np.sum(mask)
        sample_counts.append(count)
        if count > 0:
            mean_error = angular_error[mask].mean()
        else:
            mean_error = 0
        bin_errors.append(mean_error)

    # Plot
    plt.figure()
    bars = plt.bar(bin_centers, bin_errors, width=30, align='center', color='darkgreen')
    plt.xlabel("Ground Truth Heading (deg)")
    plt.ylabel("Mean Angular Error (deg)")
    plt.xticks(bins)

    # Sample count über jedem Balken anzeigen
    for bar, count in zip(bars, sample_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, str(count),
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(path)



import numpy as np

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def yolo_to_xyxy(label, img_w, img_h):
    # YOLO: class cx cy w h distance heading
    cls, cx, cy, w, h, dist, head = label
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2], dist, head

def create_distance_bins(max_distance, number_bins):
    # Calculate the width of each bin
    bin_width = max_distance / number_bins

    # Create the bins
    distance_bins = [(i * bin_width, (i + 1) * bin_width) for i in range(number_bins)]

    return distance_bins



def evaluate(gt_dir, predictions_dir, max_distance=1000, num_bins=10, iou_thresh=0.5):
    gt_images = gt_dir + "/images/"
    gt_labels = gt_dir + "/labels/"
    save_dir = os.path.join(predictions_dir, "..", "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)

    dist_pred_and_gt = []
    head_pred_and_gt = []
    dist_errors_plot = []

    # bins setup
    distance_bins = create_distance_bins(max_distance, num_bins)

    samples_per_bin = {interval: 0 for interval in distance_bins}
    mean_abs_dist_err_boat_bins = {interval: [] for interval in distance_bins}

    # prepare GT boxes
    gt_data = []

    for pred in sorted(os.listdir(predictions_dir)):
        pred_file = os.path.join(predictions_dir, pred)
        gt_img_file = os.path.join(gt_images, os.path.basename(pred).replace('.json', '.jpg'))
        gt_label_file = os.path.join(gt_labels, os.path.basename(pred).replace('.json', '.txt'))
        if not os.path.exists(gt_img_file) or not os.path.exists(gt_label_file):
            print(f"GT file for {pred} not found. Skipping.")
            continue
        img_height, img_width = cv2.imread(gt_img_file).shape[:2]

        with open(gt_label_file, 'r') as f:
            gt = f.readlines()

        with open(pred_file, 'r') as f:
            predictions = json.load(f)
        for line in gt:
            parts = line.strip().split()
            vals = list(map(float, parts))
            box, dist, head = yolo_to_xyxy(vals, img_width, img_height)
            gt_data.append({"bbox": box, "distance": dist, "heading": head, "used": False})

        # match predictions
        for pred in predictions:
            pbox = pred["bbox"]  # already xyxy
            pdist = pred["distance"]
            phead = pred["heading"]

            best_iou = 0
            best_gt = None
            for g in gt_data:
                if g["used"]:
                    continue
                i = iou(pbox, g["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_gt = g

            if best_gt is not None and best_iou >= iou_thresh:
                best_gt["used"] = True

                gdist, ghead = best_gt["distance"], best_gt["heading"]

                # 1. speichern
                if ghead != -1:  # heading vorhanden
                    head_pred_and_gt.append((phead, ghead))

                if gdist != -1:  # distance vorhanden
                    dist_pred_and_gt.append((pdist, gdist))

                    # 2. distance error plot
                    dist_errors_plot.append((gdist, pdist - gdist))

                    # 3+4. Bin stats
                    for interval in distance_bins:
                        low, high = interval
                        if low <= gdist < high:
                            samples_per_bin[interval] += 1
                            mean_abs_dist_err_boat_bins[interval].append(abs(pdist - gdist))
                            break

    mean_abs_dist_err_boat_bins = {
        interval: (np.mean(errors) if errors else 0.0)
        for interval, errors in mean_abs_dist_err_boat_bins.items()
    }

    plot_dist_err(mean_abs_dist_err_boat_bins, num_samples=samples_per_bin, labelX='GT - Distance [m]',
                  labelY=r'$\varepsilon$', path=os.path.join(save_dir, 'AbsoluteError.png'), color='red')
    # TODO Pia: relative error plot
    # plot raw dist errors
    plot_errors(dist_errors_plot, bins=5, max_dist=distance_bins[-1][1],
                path=os.path.join(save_dir, 'dist_errors.pdf'))
    plot_dist_pred(dist_pred_and_gt, path=os.path.join(save_dir, 'dist_pred.pdf'))

    # plot heading errors
    plot_heading_pred(head_pred_and_gt, path=os.path.join(save_dir, 'head_pred.pdf'))
    plot_heading_err(head_pred_and_gt, path=os.path.join(save_dir, 'head_err.pdf'))



if __name__ == "__main__":
    #run_path = "../runs/train/DetDistHead_MLP/"
    #plot_training_logs(log_file_path= run_path + "log.txt", output_path=run_path+ "training_plot.png", use_ema=True)
    evaluate(
        gt_dir="../../../data/BOArDING_Dataset/BOArDING/val",
        predictions_dir="../runs/detect/DetDistHead_MLP/labels/",
        max_distance=1000,
        num_bins=10
    )
