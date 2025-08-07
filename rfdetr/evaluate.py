import os
import json
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


""""
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }

        # sum only relevant losses for logging
        ignored_losses = [
            "abs_distance_error", "abs_heading_diff",
            "abs_distance_error_enc", "abs_heading_diff_enc",
            "class_error", "cardinality_error",
            "class_error_enc", "cardinality_error_enc"
        ]
        training_loss_sum = sum(
            v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict and k not in ignored_losses
        )
        metric_logger.update(
            loss=training_loss_sum,
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        # add extra metrics for logging
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes, args.max_distance)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
"""

if __name__ == "__main__":
    run_path = "../runs/train/DetDistHead_Freeze2/"
    plot_training_logs(log_file_path= run_path + "log.txt", output_path=run_path+ "training_plot.png", use_ema=True)
