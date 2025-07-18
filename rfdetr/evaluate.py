import json
from runpy import run_path

import matplotlib.pyplot as plt
from collections import defaultdict

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
    train_loss, test_loss = [], []
    map_50, recall, precision = [], [], []

    for ep in sorted_epochs:
        log = epoch_logs[ep][-1]  # letzter Eintrag der Epoche

        train_loss.append(log.get("train_loss", None))
        test_loss.append(log.get("ema_test_loss" if use_ema else "test_loss", None))

        results = log.get("ema_test_results_json" if use_ema else "test_results_json", {})
        map_50.append(results.get("map", None))
        recall.append(results.get("recall", None))
        precision.append(results.get("precision", None))

    # --- Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(sorted_epochs, train_loss, label="Train Loss")
    axs[0].plot(sorted_epochs, test_loss, label="Test Loss" + (" (EMA)" if use_ema else ""))
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sorted_epochs, map_50, label="mAP@50")
    axs[1].plot(sorted_epochs, precision, label="Precision")
    axs[1].plot(sorted_epochs, recall, label="Recall")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Test Metrics")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✅ Plot gespeichert unter: {output_path}")
    else:
        plt.show()



if __name__ == "__main__":
    run_path = "../runs/train/test_run/"
    plot_training_logs(log_file_path= run_path + "log.txt", output_path=run_path+ "training_plot.png", use_ema=True)
