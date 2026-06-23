import matplotlib.pyplot as plt
import json
import numpy as np
import os

def plot_figure(x_data, y_data, x_label, y_label, title, plot_dir):
    '''matplotlib helper function'''
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, color="blue", lw=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(plot_dir, dpi=300, bbox_inches="tight")
    plt.close()


def plot_results(results, save_dir, from_json=False):
    '''Plots a model's metrics and losses from a dict or json'''
    if from_json:  
        with open(results, 'r') as f:
            results_dict = json.load(f)
    else:
        results_dict = results

    precision_curve = results_dict["precision_curve"]
    f1_curve = results_dict["f1_curve"]
    confidence_scores = results_dict["confidence_scores"]
    map_50_scores = results_dict["map_50_scores"]
    total_losses = results_dict["total_losses"]
    class_losses = results_dict["class_losses"]
    bbox_losses = results_dict["bbox_losses"]
    rpn_losses = results_dict["rpn_losses"]
    obj_losses = results_dict["obj_losses"]
    recall_thresholds = np.linspace(0, 1, len(precision_curve))
    batches = range(1, len(total_losses) + 1)
    epochs = range(1, len(map_50_scores) + 1)
    os.path.join(save_dir, "precision_recall.png")

    plot_figure(recall_thresholds, precision_curve, "Recall", "Precision", "Precision-Recall Curve (IoU=0.5)", os.path.join(save_dir, "precision_recall.png"))
    plot_figure(confidence_scores, f1_curve, "Confidence", "F1-Score", "F1-Confidence Curve (IoU=0.5)", os.path.join(save_dir, "f1_confidence.png"))
    plot_figure(epochs, map_50_scores, "Epoch", "mAP", "Mean-Average-Precision (IoU=0.5)", os.path.join(save_dir, "map_50_scores.png"))
    plot_figure(batches, total_losses, "Training Batch", "Total Loss", "Total Train Loss", os.path.join(save_dir, "train_total_loss.png"))
    plot_figure(batches, class_losses, "Training Batch", "Classification Loss", "Train Classification Loss", os.path.join(save_dir, "train_class_loss.png"))
    plot_figure(batches, bbox_losses, "Training Batch", "Bbox Loss", "Train Bbox Loss", os.path.join(save_dir, "train_bbox_loss.png"))
    plot_figure(batches, rpn_losses, "Training Batch", "RPN Bbox Loss", "Train RPN Bbox Loss", os.path.join(save_dir, "train_rpn_bbox_loss.png"))
    plot_figure(batches, obj_losses, "Training Batch", "Objectness Loss", "Train Objectness Loss", os.path.join(save_dir, "train_obj_loss.png"))
