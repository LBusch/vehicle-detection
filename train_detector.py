import torch
from torch.utils.data import DataLoader
from dataset import VehicleDetectionDataset
import models
from plot import plot_results
from torchvision.transforms import v2
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import numpy as np
import os
import json

# training hyperparams
BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 5
BACKBONE_LR = 0.0001
HEAD_LR = 0.0005
WEIGHT_DECAY = 0.0 # 0.0001
RUN_DIR = "frcnn_default"

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.SanitizeBoundingBoxes(),
    v2.ToDtype(torch.float32, scale=True),
])

test_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

def collate_fn(batch):
    return tuple(zip(*batch))

def train(epochs, model, train_loader, test_loader, optimizer, scheduler, metric, run_dir):
    epoch_pbar = tqdm(range(epochs), desc="Epoch", position=0)
    total_losses = []
    class_losses = []
    bbox_losses = []
    rpn_losses = []
    obj_losses = []
    map_scores = []
    best_map = -1.0
    best_epoch = -1
    best_precision_tensor = torch.tensor([-1.0])
    best_score_tensor = torch.tensor([-1.0])

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        mean_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Train Batch", position=1, leave=False)

        # go through each train batch
        for images, targets in train_pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            class_loss = loss_dict['loss_classifier'] # loss_dict['classification']
            bbox_loss = loss_dict['loss_box_reg'] # loss_dict['bbox_regression']
            rpn_loss = loss_dict['loss_rpn_box_reg']
            obj_loss = loss_dict['loss_objectness']
            total_loss = sum(loss for loss in loss_dict.values())
            
            # backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # track losses
            total_loss_value = total_loss.item()
            mean_loss += total_loss_value * len(images)
            total_losses.append(total_loss_value)
            class_losses.append(class_loss.item())
            bbox_losses.append(bbox_loss.item())
            rpn_losses.append(rpn_loss.item())
            obj_losses.append(obj_loss.item())
            train_pbar.set_postfix({"loss": f"{total_loss_value:.4f}"})

        mean_loss /= len(train_loader.dataset)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Test Batch", leave=False, position=1):
                images = list(image.to(device) for image in images)       
                outputs = model(images)
                metric.update(outputs, list(targets))
        
        # compute metrics
        metrics = metric.compute()
        map_50 = metrics['map'].item()
        precision_tensor = metrics['precision'] # Shape: (IoU_Thresh, Recall_Thresh, Classes, Areas, Max_Dets)
        score_tensor = metrics['scores']
        metric.reset()
        map_scores.append(map_50)

        # keep best model and its metrics
        if map_50 > best_map:
            best_map = map_50
            best_precision_tensor = precision_tensor
            best_score_tensor = score_tensor
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

        scheduler.step()
        epoch_pbar.set_postfix({"Mean Loss": f"{mean_loss:.4f}", "mAP": f"{map_50:.4f}"})

    # get metrics to plot precision-recall, f1-confidence curves    
    precision_curve = best_precision_tensor.numpy()[0, :, 0, 0, -1] # Shape: (IoU=0.5, Recall=0.0:1.0, Class=0, Area=all, Max_Dets=100)
    confidence_scores = best_score_tensor.numpy()[0, :, 0, 0, -1]  
    recall_thresholds = np.linspace(0, 1, 101)
    f1_curve = 2 * (precision_curve * recall_thresholds) / (precision_curve + recall_thresholds + 1e-8)
    best_f1_idx = np.argmax(f1_curve)
    best_f1_score = f1_curve[best_f1_idx]
    best_precision = precision_curve[best_f1_idx]
    best_recall = recall_thresholds[best_f1_idx]
    best_score_threshold = confidence_scores[best_f1_idx]
    
    # save metrics and losses
    results = {
        "map_50": best_map,
        "best_f1_score": best_f1_score,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_score_threshold": best_score_threshold,
        "precision_curve": precision_curve.tolist(),
        "f1_curve": f1_curve.tolist(),
        "confidence_scores": confidence_scores.tolist(),
        "map_50_scores": map_scores,
        "total_losses": total_losses,
        "class_losses": class_losses,
        "bbox_losses": bbox_losses,
        "rpn_losses": rpn_losses,
        "obj_losses": obj_losses,
    }
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    plot_results(results, run_dir)
    print("Training complete. Best mAP: {:.4f}, best F1-Score: {:.4f} at score threshold: {:.4f}".format(best_map, best_f1_score, best_score_threshold))
    print(f"Saved best model at epoch {best_epoch+1} and results to {run_dir}")


if __name__ == "__main__":

    train_dataset = VehicleDetectionDataset(json_file="data/train.json", transform=train_transforms)
    test_dataset = VehicleDetectionDataset(json_file="data/test.json", transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.load_pretrained_fasterrcnn(
        num_classes=2, # 1 class + background
        )  
    model = model.to(device)

    # different lrs for each part of the model
    optimizer = torch.optim.SGD([
    {"params": model.backbone.parameters(), "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
    {"params": model.rpn.parameters(), "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
    {"params": model.roi_heads.parameters(), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY}
    ], momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    metric = MeanAveragePrecision(extended_summary=True, iou_thresholds=[0.5])
    os.makedirs(RUN_DIR, exist_ok=True)

    train(epochs=NUM_EPOCHS, model=model, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer, scheduler=scheduler, metric=metric, run_dir=RUN_DIR)