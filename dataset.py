import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from torchvision import tv_tensors

class VehicleDetectionDataset(Dataset):
    '''Custom PyTorch dataset that returns images and targets from data jsons'''
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data_samples = json.load(f)
        self.transform = transform
        
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data_samples)

    def __getitem__(self, idx):
        # Return the sample at index idx
        image_path = self.data_samples[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")  # Load image in RGB format
        boxes = self.data_samples[idx]["boxes"]
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=image.size[::-1])  # Convert to BoundingBoxes format
        labels = self.data_samples[idx]["labels"]
        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        if self.transform:
            image, target = self.transform(image, target)  # Apply transformations if any

        return image, target