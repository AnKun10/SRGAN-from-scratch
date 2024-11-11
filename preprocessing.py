import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from collections import defaultdict
import random


def create_imagenet_dataloader(
        root_dir,
        batch_size=32,
        images_per_class=100,
        num_workers=4,
        shuffle=True
):
    # Define the standard ImageNet transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load the full ImageNet dataset
    full_dataset = datasets.ImageNet(
        root=root_dir,
        split='train',
        transform=transform
    )

    # Create a dictionary to store indices for each class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset):
        class_indices[label].append(idx)

    # Select 100 random images from each class
    selected_indices = []
    for label in class_indices:
        indices = class_indices[label]
        if len(indices) > images_per_class:
            selected_indices.extend(random.sample(indices, images_per_class))
        else:
            selected_indices.extend(indices)

    # Create a subset of the dataset with selected indices
    subset_dataset = Subset(full_dataset, selected_indices)

    # Create the DataLoader
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    # Replace with your ImageNet dataset path
    imagenet_path = "/path/to/imagenet"

    dataloader = create_imagenet_dataloader(
        root_dir=imagenet_path,
        batch_size=32,
        images_per_class=100
    )

    # Print dataset information
    print(f"Total number of batches: {len(dataloader)}")
    print(f"Total number of images: {len(dataloader.dataset)}")
