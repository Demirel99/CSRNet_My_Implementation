# dataset.py

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import random

import config

class CrowdDataset(Dataset):
    """
    Custom PyTorch Dataset for loading crowd counting data.
    It generates density maps from point annotations on-the-fly.
    """

    def __init__(self, root_path, phase='train', transform=None):
        self.root_path = os.path.join(config.DATASET_BASE_PATH, root_path)
        self.phase = phase
        self.transform = transform
        data_folder = 'train_data' if phase == 'train' else 'test_data'
        self.img_dir = os.path.join(self.root_path, data_folder, 'images')
        self.gt_dir = os.path.join(self.root_path, data_folder, 'ground_truth')
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        gt_name = 'GT_' + img_name.replace('.jpg', '.mat')
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = loadmat(gt_path)
        gt_points = mat['image_info'][0, 0][0, 0][0]

        if self.phase == 'train':
            # --- Training Phase: Random Cropping & Flipping ---
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                gt_points[:, 0] = image.width - gt_points[:, 0]
            
            w, h = image.size
            crop_w = int(w / 2)
            crop_h = int(h / 2)
            crop_w = crop_w - (crop_w % config.DOWNSAMPLE_RATIO)
            crop_h = crop_h - (crop_h % config.DOWNSAMPLE_RATIO)

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            
            image = image.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            
            gt_points_cropped = []
            for pt in gt_points:
                if x1 <= pt[0] < x1 + crop_w and y1 <= pt[1] < y1 + crop_h:
                    gt_points_cropped.append([pt[0] - x1, pt[1] - y1])
            gt_points = np.array(gt_points_cropped)
        else:
            # --- Validation/Test Phase: Resize to be divisible by 8 ---
            w, h = image.size
            new_w = w - (w % config.DOWNSAMPLE_RATIO)
            new_h = h - (h % config.DOWNSAMPLE_RATIO)
            
            # Resize image
            image = image.resize((new_w, new_h), Image.BILINEAR)
            
            # Scale ground truth points
            # Only do this if there are points to avoid errors with empty arrays
            if gt_points.shape[0] > 0:
                gt_points[:, 0] = gt_points[:, 0] * (new_w / w)
                gt_points[:, 1] = gt_points[:, 1] * (new_h / h)
            
        # --- Common processing for both phases ---
        density_map = self.generate_density_map(image.size, gt_points)
        
        ds_rows = int(density_map.shape[0] // config.DOWNSAMPLE_RATIO)
        ds_cols = int(density_map.shape[1] // config.DOWNSAMPLE_RATIO)
        density_map = density_map.reshape(
            (ds_rows, config.DOWNSAMPLE_RATIO, ds_cols, config.DOWNSAMPLE_RATIO)
        ).sum(axis=(1, 3))

        gt_count = len(gt_points)

        if self.transform:
            image = self.transform(image)
            
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)
        
        return image, density_map, gt_count

    def generate_density_map(self, img_size, gt_points):
        width, height = img_size
        density_map = np.zeros((height, width), dtype=np.float32)
        
        if len(gt_points) == 0:
            return density_map

        if len(gt_points) > config.K_NEAREST:
            tree = KDTree(gt_points.copy(), leafsize=1024)
            distances, _ = tree.query(gt_points, k=config.K_NEAREST + 1)

        for i, pt in enumerate(gt_points):
            pt_map = np.zeros((height, width), dtype=np.float32)
            
            x = min(width - 1, max(0, int(pt[0])))
            y = min(height - 1, max(0, int(pt[1])))
            
            if x >= width or y >= height:
                continue

            pt_map[y, x] = 1.0

            if config.SIGMA_METHOD == 'adaptive':
                if len(gt_points) > config.K_NEAREST:
                    avg_distance = np.mean(distances[i, 1:config.K_NEAREST + 1])
                    sigma = config.BETA * avg_distance
                else:
                    sigma = 15.0
            else:
                sigma = config.FIXED_SIGMA

            density_map += gaussian_filter(pt_map, sigma, mode='constant')

        return density_map

if __name__ == '__main__':
    # --- Example Usage ---
    from torchvision import transforms
    
    # Define transforms (Normalization is crucial)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset instance for ShanghaiTech Part A training set
    train_dataset = CrowdDataset(root_path='part_A_final', phase='train', transform=transform)
    
    # Get one sample
    image, density_map, gt_count = train_dataset[0]
    
    print("--- Dataset Sample ---")
    print(f"Image shape: {image.shape}")
    print(f"Density map shape: {density_map.shape}")
    print(f"Ground truth count: {gt_count}")
    print(f"Sum of density map (should be close to gt_count): {torch.sum(density_map):.2f}")
    
    # Note: Sum of downsampled density map is not exactly gt_count.
    # The sum of the *original* high-res density map is ~gt_count.
    # The downsampling factor (8*8=64) changes the sum. The loss function
    # will compare the network's output to this downsampled map directly.