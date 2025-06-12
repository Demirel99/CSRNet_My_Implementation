# train.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
import os

import config
from model import CSRNet
from dataset import CrowdDataset
from loss import loss_fn

def main():
    """Main training loop."""
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and DataLoaders for ShanghaiTech Part A
    # The paper uses Part_A for validation during training. We will use the test set as validation.
    train_dataset = CrowdDataset(root_path='part_A_final', phase='train', transform=transform)
    val_dataset = CrowdDataset(root_path='part_A_final', phase='test', transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Model, Optimizer, Scheduler ---
    model = CSRNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.95,
        weight_decay=5e-4
    )
    # A learning rate scheduler can be added for better convergence, but we'll stick
    # to the paper's fixed LR for simplicity.
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # --- Load Checkpoint if specified ---
    if config.LOAD_MODEL and os.path.exists(config.MODEL_CHECKPOINT):
        print(f"Loading checkpoint: {config.MODEL_CHECKPOINT}")
        checkpoint = torch.load(config.MODEL_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mae = checkpoint.get('best_mae', float('inf'))
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_mae = float('inf')
        print("Starting training from scratch.")

    # --- Training Loop ---
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        
        # Train one epoch
        train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate one epoch
        current_mae = validate_one_epoch(model, val_loader, device)

        # Update learning rate scheduler if used
        # scheduler.step()

        # Save the model if it's the best one so far
        is_best = current_mae < best_mae
        if is_best:
            best_mae = current_mae
            print(f"🎉 New best MAE: {best_mae:.2f}. Saving model...")
            if config.SAVE_MODEL:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mae': best_mae,
                }
                torch.save(checkpoint, config.MODEL_CHECKPOINT)

def train_one_epoch(model, data_loader, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm.tqdm(data_loader, desc="Training", unit="batch")

    for images, density_maps, _ in progress_bar:
        images = images.to(device)
        density_maps = density_maps.to(device)

        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = loss_fn(outputs, density_maps)
        
        # The paper's loss is sum of squared errors, so we divide by batch size
        loss = loss / config.BATCH_SIZE

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")

def validate_one_epoch(model, data_loader, device):
    """Validates the model for one epoch and returns MAE."""
    model.eval()
    mae = 0
    progress_bar = tqdm.tqdm(data_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for images, _, gt_counts in progress_bar:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # The network output is a density map. Summing over the map gives the count.
            pred_count = torch.sum(outputs).item()
            
            # Accumulate absolute error
            mae += abs(pred_count - gt_counts.item())
            progress_bar.set_postfix(mae=f"{mae / (progress_bar.n + 1):.2f}")

    avg_mae = mae / len(data_loader)
    print(f"Validation MAE: {avg_mae:.2f}")
    return avg_mae

if __name__ == "__main__":
    main()