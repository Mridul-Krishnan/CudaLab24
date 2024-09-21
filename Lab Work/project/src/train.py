import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress visualization
import os
from datasets import CityscapesDataset
from utils.augmentation import get_augmentations
from models.depth_estimation import DepthEstimationModel
from utils.loss_functions import photometric_loss, regularization_loss, compute_total_loss
from utils.model_utils import save_checkpoint, load_checkpoint

def get_training_loaders(train_dir, val_dir, batch_size=8, num_workers=4):
    # Create training dataset and dataloader
    train_dataset = CityscapesDataset(root_dir=train_dir, transform=get_augmentations())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    # Create validation dataset and dataloader
    val_dataset = CityscapesDataset(root_dir=val_dir, transform=get_augmentations())
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=20, checkpoint_dir="checkpoints"):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Use tqdm for progress visualization in training
        train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Training)", leave=False)
        
        for batch_idx, batch in enumerate(train_loader_iter):
            images = batch[0].to(device)  # Get images from batch
            target_depths = batch[1].to(device)  # Get target depths from batch

            optimizer.zero_grad()

            # Forward pass
            pred_depth = model(images)

            # Compute total loss
            loss = photometric_loss(pred_depth, target_depths)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update tqdm progress bar with current loss
            train_loader_iter.set_postfix(loss=loss.item())

        # Validation loop (wrapped in tqdm)
        model.eval()
        val_loss = 0.0
        val_loader_iter = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Validation)", leave=False)
        
        with torch.no_grad():
            for batch in val_loader_iter:
                images = batch[0].to(device)
                target_depths = batch[1].to(device)

                # Forward pass
                pred_depth = model(images)

                # Compute validation loss
                val_loss += photometric_loss(pred_depth, target_depths).item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # Step 5: Update Learning Rate
        scheduler.step()

        # Step 6: Save Checkpoints
        torch.save(model.state_dict(), os.path.join(checkpoint_dir,'depth_estimation_epoch_{epoch+1}.pth'))

        if (epoch + 1) % 10 == 0:  # Save checkpoint every 10 epochs
            save_checkpoint(
                model, optimizer, scheduler, epoch, running_loss / len(train_loader),
                checkpoint_dir="checkpoints", filename=os.path.join(checkpoint_dir,'depth_estimation_epoch_{epoch+1}.pth.tar')
            )

    print('Training completed.')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dir = '/home/user/krishnanm0/data/cityscape/train'
    val_dir = '/home/user/krishnanm0/data/cityscape/val'

    train_loader, val_loader = get_training_loaders(train_dir, val_dir)
    
    model = DepthEstimationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Start training with tqdm progress bars
    train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=10, checkpoint_dir='/home/user/krishnanm0/project_checkpoints')
