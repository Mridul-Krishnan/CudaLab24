import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress visualization
import os
import cv2
from PIL import Image
from datasets import CityscapesDataset
from utils.augmentation import get_augmentations
from models.depth_estimation import DepthEstimationModel
from models.ego_estimation import EgoMotionModel
from models.flow_estimation import OpticalFlowModel
from utils.loss_functions2 import photometric_loss, regularization_loss, compute_total_loss
from utils.model_utils import save_checkpoint, load_checkpoint
from utils.visuals import *

def get_training_loaders(train_dir, val_dir, batch_size=8, num_workers=4):
    # Create training dataset and dataloader
    train_dataset = CityscapesDataset(root_dir=train_dir, transform=get_augmentations())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    # Create validation dataset and dataloader
    val_dataset = CityscapesDataset(root_dir=val_dir, transform=get_augmentations())
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


import os
import torch
from tqdm import tqdm

def train_model(depth_model, ego_model, flow_model, train_loader, val_loader, num_epochs, device, intrinsic_matrix, checkpoint_dir="checkpoints", vis_seq = []):
    depth_model.train()
    ego_model.train()
    flow_model.train()

    # Define optimizers
    optimizer_depth = torch.optim.Adam(depth_model.parameters(), lr=1e-4)
    optimizer_ego = torch.optim.Adam(ego_model.parameters(), lr=1e-4)
    optimizer_flow = torch.optim.Adam(flow_model.parameters(), lr=1e-4)

    # Define learning rate schedulers
    scheduler_depth = torch.optim.lr_scheduler.StepLR(optimizer_depth, step_size=10, gamma=0.1)
    scheduler_ego = torch.optim.lr_scheduler.StepLR(optimizer_ego, step_size=10, gamma=0.1)
    scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        # Training phase
        depth_model.train()
        ego_model.train()
        flow_model.train()
        running_loss = 0.0

        # Progress bar for training
        train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Training)", leave=False)

        for batch_idx, batch in enumerate(train_loader_iter):
            images = batch[0].to(device)  # Source images
            target_images = batch[1].to(device)  # Target images (next frames)

            optimizer_depth.zero_grad()
            optimizer_ego.zero_grad()
            optimizer_flow.zero_grad()

            # Forward pass for depth, ego-motion, and flow models
            pred_depth = depth_model(images)
            ego_motion = ego_model(torch.cat([images, target_images], dim=1))  # Input 2 stacked frames for ego-motion
            pred_flow = flow_model(images)

            # Compute total loss
            total_loss = compute_total_loss(pred_depth, pred_flow, images, target_images, ego_motion, intrinsic_matrix)

            # Backward pass and optimization
            total_loss.backward()
            optimizer_depth.step()
            optimizer_ego.step()
            optimizer_flow.step()

            running_loss += total_loss.item()

            # Update tqdm progress bar with current loss
            train_loader_iter.set_postfix(loss=total_loss.item())
            break

        # Validation phase
        depth_model.eval()
        ego_model.eval()
        flow_model.eval()
        val_loss = 0.0

        # Progress bar for validation
        val_loader_iter = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Validation)", leave=False)

        with torch.no_grad():
            for batch in val_loader_iter:
                images = batch[0].to(device)  # Source images
                target_images = batch[1].to(device)  # Target images (next frames)

                # Forward pass
                pred_depth = depth_model(images)
                ego_motion = ego_model(torch.cat([images, target_images], dim=1))  # Input 2 stacked frames for ego-motion
                pred_flow = flow_model(images)

                # Compute validation loss
                val_loss += compute_total_loss(pred_depth, pred_flow, images, target_images, ego_motion, intrinsic_matrix).item()
                break

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # Learning rate scheduler step
        scheduler_depth.step()
        scheduler_ego.step()
        scheduler_flow.step()

        with torch.no_grad():
            depth_sequence = [depth_model(frame.unsqueeze()) for frame in vis_seq]
            ego_motion_sequence = [ego_model(frame.unsqueeze()) for frame in vis_seq]
            flow_sequence = [flow_model(frame.unsqueeze()) for frame in vis_seq]
        # Create GIFs for each visualization
        create_depth_gif(depth_sequence, output_file=os.path.join(checkpoint_dir,f"depth_map{epoch}.gif"))
        create_ego_motion_gif(ego_motion_sequence, output_file=os.path.join(checkpoint_dir,f"ego_motion{epoch}.gif"))
        create_optical_flow_gif(flow_sequence, output_file=os.path.join(checkpoint_dir,f"depth_map{epoch}.gif"))

        # Save model checkpoints
        torch.save(depth_model.state_dict(), os.path.join(checkpoint_dir, f'depth_model_epoch_{epoch+1}.pth'))
        torch.save(ego_model.state_dict(), os.path.join(checkpoint_dir, f'ego_model_epoch_{epoch+1}.pth'))
        torch.save(flow_model.state_dict(), os.path.join(checkpoint_dir, f'flow_model_epoch_{epoch+1}.pth'))

        if (epoch + 1) % 1 == 0:  # Save additional checkpoint every 10 epochs
            torch.save({
                'epoch': epoch + 1,
                'depth_model_state_dict': depth_model.state_dict(),
                'ego_model_state_dict': ego_model.state_dict(),
                'flow_model_state_dict': flow_model.state_dict(),
                'optimizer_depth_state_dict': optimizer_depth.state_dict(),
                'optimizer_ego_state_dict': optimizer_ego.state_dict(),
                'optimizer_flow_state_dict': optimizer_flow.state_dict(),
                'scheduler_depth_state_dict': scheduler_depth.state_dict(),
                'scheduler_ego_state_dict': scheduler_ego.state_dict(),
                'scheduler_flow_state_dict': scheduler_flow.state_dict(),
                'loss': running_loss / len(train_loader)
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth.tar'))
        

    print('Training completed.')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dir = '/home/user/krishnanm0/data/cityscape/train'
    val_dir = '/home/user/krishnanm0/data/cityscape/val'

    train_loader, val_loader = get_training_loaders(train_dir, val_dir)
    vis_seqpath = []
    for city_folder in sorted(os.listdir(val_dir)):
            city_path = os.path.join(val_dir, city_folder)
            if os.path.isdir(city_path):
                frames = sorted(os.listdir(city_path))
                for i in range(len(frames) - 1):
                    vis_seqpath.append(os.path.join(city_path, frames[i]))
                break
    vis_seq = []
    for path in vis_seqpath:
        # Load the images
        print(path)
        frame1 = cv2.imread(path)
        
        # Convert from BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL images for PyTorch transforms
        frame1 = Image.fromarray(frame1)

        # Apply any preprocessing or augmentation
        transform = get_augmentations()
        if transform:
            frame1 = transform(frame1)
        vis_seq.append(frame1)
            

    intrinsic_matrix = torch.eye(3).to(device)

    depth_model = DepthEstimationModel().to(device)
    ego_model = EgoMotionModel().to(device)
    flow_model = OpticalFlowModel().to(device)

    train_model(depth_model, ego_model, flow_model, train_loader, val_loader, num_epochs=10, device=device, intrinsic_matrix=intrinsic_matrix, checkpoint_dir='/home/user/krishnanm0/project_checkpoints', vis_seq=vis_seq)
