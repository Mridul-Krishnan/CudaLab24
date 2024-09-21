import os
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir="checkpoints", filename="checkpoint.pth.tar"):
    """Saves the model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))
    print(f"Checkpoint saved at {os.path.join(checkpoint_dir, filename)}")


def load_checkpoint(model, optimizer, scheduler=None, checkpoint_path="checkpoints/checkpoint.pth.tar"):
    """Loads the model checkpoint and returns the epoch and loss."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded. Resuming from epoch {epoch} with loss {loss:.4f}.")
    
    return epoch, loss
