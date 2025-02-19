import os
import torch
import wandb
from datetime import datetime
import logging



class Logger:
    def __init__(self, use_wandb=False, project_name="image-captioning"):
        """Initialize logger with option for wandb tracking."""
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name)
        
        # Create timestamp for unique run identification
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for checkpoints
        self.checkpoint_dir = f"checkpoints_{self.timestamp}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def log_hyperparameters(self, config):
        """Log training hyperparameters."""
        if self.use_wandb:
            wandb.config.update(config)
        
        # Also save hyperparameters locally
        print("\nTraining Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")
        print("\n")

    def log_training_step(self, loss, epoch, batch_idx, phase="train", global_step=None):
        """Log metrics for a single training/validation step.
        
        Args:
            loss: The loss value to log
            epoch: The current epoch (1-based)
            batch_idx: The current batch index
            phase: Either 'train' or 'val'
            global_step: Optional global step counter
        """
        if self.use_wandb:
            # Calculate global step if not provided
            if global_step is None:
                global_step = (epoch - 1) * 1000 + batch_idx  # Adjust for 1-based epoch
                
            # Log with step to ensure proper ordering
            wandb.log({
                f"{phase}/step_loss": loss,
                f"{phase}/epoch": epoch,  # Keep as integer
                f"{phase}/batch": batch_idx,
                "epoch": epoch,  # Keep as integer
                "batch": batch_idx,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"] if 'optimizer' in locals() else wandb.run.summary.get("learning_rate", 0)
            }, step=global_step)
        
        return {
            'loss': f'{loss:.4f}',
            'epoch': epoch
        }

    def log_epoch_metrics(self, train_loss, val_loss, epoch):
        """Log metrics at the end of each epoch."""
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Training Loss: {train_loss:.4f}")
        print(f"Average Validation Loss: {val_loss:.4f}")
        
        if self.use_wandb:
            # Calculate global step for epoch end
            global_step = epoch * 1000  # Use end of epoch step
            
            # Log both losses together for easier comparison in WandB
            wandb.log({
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "epoch": epoch,  # Keep as integer
                "train/learning_rate": optimizer.param_groups[0]["lr"] if 'optimizer' in locals() else wandb.run.summary.get("learning_rate", 0)
            }, step=global_step)
            
            # Update best metrics
            if val_loss < wandb.run.summary.get("best_val_loss", float("inf")):
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"] = epoch
                
            # Track loss difference and relative improvement
            prev_val_loss = wandb.run.summary.get("prev_val_loss", val_loss)
            loss_diff = prev_val_loss - val_loss
            rel_improvement = loss_diff / prev_val_loss if prev_val_loss != 0 else 0
            
            wandb.log({
                "val/absolute_improvement": loss_diff,
                "val/relative_improvement": rel_improvement,
                "val/best_loss": wandb.run.summary["best_val_loss"]
            }, step=global_step)
            
            # Update previous loss for next comparison
            wandb.run.summary["prev_val_loss"] = val_loss

    def save_model(self, model, optimizer, epoch, train_loss, val_loss):
        """Save model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save locally
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'model_epoch_{epoch}_valloss_{val_loss:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"\nModel saved to {checkpoint_path}")
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.save(checkpoint_path)

    def load_model(self, model, optimizer, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

    def close(self):
        """Clean up logging."""
        if self.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(level=logging.INFO)
    # Example usage
    logger = Logger(use_wandb=True)
    
    # Log hyperparameters
    config = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 10
    }
    logger.log_hyperparameters(config)
    
    # Log training step
    logger.log_training_step(loss=0.5, epoch=0, batch_idx=1)
    
    # Log epoch metrics
    logger.log_epoch_metrics(train_loss=0.4, val_loss=0.45, epoch=0)
    
    # Close logger
    logger.close()