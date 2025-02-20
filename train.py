import torch
from torch import nn
from torch import optim
import random
import numpy as np
from tqdm import tqdm
import argparse
from dataloader import load_flikr_dataset
from decoder import Decoder

from trainlog import Logger


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_loss(batch, model, device):
    """Compute loss for a single batch."""
    # Move batch to device (if not already on device)
    image_embeddings = batch["image_embedding"].to(device)  # [batch_size, embed_dim]
    input_ids = batch["input_ids"].to(device)  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"].to(device)  # [batch_size, seq_len]
    labels = batch["labels"].to(device)  # [batch_size, seq_len]

    # Get model predictions
    log_probs = model(image_embeddings, input_ids, attention_mask)  # [batch_size, seq_len, vocab_size]
    
    # Reshape for loss computation
    vocab_size = log_probs.size(-1)
    log_probs_flat = log_probs.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    labels_flat = labels.view(-1)  # [batch_size * seq_len]
    
    # Compute loss with stability improvements
    loss = nn.functional.nll_loss(
        log_probs_flat,
        labels_flat,
        ignore_index=-100,
        reduction='mean'
    )
    
    return loss

def train(
    device,
    num_heads=8,
    n_inner=2048,
    clip_embedding_dim=512,
    max_seq_length=77,
    dropout=0.1,
    num_epochs=20,
    lr=1e-3,
    weight_decay=0.01,
    use_wandb=False,
    max_batches=0,
    batch_size=64,
    grad_clip=1.0,
    warmup_steps=1000,
):
    set_seed()
    
    # Initialize logger and model
    logger = Logger(use_wandb=use_wandb)
    logger.log_hyperparameters({
        "num_heads": num_heads,
        "n_inner": n_inner,
        "clip_embedding_dim": clip_embedding_dim,
        "max_seq_length": max_seq_length,
        "dropout": dropout,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "grad_clip": grad_clip,
        "warmup_steps": warmup_steps,
    })

    # Load datasets
    print("\nLoading datasets...")
    train_dataloader = load_flikr_dataset(device, split="train", batch_size=batch_size)
    val_dataloader = load_flikr_dataset(device, split="val", batch_size=batch_size)
    print(f"Dataset loaded. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    # Initialize model
    print("\nInitializing model...")
    decoder = Decoder(
        n_head=num_heads,
        n_inner=n_inner,
        clip_embedding_dim=clip_embedding_dim,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(device)

    # Initialize optimizer with a lower learning rate
    optimizer = optim.AdamW(
        decoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    best_val_loss = float("inf")
    early_stopping_patience = 3
    no_improvement = 0
    
    print("\nStarting training...")
    # Create overall progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        current_epoch = epoch + 1  # Keep track of 1-based epoch number
        
        # Training phase
        decoder.train()
        total_train_loss = 0
        valid_train_batches = 0
        
        # Create progress bar for training batches
        train_iter = tqdm(enumerate(train_dataloader, 1), 
                         total=len(train_dataloader),
                         desc=f"Training Epoch {current_epoch}/{num_epochs}",
                         position=1, 
                         leave=True)
        
        for batch_idx, batch in train_iter:
            try:
                # Compute loss
                loss = compute_loss(batch, decoder, device)
                
                # Skip if loss is NaN
                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss in batch {batch_idx}. Skipping...")
                    continue
                
                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                
                # Check for gradient explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), float('inf'))
                if torch.isnan(grad_norm):
                    print(f"\nWarning: NaN gradient in batch {batch_idx}. Skipping...")
                    continue
                
                optimizer.step()

                # Update metrics
                loss_value = loss.item()
                total_train_loss += loss_value
                valid_train_batches += 1

                # Calculate average loss and learning rate
                avg_loss = total_train_loss / valid_train_batches
                current_lr = optimizer.param_groups[0]["lr"]

                # Update progress bars
                train_iter.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                epoch_pbar.set_postfix({
                    'train_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}'
                })

                # Log step metrics
                if use_wandb:
                    global_step = (current_epoch - 1) * len(train_dataloader) + batch_idx
                    logger.log_training_step(
                        loss_value, 
                        current_epoch, 
                        batch_idx, 
                        phase="train",
                        global_step=global_step
                    )

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue

            if max_batches and batch_idx >= max_batches:
                break

        # Skip epoch if no valid batches
        if valid_train_batches == 0:
            print(f"\nNo valid training batches in epoch {current_epoch}. Skipping...")
            continue

        # Validation phase
        decoder.eval()
        total_val_loss = 0
        valid_val_batches = 0
        
        # Create progress bar for validation batches
        val_iter = tqdm(enumerate(val_dataloader, 1), 
                       total=len(val_dataloader),
                       desc=f"Validation Epoch {current_epoch}/{num_epochs}",
                       position=1, 
                       leave=True)

        with torch.no_grad():
            for batch_idx, batch in val_iter:
                try:
                    loss = compute_loss(batch, decoder, device)
                    if not torch.isnan(loss):
                        loss_value = loss.item()
                        total_val_loss += loss_value
                        valid_val_batches += 1
                        
                        avg_val_loss = total_val_loss / valid_val_batches
                        val_iter.set_postfix({
                            'loss': f'{loss_value:.4f}',
                            'avg_loss': f'{avg_val_loss:.4f}'
                        })
                        
                        if use_wandb:
                            global_step = (current_epoch - 1) * len(val_dataloader) + batch_idx
                            logger.log_training_step(
                                loss_value, 
                                current_epoch, 
                                batch_idx, 
                                phase="val",
                                global_step=global_step
                            )
                            
                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {str(e)}")
                    continue

                if max_batches and batch_idx >= max_batches:
                    break

        # Compute and log metrics
        if valid_val_batches > 0:
            avg_train_loss = total_train_loss / valid_train_batches
            avg_val_loss = total_val_loss / valid_val_batches
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log epoch metrics
            print(f"\n{'='*50}")
            print(f"Epoch {current_epoch}/{num_epochs} Summary:")
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*50}\n")
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if use_wandb:
                logger.log_epoch_metrics(
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    epoch=current_epoch
                )

            # Save best model and check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.save_model(decoder, optimizer, current_epoch, avg_train_loss, avg_val_loss)
                print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= early_stopping_patience:
                print(f"\nNo improvement for {early_stopping_patience} epochs. Early stopping...")
                break
        else:
            print(f"\nNo valid validation batches in epoch {current_epoch}. Skipping metrics...")

    logger.close()
    print("\nTraining completed.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches per epoch (0 for all)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--n-inner", type=int, default=1024, help="Inner dimension size")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    args = parser.parse_args()

    train(
        device=device,
        num_heads=args.num_heads,
        n_inner=args.n_inner,
        dropout=args.dropout,
        use_wandb=args.wandb,
        num_epochs=args.epochs,
        max_batches=args.max_batches,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )