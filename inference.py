import os
import torch
from decoder import Decoder
from dataloader import load_flikr_dataset
import argparse
import warnings
import matplotlib.pyplot as plt
import textwrap
from transformers import CLIPProcessor
from tqdm import tqdm

warnings.simplefilter("ignore", category=FutureWarning)

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"\nLoading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with saved configuration
    config = checkpoint.get('config', {
        'n_head': 4,
        'n_inner': 1024,
        'clip_embedding_dim': 512,
        'max_seq_length': 77,
        'dropout': 0.1
    })
    
    model = Decoder(**config).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # Assume it's just the state dict
        
    print("Model loaded successfully")
    return model

def generate_caption(model, image_embedding, processor, max_length=77, min_length=5, temperature=1.0):
    """Generate a caption for an image."""
    model.eval()
    
    with torch.no_grad():
        print(f"\nInitial image_embedding shape: {image_embedding.shape}")
        
        # Version 1: Simple shape handling (currently using this)
        # Ensure image_embedding has correct shape [batch_size, hidden_size]
        if len(image_embedding.shape) == 3:  # [1, 1, hidden_size]
            print("Case 1: Squeezing dimension 1")
            image_embedding = image_embedding.squeeze(1)  # Remove middle dimension
        elif len(image_embedding.shape) == 1:  # [hidden_size]
            print("Case 2: Adding batch dimension")
            image_embedding = image_embedding.unsqueeze(0)  # Add batch dimension
            
        print(f"Final image_embedding shape: {image_embedding.shape}")
            
        # Start with empty token sequence
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=image_embedding.device)
        
        # Version 1: Simple attention mask (currently using this)
        attention_mask = torch.ones_like(input_ids)  # [1, 1]
        
        # Version 2: Complex attention mask (commented out for now)
        # attention_mask = torch.ones((1, 1, 1), dtype=torch.float, device=image_embedding.device)
        # attention_mask = attention_mask.expand(-1, -1, 2)  # Expand for image token
        
        for i in range(max_length - 1):
            # Version 1: Simple forward pass (currently using this)
            log_probs = model(image_embedding, input_ids, attention_mask)
            
            # Version 2: Complex forward pass (commented out for now)
            # log_probs = model(image_embedding.unsqueeze(0), input_ids, attention_mask)
            
            next_token_logits = log_probs[:, -1, :] / temperature
            
            # Prevent EOS before min_length
            if i < min_length:
                next_token_logits[0, processor.tokenizer.eos_token_id] = float('-inf')
            
            # Sample from the distribution for more diverse captions
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)[0]
            
            # Stop if EOS token (after min_length)
            if next_token.item() == processor.tokenizer.eos_token_id:
                break
                
            # Add token to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Version 1: Simple mask update (currently using this)
            attention_mask = torch.ones_like(input_ids)
            
            # Version 2: Complex mask update (commented out for now)
            # seq_length = input_ids.size(1) + 1  # +1 for image token
            # attention_mask = torch.ones((1, seq_length, seq_length), dtype=torch.float, device=image_embedding.device)
        
        # Decode caption
        caption = processor.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return caption

def display_results(image, generated_caption, original_caption, save_path=None):
    """Display image with generated and original captions."""
    # Wrap captions for display
    wrapped_gen = textwrap.fill(f"Generated: {generated_caption}", width=60)
    wrapped_orig = textwrap.fill(f"Original: {original_caption}", width=60)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Add title with captions
    plt.title(
        f"{wrapped_gen}\n\n{wrapped_orig}",
        pad=20,
        fontsize=12,
        wrap=True,
        y=1.05
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    plt.close()

def save_weights(model, save_dir="saved_weights", filename="model_weights.pt"):
    """Save model weights to specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_head': model.layers[0].n_head,
            'n_inner': model.layers[0].d_model * 4,
            'clip_embedding_dim': model.hidden_size,
            'max_seq_length': model.token_embedding.num_embeddings,
            'dropout': model.embed_dropout.p
        }
    }, save_path)
    
    print(f"\nModel weights saved to: {save_path}")
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using trained model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate captions for")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more diverse)")
    parser.add_argument("--save-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--save-weights", action="store_true", help="Save current model weights")
    parser.add_argument("--weights-dir", type=str, default="saved_weights", help="Directory to save model weights")
    args = parser.parse_args()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    model = Decoder().to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        model = load_model(args.checkpoint, device)
    
    # Save weights if requested
    if args.save_weights:
        weights_path = save_weights(model, args.weights_dir)
        if not args.checkpoint:  # If no checkpoint was loaded, use the saved weights
            args.checkpoint = weights_path
    
    # Ensure we have weights to use
    if not args.checkpoint and not args.save_weights:
        raise ValueError("Either --checkpoint or --save-weights must be provided")
    
    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataloader = load_flikr_dataset(device, split="val", batch_size=args.batch_size)
    processor = val_dataloader.dataset.processor
    
    # Generate captions
    print("\nGenerating captions...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Processing images")):
            if i >= args.num_samples:
                break
            
            # Get image and embeddings
            image = batch["image"][0]  # Get first image from batch
            print(f"\nBatch image_embedding shape: {batch['image_embedding'].shape}")
            image_embeddings = batch["image_embedding"][0]  # Remove batch dimension
            print(f"Selected image_embedding shape: {image_embeddings.shape}")
            true_caption = batch["caption"][0]
            
            # Generate caption
            generated_caption = generate_caption(
                model, 
                image_embeddings,  # Pass without extra unsqueeze
                processor,
                temperature=args.temperature
            )
            
            # Display results
            save_path = os.path.join(args.save_dir, f"sample_{i+1}.png") if args.save_dir else None
            display_results(image, generated_caption, true_caption, save_path)
            
            # Print captions
            print(f"\nImage {i+1}:")
            print(f"True caption: {true_caption}")
            print(f"Generated caption: {generated_caption}")
            print("-" * 80)

if __name__ == "__main__":
    main()