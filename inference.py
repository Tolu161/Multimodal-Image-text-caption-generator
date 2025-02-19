import os
import torch
from decoder import Decoder
from dataloader import load_flikr_dataset
import argparse
import warnings
import matplotlib.pyplot as plt
import textwrap
from transformers import CLIPProcessor

warnings.simplefilter("ignore", category=FutureWarning)

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
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
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def generate_caption(model, image_embedding, processor, max_length=77, min_length=5):
    """Generate a caption for an image."""
    model.eval()
    
    with torch.no_grad():
        # Start with empty token sequence
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=image_embedding.device)
        attention_mask = torch.ones_like(input_ids)
        
        for i in range(max_length - 1):
            # Get model predictions
            log_probs = model(image_embedding, input_ids, attention_mask)
            next_token_logits = log_probs[:, -1, :]
            
            # Prevent EOS before min_length
            if i < min_length:
                next_token_logits[0, processor.tokenizer.eos_token_id] = float('-inf')
            
            # Get next token
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Stop if EOS token (after min_length)
            if next_token.item() == processor.tokenizer.eos_token_id:
                break
                
            # Add token to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            attention_mask = torch.ones_like(input_ids)
        
        # Decode caption
        caption = processor.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return caption

def display_results(image, generated_caption, original_caption):
    """Display image with generated and original captions."""
    # Wrap captions for display
    wrapped_gen = textwrap.fill(f"Generated: {generated_caption}", width=60)
    wrapped_orig = textwrap.fill(f"Original: {original_caption}", width=60)
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # Add title with captions
    plt.title(
        f"{wrapped_gen}\n\n{wrapped_orig}",
        pad=20,
        fontsize=10,
        wrap=True,
        y=1.05
    )
    
    # Adjust layout
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image-idx",
        type=int,
        default=0,
        help="Index of validation image to caption"
    )
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load CLIP processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Get validation dataloader
    val_dataloader = load_flikr_dataset(device, split="val", batch_size=1)
    
    # Get specified image
    for i, batch in enumerate(val_dataloader):
        if i == args.image_idx:
            break
    else:
        raise ValueError(f"Image index {args.image_idx} out of range")
    
    # Generate caption
    image = batch["image"]
    original_caption = batch["caption"][0]
    image_embedding = batch["image_embedding"].to(device)
    
    generated_caption = generate_caption(
        model,
        image_embedding,
        processor,
        max_length=77,
        min_length=5
    )
    
    # Print results
    print("\nGenerated caption:", generated_caption)
    print("Original caption:", original_caption)
    
    # Display results
    display_results(image, generated_caption, original_caption)