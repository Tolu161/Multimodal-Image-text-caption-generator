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

def save_current_weights(model, save_dir="saved_weights"):
    """Save the current model weights."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_weights.pt")
    
    # Save only the model state dict
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")
    return save_path

def load_weights(model, weights_path):
    """Load weights into the model."""
    if os.path.exists(weights_path):
        print(f"\nLoading weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path))
        print("Weights loaded successfully")
    else:
        raise FileNotFoundError(f"No weights file found at {weights_path}")

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

def main(weights_path=None, save_weights=True):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load validation dataset (for testing)
    print("\nLoading validation dataset...")
    val_dataloader = load_flikr_dataset(device, split="val", batch_size=1)
    
    # Initialize model
    print("\nInitializing model...")
    model = Decoder().to(device)
    
    # Save current weights if requested
    if save_weights and weights_path is None:
        weights_path = save_current_weights(model)
    
    # Load weights if path is provided
    if weights_path:
        load_weights(model, weights_path)
    
    # Get processor from dataloader for tokenization
    processor = val_dataloader.dataset.processor
    
    # Generate captions for a few validation images
    print("\nGenerating captions...")
    num_samples = 5
    for i, batch in enumerate(val_dataloader):
        if i >= num_samples:
            break
            
        # Get image and embeddings
        image = batch["image"]
        image_embeddings = batch["image_embedding"][0]  # [512]
        image_embeddings = image_embeddings.unsqueeze(0)  # [1, 512]
        
        # Get true caption
        true_caption = processor.tokenizer.decode(
            batch["input_ids"][0],
            skip_special_tokens=True
        )
        
        # Generate caption
        generated_caption = generate_caption(model, image_embeddings, processor)
        
        print(f"\nImage {i+1}:")
        print(f"True caption: {true_caption}")
        print(f"Generated caption: {generated_caption}")
        
        # Display results
        display_results(image[0], generated_caption, true_caption)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--save-weights", action="store_true", help="Save current weights before inference")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate captions for")
    args = parser.parse_args()
    
    main(weights_path=args.weights, save_weights=args.save_weights)