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
    # Handle missing config
    default_config = {
        'n_head': 1,
        'n_inner': 1024,
        'clip_embedding_dim': 512,
        'max_seq_length': 77,
        'dropout': 0.1
    }
    config = checkpoint.get('config', default_config)  # Use saved config if available

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
        try:
            # Ensure image_embedding has shape [batch_size, hidden_size]
            if len(image_embedding.shape) == 3:  # [1, 1, hidden_size]
                image_embedding = image_embedding.squeeze()
            if len(image_embedding.shape) == 1:  # [hidden_size]
                image_embedding = image_embedding.unsqueeze(0)
            
            # Start with BOS token
            input_ids = torch.tensor([[processor.tokenizer.bos_token_id]], dtype=torch.long, device=image_embedding.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.float)
            
            generated_tokens = []
            eos_token_id = processor.tokenizer.eos_token_id
            
            for i in range(max_length - 1):
                # Create causal attention mask
                seq_length = input_ids.size(1)
                causal_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=input_ids.device))
                
                # Forward pass
                log_probs = model(image_embedding, input_ids, attention_mask)
                next_token_logits = log_probs[:, -1, :] / temperature
                
                # Prevent EOS before min_length
                if i < min_length:
                    next_token_logits[0, eos_token_id] = float('-inf')
                
                # Sample from the filtered distribution
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)[0]
                
                # If we get EOS token and we're past min_length, stop
                if next_token.item() == eos_token_id and i >= min_length:
                    generated_tokens.append(next_token.item())
                    break
                
                # Add the token to our sequence
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.float)
                
                # Print progress for debugging
                if i % 10 == 0:
                    print(f"Generated {i} tokens: {processor.tokenizer.decode(generated_tokens)}")
            
            # Decode the sequence
            if not generated_tokens:
                return "Failed to generate caption"
            
            # Include BOS token in decoding
            all_tokens = [processor.tokenizer.bos_token_id] + generated_tokens
            caption = processor.tokenizer.decode(all_tokens, skip_special_tokens=True)
            
            # Print final caption for debugging
            print(f"Final generated caption: {caption}")
            
            return caption.strip() if caption.strip() else "Failed to generate meaningful caption"
            
        except Exception as e:
            print(f"Error in caption generation: {str(e)}")
            return "Error generating caption"

def display_results(image, generated_caption, original_caption, save_path=None):
    """Display image with generated and original captions."""
    plt.figure(figsize=(12, 8))
    
    # Create subplot for the image
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.axis('off')
    
    # Format captions with textwrap
    wrapped_gen = textwrap.fill(f"Generated: {generated_caption}", width=60)
    wrapped_orig = textwrap.fill(f"Ground Truth: {original_caption}", width=60)
    
    # Add title with both captions
    plt.title(
        f"{wrapped_orig}\n\n{wrapped_gen}",
        pad=20,
        fontsize=12,
        wrap=True,
        y=1.05,
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='gray',
            boxstyle='round,pad=1'
        )
    )
    
    # Adjust layout to prevent caption cutoff
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    plt.close()

def generate_and_display_batch(model, dataloader, processor, num_images=5, save_dir=None, temperature=1.0):
    """Generate and display captions for a batch of images."""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_images:
                break
                
            try:
                # Get image and embeddings
                image = batch["image"][0]  # Get first image from batch
                image_embeddings = batch["image_embedding"][0]  # Remove batch dimension
                original_caption = batch["caption"][0]
                
                # Generate caption
                generated_caption = generate_caption(
                    model,
                    image_embeddings,
                    processor,
                    temperature=temperature
                )
                
                # Create save path if directory provided
                save_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    #save_path = os.path.join(save_dir, f"sample_{i+1}.png")
                    # Define save paths for the image and visualization
                    image_save_path = os.path.join(save_dir, f"image_{i+1}.png")
                    visualization_save_path = os.path.join(save_dir, f"sample_{i+1}.png")


                 # Save the original image
                    plt.imsave(image_save_path, image.cpu().numpy())  # Make sure to convert tensor to numpy
                    print(f"Saved original image to {image_save_path}") 

                
                # Display results
                display_results(
                    image,
                    generated_caption,
                    original_caption,
                    save_path=visualization_save_path
                )
                
                # Print comparison
                print(f"\nImage {i+1}:")
                print(f"Ground Truth: {original_caption}")
                print(f"Generated  : {generated_caption}")
                print("-" * 80)
                
            except Exception as e:
                print(f"Error processing image {i+1}: {str(e)}")
                continue

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
    
    try:
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model and checkpoint
        if not args.checkpoint:
            raise ValueError("Please provide a checkpoint path using --checkpoint")
            
        model = load_model(args.checkpoint, device)
        
        # Save weights if requested
        if args.save_weights:
            weights_path = save_weights(model, args.weights_dir)
            if not args.checkpoint:  # If no checkpoint was loaded, use the saved weights
                args.checkpoint = weights_path
        
        # Ensure we have weights to use
        if not args.checkpoint and not args.save_weights:
            raise ValueError("Either --checkpoint or --save-weights must be provided")
        
        # Load validation dataset
        print("\nLoading validation dataset...")
        val_dataloader = load_flikr_dataset(device, split="validation", batch_size=args.batch_size)
        processor = val_dataloader.dataset.processor
        
        # Generate and display captions
        print("\nGenerating captions...")
        generate_and_display_batch(
            model,
            val_dataloader,
            processor,
            num_images=args.num_samples,
            save_dir=args.save_dir,
            temperature=args.temperature
        )
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()