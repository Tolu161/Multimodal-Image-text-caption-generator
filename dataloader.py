










# Import necessary libraries
import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader  # Import Dataset from torch.utils.data
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def collate_fn(batch):
    """Custom collate function to properly stack the batch items."""
    # Handle non-tensor items separately
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    
    # Stack all tensors in the batch
    stacked = {
        "image_embedding": torch.stack([item["image_embedding"] for item in batch]),  # [batch_size, embed_dim]
        "input_ids": torch.stack([item["input_ids"] for item in batch]),  # [batch_size, seq_len]
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),  # [batch_size, seq_len]
        "labels": torch.stack([item["labels"] for item in batch])  # [batch_size, seq_len]
    }
    
    # Add non-tensor items
    stacked["image"] = images
    stacked["caption"] = captions
    
    # Print shapes for debugging
    for key, value in stacked.items():
        if isinstance(value, torch.Tensor):
            print(f"Collated {key} shape: {value.shape}")
    
    return stacked


# Define dataset class
class Flickr30kDataset(Dataset):  # Now inherits from torch.utils.data.Dataset
    def __init__(self, dataset, processor, model, device, split="train"):
        self.dataset = dataset
        self.processor = processor
        self.model = model
        self.device = device
        self.max_length = 77
        self.pad_token_id = processor.tokenizer.pad_token_id
        self._split = split  # Use private attribute with underscore

    @property
    def split(self):
        return self._split

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get a single item
        item = self.dataset[idx]
        image = item["image"]
        captions = item["caption"]

        if self.split == "train":
            # Random caption for training
            caption = captions[torch.randint(0, len(captions), (1,)).item()]
        else:
            # First caption for validation/testing (deterministic)
            caption = captions[0]

        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**image_inputs).squeeze(0)

        # Process text
        text_tokens = self.processor.tokenizer.encode(
            caption,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )

        # Convert to tensors
        input_ids = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.pad_token_id).float()  # [seq_length]

        # Create labels
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        labels[input_ids == self.pad_token_id] = -100

        return {
            "image": image,
            "image_embedding": image_embeddings,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption
        }
        

# Load OpenAI API key from environment variable (secure)
#openai.api_key = os.getenv("OPENAI_API_KEY")

def load_flikr_dataset(device, split="train", batch_size=32, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Load and prepare the Flickr30k dataset.
    Args:
        device: The device to load the data on
        split: One of 'train', 'validation', or 'test'
        batch_size: Batch size for the dataloader
        train_ratio: Ratio of data to use for training (default 0.8)
        val_ratio: Ratio of data to use for validation (default 0.1)
        seed: Random seed for reproducibility
    """
    # Load the full dataset
    full_dataset = load_dataset("nlphuji/flickr30k", split="test")
    print(f"\nLoaded full dataset with {len(full_dataset)} samples")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Shuffle and split the dataset
    shuffled_indices = torch.randperm(total_size)
    
    if split == "train":
        indices = shuffled_indices[:train_size]
        print(f"Using {len(indices)} samples for training")
        dataset = full_dataset.select(indices.tolist())
    elif split == "validation":
        indices = shuffled_indices[train_size:train_size + val_size]
        print(f"Using {len(indices)} samples for validation")
        dataset = full_dataset.select(indices.tolist())
    else:  # test
        indices = shuffled_indices[train_size + val_size:]
        print(f"Using {len(indices)} samples for testing")
        dataset = full_dataset.select(indices.tolist())
    
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create dataset instance
    flickr_dataset = Flickr30kDataset(
        dataset=dataset,
        processor=processor,
        model=model,
        device=device,
        split=split
    )
    
    return DataLoader(
        flickr_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )

        
if __name__ == "__main__":
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"\nUsing device: {device}")

    """
    # Original test code (commented out)
    # Test single item first
    dataloader = load_flikr_dataset(device, batch_size=1)
    single_batch = next(iter(dataloader))
    print("\nSingle item shapes:")
    print("Image Embeddings:", single_batch["image_embedding"].shape)  # [1, 512]
    print("Input IDs:", single_batch["input_ids"].shape)  # [1, 77]
    print("Attention Mask:", single_batch["attention_mask"].shape)  # [1, 77]
    print("Labels:", single_batch["labels"].shape)  # [1, 77]

    # Test with actual batch size
    dataloader = load_flikr_dataset(device, batch_size=32)
    batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print("Image Embeddings:", batch["image_embedding"].shape)  # [32, 512]
    print("Input IDs:", batch["input_ids"].shape)  # [32, 77]
    print("Attention Mask:", batch["attention_mask"].shape)  # [32, 77]
    print("Labels:", batch["labels"].shape)  # [32, 77]
    """

    # New detailed verification code
    print("\nRunning detailed dataset verification...")
    
    # First, verify the raw dataset
    print("\nVerifying raw dataset...")
    raw_dataset = load_dataset("nlphuji/flickr30k", split="test")
    print(f"Total dataset size: {len(raw_dataset)}")
    
    # Show example from raw dataset
    print("\nExample from raw dataset:")
    example = raw_dataset[0]
    print(f"Number of captions: {len(example['caption'])}")
    print("Captions:")
    for i, cap in enumerate(example['caption'], 1):
        print(f"{i}. {cap}")
    
    # Now test our split functionality
    print("\nTesting dataset splits...")
    for split in ["train", "validation", "test"]:
        print(f"\n{'='*50}")
        print(f"Testing {split} split:")
        print('='*50)
        
        # Test dataloader
        print(f"\nTesting dataloader for {split} split...")
        dataloader = load_flikr_dataset(device, split=split, batch_size=2)
        batch = next(iter(dataloader))
        
        print("\nDataloader batch contents:")
        print(f"Batch size: {len(batch['caption'])}")
        print("\nShapes:")
        print(f"Image Embeddings: {batch['image_embedding'].shape}")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Attention Mask: {batch['attention_mask'].shape}")
        print(f"Labels: {batch['labels'].shape}")
        
        print("\nCaptions from batch:")
        for i, cap in enumerate(batch['caption'], 1):
            print(f"{i}. {cap}")
            
        # Verify tokenization
        print("\nVerifying tokenization:")
        processor = dataloader.dataset.processor
        for i, cap in enumerate(batch['caption'][:2]):  # Check first two captions
            print(f"\nCaption {i+1}:")
            print("Original:", cap)
            # Decode the input_ids to verify tokenization is correct
            decoded = processor.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
            print("Decoded from tokens:", decoded)
            
        print("\nVerification complete for", split)





'''
def __getitem__(self, idx):
        # Get a single item
        item = self.dataset[idx]
        image = item["image"]
        
        # Get captions - Flickr30k provides 5 captions per image
        captions = item["caption"]
        
        if self.split == "train":
            # Random caption for training
            caption = captions[torch.randint(0, len(captions), (1,)).item()]
        else:
            # First caption for validation/testing (deterministic)
            caption = captions[0]
            
        # print(f"Selected caption: {caption}")  # Debug print commented out
        
        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**image_inputs).squeeze(0)
        
        # Process text
        text_tokens = self.processor.tokenizer.encode(
            caption,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Convert to tensors
        input_ids = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.pad_token_id).float()
        
        # Create labels
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        labels[input_ids == self.pad_token_id] = -100
        
        return {
            "image": image,
            "image_embedding": image_embeddings,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "caption": caption
        }


'''





























































































