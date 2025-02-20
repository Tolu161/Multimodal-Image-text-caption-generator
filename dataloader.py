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
        
        # Select caption based on split
        captions = item["caption"][0]
        if self.split == "train":
            # Random caption for training
            caption = captions[torch.randint(0, len(captions), (1,)).item()]
        else:
            # First caption for validation/testing (deterministic)
            caption = captions[0]
        
        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Get image embeddings [1, embed_dim] -> [embed_dim]
            image_embeddings = self.model.get_image_features(**image_inputs).squeeze(0)  # Remove batch dimension
        
        # Process text - manually handle padding
        text_tokens = self.processor.tokenizer.encode(
            caption,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Ensure fixed length
            return_tensors=None  # Return list
        )
        
        # Convert to tensors and move to device - single item, no batch dimension
        input_ids = torch.tensor(text_tokens, dtype=torch.long, device=self.device)  # [seq_len]
        attention_mask = (input_ids != self.pad_token_id).float()  # [seq_len]
        
        # Create labels (shifted input_ids)
        labels = input_ids.clone()  # [seq_len]
        labels[:-1] = input_ids[1:]  # shift left by 1
        labels[-1] = -100  # Ignore last prediction
        labels[input_ids == self.pad_token_id] = -100  # Ignore padding tokens
        
        # Verify shapes
        assert input_ids.shape[0] == self.max_length, f"input_ids shape {input_ids.shape} != {self.max_length}"
        assert attention_mask.shape[0] == self.max_length, f"attention_mask shape {attention_mask.shape} != {self.max_length}"
        assert labels.shape[0] == self.max_length, f"labels shape {labels.shape} != {self.max_length}"
        
        return {
            "image": image,  # Original PIL image
            "image_embedding": image_embeddings,  # [embed_dim]
            "input_ids": input_ids,  # [max_length]
            "attention_mask": attention_mask,  # [max_length]
            "labels": labels,  # [max_length]
            "caption": caption  # Original caption string
        }




# Load OpenAI API key from environment variable (secure)
#openai.api_key = os.getenv("OPENAI_API_KEY")


def load_flikr_dataset(device, split="train", batch_size=32, train_ratio=0.8, seed=42):
    # Load dataset
    f_dataset = load_dataset("nlphuji/flickr30k", split='test')
    #dataset = dataset.select(range(500))  # Load only 500 samples

    # Calculate split sizes
    total_size = len(f_dataset)
    train_size = int(total_size * train_ratio)

    # Split dataset
    f_dataset = f_dataset.shuffle(seed=seed)
    train_dataset = f_dataset.select(range(train_size))
    val_dataset = f_dataset.select(range(train_size, total_size))

    # Select appropriate split
    dataset = train_dataset if split == "train" else val_dataset
    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokeniser = processor.tokenizer
    # Device setup
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    # Create dataset instance with split parameter
    flickr_dataset = Flickr30kDataset(
        dataset=dataset,
        processor=processor,
        model=model,
        device=device,
        split=split  # Pass the split parameter
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



































































































