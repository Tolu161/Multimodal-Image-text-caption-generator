# import necessary libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import load_flikr_dataset 
from transformers import CLIPProcessor, CLIPModel

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        # Shape checks
        batch_size, seq_length, hidden_size = x.size()
        assert hidden_size == self.d_model, \
            f"Input hidden size {hidden_size} doesn't match layer's d_model {self.d_model}"

        # Attention with residual
        residual = x
        x = self.attn_norm(x)

        # Handle attention mask for PyTorch's MultiheadAttention
        if attention_mask is not None:
            # Convert to key_padding_mask format (batch_size, seq_length)
            # True means position is masked, False means position is attended to
            key_padding_mask = (attention_mask == 0) # [batch_size, seq_length_with_image]
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = residual + attn_out

        # Feed-forward with residual
        residual = x
        x = self.ff_norm(x)
        x = residual + self.ff(x)

        return x


class Decoder(nn.Module):
    def __init__(self, n_head=8, n_inner=2048, clip_embedding_dim=512, max_seq_length=77, dropout=0.1, vocab_size=49408, n_layers=4):
        super().__init__()
        self.hidden_size = clip_embedding_dim
        
        # Embeddings with dropout
        self.token_embedding = nn.Embedding(vocab_size, clip_embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, clip_embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Image projection with residual connection
        self.image_norm = nn.LayerNorm(clip_embedding_dim)
        self.image_projection = nn.Sequential(
            nn.Linear(clip_embedding_dim, clip_embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clip_embedding_dim * 4, clip_embedding_dim),
            nn.Dropout(dropout),
        )
        
        # Create causal attention mask once during initialization
        mask = torch.tril(torch.ones(max_seq_length + 1, max_seq_length + 1))
        self.register_buffer("causal_mask", mask)
        
        # Multiple transformer layers with residual connections
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(clip_embedding_dim, n_head, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.final_norm = nn.LayerNorm(clip_embedding_dim)
        
        # Output head with residual connection
        self.lm_head = nn.Sequential(
            nn.Linear(clip_embedding_dim, clip_embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clip_embedding_dim * 4, clip_embedding_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(clip_embedding_dim),
            nn.Linear(clip_embedding_dim, vocab_size),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, image_embeddings, input_ids, attention_mask=None):
        # Input shape checks
        batch_size, seq_length = input_ids.size()
        assert image_embeddings.size() == (batch_size, self.hidden_size), \
            f"Expected image_embeddings shape ({batch_size}, {self.hidden_size}), got {image_embeddings.size()}"
        if attention_mask is not None:
            assert attention_mask.size() == (batch_size, seq_length), \
                f"Expected attention_mask shape ({batch_size}, {seq_length}), got {attention_mask.size()}"

        # Project image embeddings with residual connection
        residual = image_embeddings  # [batch_size, hidden_size]
        image_embeddings = self.image_norm(image_embeddings)
        image_embeddings = residual + self.image_projection(image_embeddings)

        # Get token embeddings and add positional embeddings with dropout
        token_embeddings = self.token_embedding(input_ids)  # [batch_size, seq_length, hidden_size]
        positions = torch.arange(seq_length, device=input_ids.device)
        pos_embeddings = self.pos_embedding(positions)  # [seq_length, hidden_size]
        token_embeddings = self.embed_dropout(token_embeddings + pos_embeddings.unsqueeze(0))

        # Concatenate image and text embeddings
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        sequence = torch.cat([image_embeddings, token_embeddings], dim=1)  # [batch_size, seq_length + 1, hidden_size]
        seq_length_with_image = seq_length + 1

        # Apply causal masking to attention mask
        causal = self.causal_mask[:seq_length_with_image, :seq_length_with_image]  # [seq_length_with_image, seq_length_with_image]

        # Ensure attention_mask is of shape [batch_size, seq_length_with_image]
        if attention_mask is not None:
            # Add attention for image token (1 = not masked, 0 = masked)
            image_attention = torch.ones((batch_size, 1), device=attention_mask.device)
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)  # [batch_size, seq_length + 1]
        else:
            attention_mask = torch.ones((batch_size, seq_length_with_image), device=input_ids.device)

        
                # Transpose attention_mask to [seq_length_with_image, batch_size] and perform matmul
        attention_mask = attention_mask.transpose(0, 1)  # [seq_length_with_image, batch_size]
        attention_mask = torch.matmul(causal, attention_mask)  # [seq_length_with_image, batch_size]

        # Transpose back to [batch_size, seq_length_with_image]
        attention_mask = attention_mask.transpose(0, 1)  # [batch_size, seq_length_with_image]

        # Process through transformer layers
        for layer in self.layers:
            sequence = layer(sequence, attention_mask=attention_mask)

        # Final normalization
        sequence = self.final_norm(sequence)

        # Get text sequence and compute logits
        text_sequence = sequence[:, 1:, :]  # Remove image token
        logits = self.lm_head(text_sequence)

        # Return log probabilities
        return torch.log_softmax(logits.float(), dim=-1)

        







    '''
    def forward(self, image_embeddings, input_ids, attention_mask=None):
        # Input shape checks
        batch_size, seq_length = input_ids.size()
        assert image_embeddings.size() == (batch_size, self.hidden_size), \
            f"Expected image_embeddings shape ({batch_size}, {self.hidden_size}), got {image_embeddings.size()}"
        if attention_mask is not None:
            assert attention_mask.size() == (batch_size, seq_length), \
                f"Expected attention_mask shape ({batch_size}, {seq_length}), got {attention_mask.size()}"

        # Project image embeddings with residual connection
        residual = image_embeddings  # [batch_size, hidden_size]
        image_embeddings = self.image_norm(image_embeddings)
        image_embeddings = residual + self.image_projection(image_embeddings)

        # Get token embeddings and add positional embeddings with dropout
        token_embeddings = self.token_embedding(input_ids)  # [batch_size, seq_length, hidden_size]
        positions = torch.arange(seq_length, device=input_ids.device)
        pos_embeddings = self.pos_embedding(positions)  # [seq_length, hidden_size]
        token_embeddings = self.embed_dropout(token_embeddings + pos_embeddings.unsqueeze(0))

        # Concatenate image and text embeddings
        image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        sequence = torch.cat([image_embeddings, token_embeddings], dim=1)  # [batch_size, seq_length + 1, hidden_size]
        seq_length_with_image = seq_length + 1

        # Create attention mask including image token
        if attention_mask is not None:
            # Add attention for image token (1 = not masked, 0 = masked)
            image_attention = torch.ones((batch_size, 1), device=attention_mask.device)
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)  # [batch_size, seq_length + 1]
        else:
            attention_mask = torch.ones((batch_size, seq_length_with_image), device=input_ids.device)

        # Apply causal masking to attention mask
        causal = self.causal_mask[:seq_length_with_image, :seq_length_with_image]  # [seq_length_with_image, seq_length_with_image]
        attention_mask = attention_mask.unsqueeze(1) * causal.unsqueeze(0)  # [batch_size, 1, seq_length_with_image]
        attention_mask = attention_mask.squeeze(1)  # [batch_size, seq_length_with_image]

        # Process through transformer layers
        for layer in self.layers:
            sequence = layer(sequence, attention_mask=attention_mask)

        # Final normalization
        sequence = self.final_norm(sequence)

        # Get text sequence and compute logits
        text_sequence = sequence[:, 1:, :]  # Remove image token
        logits = self.lm_head(text_sequence)

        # Return log probabilities
        return torch.log_softmax(logits.float(), dim=-1)
''' 



'''
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, attention_mask=None):
        # Shape checks
        batch_size, seq_length, hidden_size = x.size()
        assert hidden_size == self.d_model, \
            f"Input hidden size {hidden_size} doesn't match layer's d_model {self.d_model}"
        
        # Attention with residual
        residual = x
        x = self.attn_norm(x)
        
        # Handle attention mask for PyTorch's MultiheadAttention
        if attention_mask is not None:
            # Convert to key_padding_mask format (batch_size, seq_length)
            # True means position is masked, False means position is attended to
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = residual + attn_out
        
        # FF with residual
        residual = x
        x = self.ff_norm(x)
        x = residual + self.ff(x)
        
        return x

'''











''' 

    def forward(self, image_embeddings, input_ids, attention_mask=None):
        # Input shape checks
        batch_size, seq_length = input_ids.size()
        assert image_embeddings.size() == (batch_size, self.hidden_size), \
            f"Expected image_embeddings shape ({batch_size}, {self.hidden_size}), got {image_embeddings.size()}"
        if attention_mask is not None:
            assert attention_mask.size() == (batch_size, seq_length), \
                f"Expected attention_mask shape ({batch_size}, {seq_length}), got {attention_mask.size()}"
        
        # Project image embeddings with residual connection
        residual = image_embeddings  # [batch_size, hidden_size]
        image_embeddings = self.image_norm(image_embeddings)
        image_embeddings = residual + self.image_projection(image_embeddings)
        
        # Get token embeddings and add positional embeddings with dropout
        token_embeddings = self.token_embedding(input_ids)  # [batch_size, seq_length, hidden_size]
        positions = torch.arange(seq_length, device=input_ids.device)
        pos_embeddings = self.pos_embedding(positions)  # [seq_length, hidden_size]
        token_embeddings = self.embed_dropout(token_embeddings + pos_embeddings.unsqueeze(0))
        
        # Concatenate image and text embeddings
        sequence = torch.cat([image_embeddings.unsqueeze(1), token_embeddings], dim=1)
        seq_length_with_image = seq_length + 1
        
        # Create attention mask including image token
        if attention_mask is not None:
            # Add attention for image token
            image_attention = torch.ones((batch_size, 1), device=attention_mask.device)
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            attention_mask = torch.ones((batch_size, seq_length_with_image), device=input_ids.device)
        
        # Apply causal masking to attention mask - this is not needed as we are using key_padding_mask
        #causal = self.causal_mask[:seq_length_with_image, :seq_length_with_image]
        #attention_mask = attention_mask.unsqueeze(1) * causal.unsqueeze(0)
        #attention_mask = attention_mask.squeeze(1)  # [batch_size, seq_length_with_image]

        # Apply causal masking to attention mask
        causal = self.causal_mask[:seq_length_with_image, :seq_length_with_image]  # [seq_length_with_image, seq_length_with_image]
        attention_mask = attention_mask * causal.unsqueeze(0)  # [batch_size, seq_length_with_image]
        
        # Process through transformer layers
        for layer in self.layers:
            sequence = layer(sequence, attention_mask=attention_mask)
        
        # Final normalization
        sequence = self.final_norm(sequence)
        
        # Get text sequence and compute logits
        text_sequence = sequence[:, 1:, :]  # Remove image token
        logits = self.lm_head(text_sequence)
        
        # Return log probabilities
        return torch.log_softmax(logits.float(), dim=-1)


'''














