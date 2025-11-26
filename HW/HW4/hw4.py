"""
COMPLETE PyTorch LSTM Text Generation Implementation
Based on MachineLearningMastery approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

class TextDataset(Dataset):
    """Convert text into character sequences for LSTM training"""
    
    def __init__(self, text, seq_length=50):
        """
        Args:
            text: Raw text string
            seq_length: Length of input sequences
        """
        self.seq_length = seq_length
        
        # Build character vocabulary
        self.chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        
        # Encode entire text to indices
        self.text_encoded = [self.char_to_idx[c] for c in text]
    
    def __len__(self):
        return len(self.text_encoded) - self.seq_length
    
    def __getitem__(self, idx):
        """Return (input_sequence, target_sequence) pair"""
        x = torch.tensor(
            self.text_encoded[idx:idx + self.seq_length],
            dtype=torch.long
        )
        y = torch.tensor(
            self.text_encoded[idx + 1:idx + self.seq_length + 1],
            dtype=torch.long
        )
        return x, y
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[i] for i in indices])


# ============================================================================
# STEP 2: MODEL ARCHITECTURE
# ============================================================================

class LSTMTextGenerator(nn.Module):
    """LSTM-based character-level text generation model"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.5):
        """
        Args:
            vocab_size: Number of unique characters
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between LSTM layers
        """
        super(LSTMTextGenerator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Tuple of (hidden_state, cell_state) or None
        
        Returns:
            logits: Output logits of shape (batch_size, seq_length, vocab_size)
            hidden: Updated (hidden_state, cell_state)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Linear: (batch, seq_len, hidden_dim) -> (batch, seq_len, vocab_size)
        logits = self.fc(lstm_out)
        
        return logits, hidden


# ============================================================================
# STEP 3: TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits, _ = model(x)  # (batch, seq_len, vocab_size)
        
        # Compute loss: reshape for CrossEntropyLoss
        # CrossEntropyLoss expects (N, C) where N = batch*seq_len, C = vocab_size
        loss = criterion(
            logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
            y.view(-1)                          # (batch*seq_len,)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train(model, train_loader, num_epochs, learning_rate, device='cpu'):
    """Train the LSTM model"""
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")


# ============================================================================
# STEP 4: TEXT GENERATION
# ============================================================================

def generate_text(model, dataset, seed_text, length, temperature=1.0,
                  device='cpu'):
    """
    Generate text starting from seed_text
    
    Args:
        model: Trained LSTM model
        dataset: TextDataset instance (for encoding/decoding)
        seed_text: Starting text
        length: Number of characters to generate
        temperature: Controls randomness
                    - Low (0.5): More deterministic
                    - High (1.5): More random
        device: 'cpu' or 'cuda'
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Convert seed to indices
    indices = [dataset.char_to_idx[c] for c in seed_text]
    
    with torch.no_grad():
        for _ in range(length):
            # Use last seq_length characters as context
            if len(indices) >= dataset.seq_length:
                x = torch.tensor(
                    indices[-dataset.seq_length:],
                    dtype=torch.long
                ).unsqueeze(0).to(device)
            else:
                x = torch.tensor(
                    indices,
                    dtype=torch.long
                ).unsqueeze(0).to(device)
            
            # Get model prediction
            logits, _ = model(x)
            
            # Get logits for next character (last position in sequence)
            next_logits = logits[0, -1, :] / temperature
            
            # Apply softmax and sample
            probs = torch.softmax(next_logits, dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            
            indices.append(next_idx)
    
    return dataset.decode(indices)


# ============================================================================
# STEP 5: COMPLETE USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    TEXT_FILE = "theprince.txt"  # Your text file
    SEQ_LENGTH = 50
    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # ===== Load and Prepare Data =====
    # Load your text file
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        text = f.read().lower()  # Lowercase for consistency
    
    print(f"Loaded {len(text)} characters")
    
    # Create dataset and dataloader
    dataset = TextDataset(text, seq_length=SEQ_LENGTH)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    print(f"Vocab size: {len(dataset.chars)}")
    print(f"Dataset size: {len(dataset)} sequences")
    
    # ===== Create Model =====
    model = LSTMTextGenerator(
        vocab_size=len(dataset.chars),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # ===== Train =====
    train(
        model,
        train_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # ===== Generate Text =====
    seed = "the great"
    print(f"\nGenerating text starting with: '{seed}'")
    
    for temperature in [0.5, 1.0, 1.5]:
        print(f"\nTemperature: {temperature}")
        generated = generate_text(
            model,
            dataset,
            seed,
            length=200,
            temperature=temperature,
            device=DEVICE
        )
        print(generated)
    
    # ===== Save Model =====
    torch.save(model.state_dict(), 'lstm_model.pt')
    print("\nModel saved to 'lstm_model.pt'")


