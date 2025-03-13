import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import glob
from tqdm import tqdm

class VanillaSAE(pl.LightningModule):
    """
    Vanilla Sparse Autoencoder with a single hidden layer and L1 regularization
    """
    def __init__(self, input_dim, hidden_dim, l1_coefficient=1e-3, tied_weights=True, 
                 learning_rate=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Define the encoder
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        
        # Define the decoder (if using tied weights, this weight matrix is shared with encoder)
        if self.tied_weights:
            # We'll use a lambda in forward() for the decoder
            self.decoder_bias = nn.Parameter(torch.zeros(self.input_dim))
        else:
            self.decoder = nn.Linear(self.hidden_dim, self.input_dim, bias=True)
        
        # Initialize weights
        self.init_weights()
        
        # Metrics
        self.train_l1 = []
        self.train_loss = []
        self.val_loss = []
        self.sparsity = []
    
    def init_weights(self):
        # Glorot/Xavier initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        return self.encoder(x)
    
    def activate(self, h):
        # ReLU activation for vanilla SAE
        return F.relu(h)
    
    def decode(self, h):
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t()) + self.decoder_bias
        else:
            return self.decoder(h)
    
    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
            
        # Encode
        h = self.encode(x)
        
        # Activate
        h_activated = self.activate(h)
        
        # Decode
        x_reconstructed = self.decode(h_activated)
        
        return x_reconstructed, h_activated
    
    def training_step(self, batch, batch_idx):
        # Get data
        x, _ = batch
        
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        
        # Forward pass
        x_reconstructed, h_activated = self(x)
        
        # Compute MSE reconstruction loss
        mse_loss = F.mse_loss(x_reconstructed, x)
        
        # Compute L1 sparsity penalty
        l1_penalty = self.l1_coefficient * h_activated.abs().mean()
        
        # Total loss
        loss = mse_loss + l1_penalty
        
        # Log metrics
        self.log('train_mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_l1', l1_penalty, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate sparsity metrics
        dead_units = (h_activated.abs().sum(dim=0) == 0).float().mean()
        active_units = (h_activated.abs().sum(dim=0) > 0).float().mean()
        feature_sparsity = (h_activated == 0).float().mean()
        
        self.log('dead_units', dead_units, on_step=False, on_epoch=True, prog_bar=True)
        self.log('active_units', active_units, on_step=False, on_epoch=True, prog_bar=True)
        self.log('feature_sparsity', feature_sparsity, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for later analysis
        self.train_l1.append(l1_penalty.item())
        self.train_loss.append(loss.item())
        self.sparsity.append(feature_sparsity.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get data
        x, _ = batch
        
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        
        # Forward pass
        x_reconstructed, h_activated = self(x)
        
        # Compute MSE reconstruction loss
        mse_loss = F.mse_loss(x_reconstructed, x)
        
        # Compute L1 sparsity penalty
        l1_penalty = self.l1_coefficient * h_activated.abs().mean()
        
        # Total loss
        loss = mse_loss + l1_penalty
        
        # Log metrics
        self.log('val_mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_l1', l1_penalty, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for later analysis
        self.val_loss.append(loss.item())
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Same as validation step
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def get_features(self, x):
        """
        Get sparse features/activations for new data
        """
        self.eval()
        with torch.no_grad():
            # Flatten if needed
            if len(x.shape) > 2:
                x = x.reshape(x.size(0), -1)
                
            # Encode
            h = self.encode(x)
            
            # Activate
            h_activated = self.activate(h)
            
            return h_activated


class EnformerEmbeddingDataset(Dataset):
    """
    Dataset for loading Enformer embeddings from NPZ files
    """
    def __init__(self, data_dir, split="train", species="human", max_samples=None):
        """
        Args:
            data_dir: Directory containing the embeddings
            split: One of 'train', 'valid', or 'test'
            species: One of 'human' or 'mouse'
            max_samples: Maximum number of samples to load (for debugging)
        """
        super().__init__()
        self.data_dir = os.path.join(data_dir, species, split)
        self.file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        
        if max_samples is not None:
            self.file_paths = self.file_paths[:max_samples]
            
        print(f"Found {len(self.file_paths)} embedding files in {self.data_dir}")
        
        # Load a sample file to determine embedding dimensions
        if len(self.file_paths) > 0:
            sample_data = np.load(self.file_paths[0])
            self.embedding_shape = sample_data['embedding'].shape
            print(f"Embedding shape: {self.embedding_shape}")
        else:
            self.embedding_shape = None
            print("Warning: No embedding files found!")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        # Extract only the embeddings for SAE training
        embedding = torch.from_numpy(data['embedding']).float()
        
        # For an autoencoder, input = target
        return embedding, embedding


class EnformerEmbeddingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Enformer embeddings
    """
    def __init__(self, data_dir, species="human", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.species = species
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        """
        Called on every GPU
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = EnformerEmbeddingDataset(
                data_dir=self.data_dir,
                split="train",
                species=self.species
            )
            
            self.val_dataset = EnformerEmbeddingDataset(
                data_dir=self.data_dir,
                split="valid",
                species=self.species
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = EnformerEmbeddingDataset(
                data_dir=self.data_dir,
                split="test",
                species=self.species
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 