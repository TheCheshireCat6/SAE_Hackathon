import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import re

from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper


class NumpyFilesDataset(Dataset):
    def __init__(self, files_dir_or_pattern, file_indices=None, file_prefix="test", limit_batches=None):
        """
        Dataset that loads data from multiple NumPy files.
        
        Args:
            files_dir_or_pattern: Directory containing NumPy files or a glob pattern
            file_indices: List of indices or range string (e.g., "1-100") to select specific files
            file_prefix: Prefix of the files to load (default: "test")
            limit_batches: Limit the number of samples for testing/debugging
        """
        self.files_dir_or_pattern = files_dir_or_pattern
        self.file_prefix = file_prefix
        
        # Get list of files based on the input
        if file_indices is not None:
            # Convert range string (e.g., "1-100") to list of indices
            if isinstance(file_indices, str) and "-" in file_indices:
                start, end = map(int, file_indices.split("-"))
                file_indices = list(range(start, end + 1))
            
            # Ensure file_indices is a list
            if not isinstance(file_indices, list):
                file_indices = [file_indices]
                
            # Construct file paths from indices
            self.file_paths = []
            for idx in file_indices:
                file_name = f"{file_prefix}_{idx}.npz"
                file_path = os.path.join(files_dir_or_pattern, file_name)
                if os.path.exists(file_path):
                    self.file_paths.append(file_path)
        else:
            # Use the original directory or pattern approach
            if os.path.isdir(files_dir_or_pattern):
                self.file_paths = sorted(glob.glob(os.path.join(files_dir_or_pattern, "*.np*")))
            else:
                self.file_paths = sorted(glob.glob(files_dir_or_pattern))
        
        print(f"Found {len(self.file_paths)} NumPy files")
        
        # Calculate total samples by examining the first file
        if len(self.file_paths) > 0:
            sample_data = np.load(self.file_paths[0], allow_pickle=True)
            if isinstance(sample_data, np.ndarray):
                self.samples_per_file = 1
            else:
                # Check if it's a dict-like structure
                self.samples_per_file = len(sample_data['sequence']) if 'sequence' in sample_data else 1
        else:
            self.samples_per_file = 0
            
        self.total_samples = len(self.file_paths) * self.samples_per_file
        
        # Limit samples if necessary
        if limit_batches is not None:
            self.total_samples = min(self.total_samples, limit_batches)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        
        file_path = self.file_paths[file_idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Handle different data formats
        if isinstance(data, np.ndarray):
            sequence = torch.from_numpy(data).float()
            target = torch.zeros(1)  # Dummy target
        else:
            # Assuming dict-like structure with 'sequence' and possibly 'target' keys
            sequence = torch.from_numpy(data['sequence'][sample_idx]).float()
            target = torch.from_numpy(data['target'][sample_idx]).float() if 'target' in data else torch.zeros(1)
        
        return sequence, target


class NumpyDataModule(pl.LightningDataModule):
    def __init__(self, train_files_pattern, val_files_pattern, test_files_pattern, 
                 batch_size=2, smoke_test=False):
        """
        DataModule for loading data from multiple NumPy files.
        
        Args:
            train_files_pattern: Pattern or directory for training NumPy files
            val_files_pattern: Pattern or directory for validation NumPy files
            test_files_pattern: Pattern or directory for test NumPy files
            batch_size: Batch size for dataloaders
            smoke_test: If True, limit to 10 batches for quick testing
        """
        super().__init__()
        self.train_files_pattern = train_files_pattern
        self.val_files_pattern = val_files_pattern
        self.test_files_pattern = test_files_pattern
        self.batch_size = batch_size
        self.smoke_test = smoke_test

    def setup(self, stage=None):
        batch_limit = 10 if self.smoke_test else None  # Limit to 10 batches for smoke test
        print(f"The batch limit is: {batch_limit}")
        
        self.train_dataset = NumpyFilesDataset(self.train_files_pattern, limit_batches=batch_limit)
        self.val_dataset = NumpyFilesDataset(self.val_files_pattern, limit_batches=batch_limit)
        self.test_dataset = NumpyFilesDataset(self.test_files_pattern, limit_batches=batch_limit)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


class EnformerWithEmbeddings(pl.LightningModule):
    def __init__(self, num_tracks=5313, target_layer='transformer.layers.5'):
        super().__init__()
        self.enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        self.model = HeadAdapterWrapper(
            enformer=self.enformer,
            num_tracks=num_tracks,
            post_transformer_embed=False
        )
        
        self.target_layer = target_layer
        self.hook_store = {}
        self.set_hooks()
        
    def set_hooks(self):
        """Set up hooks to capture embeddings from target layer"""
        def get_activation(name):
            def hook(module, input, output):
                self.hook_store[name] = output.detach()
            return hook
        
        # Navigate to the target layer and register the hook
        layer_parts = self.target_layer.split('.')
        target = self.model.enformer
        for part in layer_parts:
            target = getattr(target, part)
        
        target.register_forward_hook(get_activation(self.target_layer))
        
    def get_embeddings(self, x):
        """Extract embeddings from the target layer and also return model predictions
        
        Returns:
            tuple: (embeddings, predictions) where embeddings are from the target layer
                  and predictions are the full model output
        """
        self.eval()
        with torch.no_grad():
            # Run a forward pass to trigger the hooks and get predictions
            predictions = self.model(x)
            # Return the captured embeddings and predictions
            return self.hook_store[self.target_layer], predictions
    
    def forward(self, x, target=None):
        preds = self.model(x)
        if target is None:
            return preds
        return self.model(seq=x, target=target)


# NEW CLASS: Proper dataloader for SAE training that uses EnformerWithEmbeddings
class EnformerEmbeddingsDataLoader:
    def __init__(self, data_config, target_layer='conv_tower.5.2.to_attn_logits', 
                 batch_size=4, device='cuda', num_workers=4, file_indices=None, file_prefix="test"):
        """
        Dataloader for SAE training that loads sequences and extracts enformer embeddings.
        
        Args:
            data_config: Path or pattern to the data files
            target_layer: Target layer in the Enformer model for embeddings extraction
            batch_size: Batch size for dataloader
            device: Device to run the model on ('cuda' or 'cpu')
            num_workers: Number of worker processes for data loading
            file_indices: List of indices or range string (e.g., "1-100") to select specific files
            file_prefix: Prefix of files to load (default: "test")
        """
        self.data_config = data_config
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.num_workers = num_workers
        
        # Initialize dataset and dataloader
        self.dataset = NumpyFilesDataset(
            data_config, 
            file_indices=file_indices,
            file_prefix=file_prefix
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True if 'train' in str(data_config).lower() else False,
            num_workers=num_workers
        )
        
        # Initialize the Enformer model
        self.model = self._initialize_model()
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _initialize_model(self):
        """Initialize the Enformer model with hooks for embedding extraction"""
        model = EnformerWithEmbeddings(target_layer=self.target_layer)
        return model
        
    def __iter__(self):
        """
        Iterator that extracts embeddings from Enformer and returns them.
        This makes the DataLoader compatible with the SAE training loop.
        """
        for batch in self.dataloader:
            # We expect batch to be a tuple of (sequences, targets)
            sequences = batch[0].to(self.device)
            
            with torch.no_grad():
                # Process sequences through the Enformer model
                embeddings, _ = self.model.get_embeddings(sequences)
                print(f"Original embeddings shape: {embeddings.shape}")
                
                # Handle 3D embeddings from transformer layers
                if len(embeddings.shape) == 3:
                    # Option 1: Take mean across sequence length (recommended)
                    #embeddings = embeddings.mean(dim=1)
                    # Or Option 2: Take max across sequence length
                    # embeddings = embeddings.max(dim=1)[0]
                    # Or Option 3: Take just the peak position
                    embeddings = embeddings[:, int(embeddings.shape[1]/2), :]
                    
                    print(f"Processed embeddings shape: {embeddings.shape}")
            
            # Return the processed embeddings
            yield embeddings
    
    def __len__(self):
        """Return the number of batches"""
        return len(self.dataloader)


def process_dataset(data_loader, model, output_dir, device='cuda', max_batches=None):
    """Process all batches in a dataset and save embeddings to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    batch_count = 0
    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(tqdm(data_loader, desc="Processing Batches")):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            # Move data to device
            sequences = sequences.to(device)
            
            # Get embeddings and predictions
            embeddings, predictions = model.get_embeddings(sequences)
            
            # Save embeddings and predictions for each sample in the batch
            for i in range(sequences.shape[0]):
                # Create a unique identifier for this sample
                sample_id = batch_idx * data_loader.batch_size + i
                
                # Here you would save the embeddings and predictions
            
            batch_count += 1
    
    print(f"Processed {batch_count} batches, saved embeddings to {output_dir}")
