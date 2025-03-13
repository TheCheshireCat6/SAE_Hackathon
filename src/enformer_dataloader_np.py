import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class NumpyFilesDataset(Dataset):
    def __init__(self, files_dir_or_pattern, limit_batches=None):
        """
        Dataset that loads data from multiple NumPy files.
        
        Args:
            files_dir_or_pattern: Directory containing NumPy files or a glob pattern
            limit_batches: Limit the number of samples for testing/debugging
        """
        # Get list of all NumPy files
        if os.path.isdir(files_dir_or_pattern):
            self.file_paths = sorted(glob.glob(os.path.join(files_dir_or_pattern, "*.np*")))
        else:
            self.file_paths = sorted(glob.glob(files_dir_or_pattern))
            
        print(f"Found {len(self.file_paths)} NumPy files")
        
        # Calculate total samples by examining the first file
        if len(self.file_paths) > 0:
            sample_data = np.load(self.file_paths[0], allow_pickle=True)
            self.samples_per_file = len(sample_data['sequence'])
        else:
            self.samples_per_file = 0
            
        self.total_samples = len(self.file_paths) * self.samples_per_file
        
        # Limit samples if necessary
        if limit_batches is not None:
            self.total_samples = min(self.total_samples, limit_batches)
            
        # Create an index mapping for fast lookup
        self.index_map = {}
        for i in range(min(self.total_samples, len(self.file_paths) * self.samples_per_file)):
            file_idx = i // self.samples_per_file
            sample_idx = i % self.samples_per_file
            self.index_map[i] = (file_idx, sample_idx)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        
        # Load data from NumPy file
        data = np.load(file_path, allow_pickle=True)
        
        # Extract sequences and keys
        sequence = torch.from_numpy(data['sequence'][sample_idx]).float()
        key = torch.from_numpy(data['target'][sample_idx]).float()
        
        return (sequence, key)


class NumpyDataModule(LightningDataModule):
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