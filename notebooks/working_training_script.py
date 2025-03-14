# %% [markdown]
# # Enformer Model

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# -------------------------------------------------------------------------
# 1) Enformer Wrapper for Embeddings
# -------------------------------------------------------------------------
from enformer_pytorch import Enformer
from enformer_pytorch.finetune import HeadAdapterWrapper

class EnformerWithEmbeddings(pl.LightningModule):
    """
    Wraps the Enformer model and captures a specific layer's output.
    """
    def __init__(self, num_tracks=5313, target_layer='transformer.10.1.fn.4'):
        super().__init__()
        # Pretrained Enformer
        self.enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
        
        # HeadAdapter to keep the shape consistent with existing code
        self.model = HeadAdapterWrapper(
            enformer=self.enformer,
            num_tracks=num_tracks,
            post_transformer_embed=False
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.target_layer = target_layer
        self.hook_store = {}
        self._set_hooks()

    def _set_hooks(self):
        """Register a forward hook on the user-specified target layer."""
        def get_activation(name):
            def hook(module, inp, out):
                # Detach to avoid storing gradients
                self.hook_store[name] = out.detach()
            return hook
        
        # Navigate to the target layer
        layer_parts = self.target_layer.split('.')
        target = self.model.enformer
        for part in layer_parts:
            target = getattr(target, part)
        
        # Register forward hook
        target.register_forward_hook(get_activation(self.target_layer))

    @torch.no_grad()
    def get_embeddings(self, x):
        """
        Forward pass through Enformer to populate hook_store with target layer embeddings.
        Returns (embeddings, predictions).
        """
        _ = self.model(x)  # triggers forward hooks
        emb = self.hook_store[self.target_layer]
        return emb, _

    def forward(self, x):
        return self.model(x)

# %% [markdown]
# # Enformer Dataloader

# %%
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

# %%
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Assuming your NumpyDataModule and EnformerWithEmbeddings are already imported.
# For example:
# from enformer_dataloader_np import NumpyDataModule
# from your_enformer_module import EnformerWithEmbeddings

# Set up the original data module.
data_module = NumpyDataModule(
    train_files_pattern="/home/rajesh/projects/hackathon/SAE_Hackathon/data/enformer_data_partitioned/human/train",
    val_files_pattern="/home/rajesh/projects/hackathon/SAE_Hackathon/data/enformer_data_partitioned/human/valid",
    test_files_pattern="/home/rajesh/projects/hackathon/SAE_Hackathon/data/enformer_data_partitioned/human/test",
    batch_size=16,  # Increase batch size if possible.
    smoke_test=False
)
data_module.setup()

enformer_target_layer = 'transformer.10.1.fn.4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the Enformer model and freeze it.
model = EnformerWithEmbeddings(target_layer=enformer_target_layer)
model = model.to(device)
model.eval()

def extract_embeddings(dataloader, desc):
    """Extract embeddings for all batches in a dataloader with a tqdm progress bar."""
    embedding_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc=desc):
            # Move sequences to device.
            sequences = batch[0].to(device)
            # Get embeddings.
            embeddings, _ = model.get_embeddings(sequences)
            # slice middle dimension --  emb[:, :, emb.shape[2] // 2]
            embeddings = embeddings[:, :, embeddings.shape[2] // 2]
            # Append embeddings to list (move to CPU for concatenation).
            embedding_list.append(embeddings.cpu())
    # Concatenate all embeddings.
    return torch.cat(embedding_list, dim=0)

# Extract embeddings for train, validation, and test splits.
train_embeddings = extract_embeddings(data_module.train_dataloader(), "Extracting train embeddings")
val_embeddings = extract_embeddings(data_module.val_dataloader(), "Extracting val embeddings")
test_embeddings = extract_embeddings(data_module.test_dataloader(), "Extracting test embeddings")

print("Train embeddings shape:", train_embeddings.shape)
print("Val embeddings shape:", val_embeddings.shape)
print("Test embeddings shape:", test_embeddings.shape)

# Wrap the extracted embeddings in new TensorDatasets.
train_embed_ds = TensorDataset(train_embeddings)
val_embed_ds = TensorDataset(val_embeddings)
test_embed_ds = TensorDataset(test_embeddings)

# Create new DataLoaders from the embedding datasets.
batch_size_new = 1  # Adjust as needed.
train_embed_dl = DataLoader(train_embed_ds, batch_size=batch_size_new, shuffle=True)
val_embed_dl = DataLoader(val_embed_ds, batch_size=batch_size_new)
test_embed_dl = DataLoader(test_embed_ds, batch_size=batch_size_new)

print("New DataLoaders created:")
print("Train:", len(train_embed_dl), "batches")
print("Val:", len(val_embed_dl), "batches")
print("Test:", len(test_embed_dl), "batches")


# %%
embeddings = model.get_embeddings(batch[0])

# %%
embeddings[0].shape

# %% [markdown]
# # SAE Training code

# %%
"""
Sparse autoencoder models and utilities for training.

This module contains implementations of various sparse autoencoder architectures including:
- Base autoencoder class
- BatchTopK sparse autoencoder
- TopK sparse autoencoder 
- Vanilla sparse autoencoder
- JumpReLU sparse autoencoder

The implementations are adapted from https://github.com/bartbussmann/BatchTopK/blob/main/sae.py
"""

import torch 
import torch.nn as nn
import lightning.pytorch as L
from torch.optim.optimizer import Optimizer
import torchmetrics
import torch.nn.functional as F
import torch.autograd as autograd

def get_cfg(**kwargs):
    """
    Get default configuration dictionary with optional overrides.

    Args:
        **kwargs: Keyword arguments to override default config values

    Returns:
        dict: Configuration dictionary with default values and any overrides
    """
    default_cfg = {
        "seed": 49,
        "batch_size": 4096,
        "lr": 3e-4,
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        "act_size": 768,
        "dict_size": 12288,
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,
        "warmstart_batches": 1000,
        "warmstart_start_factor": 0.0001,
        "warmstart_end_factor": 1,
        "scheduler": "RedOnPlateau",
        "weight_decay": 0.0001,
        "reduceLROnPlateau_factor": 0.1,
        "reduceLROnPlateau_patience": 4,
        "reduceLROnPlateau_threshold": 0.0001,
        "reduceLROnPlateau_cooldown": 0,
        "reduceLROnPlateau_min": 0,
        "reduceLROnPlateau_eps": 1e-08,
        "epochs": 1000,
        "training_set_batches": 1000,
        "outpath":"./out/",
        "min_delta":0,
        'patience':10,
        "accelerator":"auto",
        "devices":"auto",
        "include_checkpointing":True,
        "include_early_stopping":True,
        "track_LR":True,
        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        # for jumprelu
        "bandwidth": 0.001,
    }
    default_cfg.update(kwargs)
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    """
    Post-process configuration dictionary to add derived fields.

    Args:
        cfg (dict): Configuration dictionary

    Returns:
        dict: Configuration with added name field
    """
    cfg["name"] = f"{cfg.get('model_name','UnknownModel')}_{cfg.get('hook_point','UnknownHookPoint')}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg

class BaseAutoencoder(L.LightningModule):
    """
    Base class for autoencoder models.

    Implements core autoencoder functionality including:
    - Weight initialization
    - Input preprocessing and normalization
    - Training/validation/test loops
    - Optimizer and learning rate scheduler configuration
    - Dead feature tracking
    - Weight normalization

    Args:
        cfg (dict): Configuration dictionary containing model hyperparameters
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        torch.manual_seed(self.cfg["seed"])

        self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
            )
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.register_buffer('num_batches_not_active', torch.zeros((self.cfg["dict_size"],)))

        self.to(cfg["dtype"])

    def preprocess_input(self, x):
        """
        Preprocess input data by optionally normalizing.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (normalized_x, x_mean, x_std) if normalizing, else (x, None, None)
        """
        # convert x from list to tensor
        x = torch.cat(x, dim=0)
        if self.cfg.get("input_unit_norm", False):
            x_mean = x.mean(dim=-1, keepdim=True)
            x = x - x_mean
            if self.cfg.get("standardize_input",False):
                x_std = x.std(dim=-1, keepdim=True)
                x = x / (x_std + 1e-5)
            else:
                x_std = x.norm(dim=-1, keepdim=True)
                x = x / (x_std + 1e-5)
            return x, x_mean, x_std
        else:
            return x, None, None

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        """
        Postprocess reconstructed output by denormalizing if needed.

        Args:
            x_reconstruct (torch.Tensor): Reconstructed output tensor
            x_mean (torch.Tensor): Mean used for normalization
            x_std (torch.Tensor): Standard deviation used for normalization

        Returns:
            torch.Tensor: Denormalized output if normalization was used, else unchanged output
        """
        if self.cfg.get("input_unit_norm", False):
            x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        """
        Normalize decoder weights and their gradients to have unit norm.
        """
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        self.W_dec.data = W_dec_normed

    def update_inactive_features(self, acts):
        """
        Update tracking of inactive features.

        Args:
            acts (torch.Tensor): Feature activations tensor
        """
        self.num_batches_not_active += (acts.sum(0) == 0).float()
        self.num_batches_not_active[acts.sum(0) > 0] = 0

    def on_train_epoch_start(self):
        """Set model to training mode at start of training epoch."""
        self.train()

    def on_validation_epoch_start(self):
        """Set model to evaluation mode at start of validation epoch."""
        self.eval()

    def on_test_epoch_start(self):
        """Set model to evaluation mode at start of test epoch."""
        self.eval()

    def on_predict_epoch_start(self):
        """Set model to evaluation mode at start of prediction epoch."""
        self.eval()
    
    def training_step(self, batch, batch_idx):
        """
        Perform single training step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss value
        """
        x = batch
        output = self(x)
        loss = output["loss"]
        output = {
            "train_num_dead_features": output["num_dead_features"],
            "train_loss": output["loss"],
            "train_l1_loss": output["l1_loss"],
            "train_l2_loss": output["l2_loss"],
            "train_l0_norm": output["l0_norm"],
            "train_l1_norm": output["l1_norm"],
            "train_aux_loss": output["aux_loss"],
        }
        self.log_dict(output, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform single validation step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index
        """
        x = batch
        output = self(x)
        output = {
            "val_num_dead_features": output["num_dead_features"],
            "val_loss": output["loss"],
            "val_l1_loss": output["l1_loss"],
            "val_l2_loss": output["l2_loss"],
            "val_l0_norm": output["l0_norm"],
            "val_l1_norm": output["l1_norm"],
            "val_aux_loss": output["aux_loss"],
        }
        self.log_dict(output, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """
        Perform single test step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index
        """
        x = batch
        output = self(x)
        output = {
            "test_num_dead_features": output["num_dead_features"],
            "test_loss": output["loss"],
            "test_l1_loss": output["l1_loss"],
            "test_l2_loss": output["l2_loss"],
            "test_l0_norm": output["l0_norm"],
            "test_l1_norm": output["l1_norm"],
            "test_aux_loss": output["aux_loss"],
        }
        self.log_dict(output, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        """
        Perform single prediction step.

        Args:
            batch: Input batch
            batch_idx (int): Batch index

        Returns:
            tuple: (reconstructed_output, feature_activations)
        """
        x = batch
        output = self(x)
        return output['sae_out'], output['feature_acts']
    
    def on_before_optimizer_step(self, optimizer):
        """
        Perform operations before optimizer step.

        Args:
            optimizer: The optimizer
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.cfg["max_grad_norm"])
        self.make_decoder_weights_and_grad_unit_norm()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate schedulers.

        Returns:
            tuple: ([optimizer], [scheduler_configs])
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"], betas=(self.cfg["beta1"], self.cfg["beta2"]), weight_decay=self.cfg["weight_decay"])

        schedulers = []
        if self.cfg['warmstart_batches']>0:
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=self.cfg['warmstart_start_factor'],
                            end_factor=self.cfg['warmstart_end_factor'],
                            total_iters=self.cfg['warmstart_batches'],
                            ),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'WarmstartLinear',
            })

        if self.cfg['scheduler']=='RedOnPlateau':
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                        factor=self.cfg['reduceLROnPlateau_factor'], patience=self.cfg['reduceLROnPlateau_patience'], 
                                                                        threshold=self.cfg['reduceLROnPlateau_threshold'], threshold_mode='rel', 
                                                                        cooldown=self.cfg['reduceLROnPlateau_cooldown'], min_lr=self.cfg['reduceLROnPlateau_min'], eps=self.cfg['reduceLROnPlateau_eps']),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "val_loss",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'RedOnPlateau',
            })
        elif self.cfg['scheduler']=='OneCycleLR':
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, self.cfg['lr'], 
                                                                 epochs=self.cfg['epochs'], steps_per_epoch=self.cfg['training_set_batches']),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'OneCycleLR',
            })

        elif self.cfg['scheduler']=='CosineAnnealingLR':
            schedulers.append({
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs']*self.cfg['training_set_batches']),
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": 'CosineAnnealingLR',
            })
        
        return([optimizer],schedulers)

class BatchTopKSAE(BaseAutoencoder):
    """
    Batch-wise top-k sparse autoencoder.

    Takes top k activations across entire batch rather than per sample.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructed data and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts.flatten(), self.cfg["top_k"] * x.shape[0], dim=-1)
        acts_topk = (
            torch.zeros_like(acts.flatten())
            .scatter(-1, acts_topk.indices, acts_topk.values)
            .reshape(acts.shape)
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        """
        Calculate loss terms.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Pre-sparsity activations
            acts_topk (torch.Tensor): Sparse activations after top-k
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary of loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        """
        Calculate auxiliary loss for dead feature reactivation.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Feature activations

        Returns:
            torch.Tensor: Auxiliary loss value
        """
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

class TopKSAE(BaseAutoencoder):
    """
    Sample-wise top-k sparse autoencoder.

    Takes top k activations per sample.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructed data and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)

        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        acts_topk = torch.zeros_like(acts).scatter(
            -1, acts_topk.indices, acts_topk.values
        )
        x_reconstruct = acts_topk @ self.W_dec + self.b_dec

        self.update_inactive_features(acts_topk)
        output = self.get_loss_dict(x, x_reconstruct, acts, acts_topk, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, acts_topk, x_mean, x_std):
        """
        Calculate loss terms.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Pre-sparsity activations
            acts_topk (torch.Tensor): Sparse activations after top-k
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary of loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts_topk.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts_topk > 0).float().sum(-1).mean()
        aux_loss = self.get_auxiliary_loss(x, x_reconstruct, acts)
        loss = l2_loss + l1_loss + aux_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()
        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts_topk,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": aux_loss,
        }
        return output

    def get_auxiliary_loss(self, x, x_reconstruct, acts):
        """
        Calculate auxiliary loss for dead feature reactivation.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Feature activations

        Returns:
            torch.Tensor: Auxiliary loss value
        """
        dead_features = self.num_batches_not_active >= self.cfg["n_batches_to_dead"]
        if dead_features.sum() > 0:
            residual = x.float() - x_reconstruct.float()
            acts_topk_aux = torch.topk(
                acts[:, dead_features],
                min(self.cfg["top_k_aux"], dead_features.sum()),
                dim=-1,
            )
            acts_aux = torch.zeros_like(acts[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )
            x_reconstruct_aux = acts_aux @ self.W_dec[dead_features]
            l2_loss_aux = (
                self.cfg["aux_penalty"]
                * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            )
            return l2_loss_aux
        else:
            return torch.tensor(0, dtype=x.dtype, device=x.device)

class VanillaSAE(BaseAutoencoder):
    """
    Basic sparse autoencoder without top-k sparsification.

    Uses L1 regularization for sparsity.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructed data and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        """
        Calculate loss terms.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Feature activations
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary of loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": 0,
        }
        return output
    

class VanillaSAEEnformer(BaseAutoencoder):
    """
    Basic sparse autoencoder without top-k sparsification.

    Uses L1 regularization for sparsity.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            dict: Output dictionary containing reconstructed data and metrics
        """
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.update_inactive_features(acts)
        output = self.get_loss_dict(x, x_reconstruct, acts, x_mean, x_std)
        return output

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        """
        Calculate loss terms.

        Args:
            x (torch.Tensor): Input tensor
            x_reconstruct (torch.Tensor): Reconstructed output
            acts (torch.Tensor): Feature activations
            x_mean (torch.Tensor): Input mean for denormalization
            x_std (torch.Tensor): Input std for denormalization

        Returns:
            dict: Dictionary of loss terms and metrics
        """
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()
        l1_norm = acts.float().abs().sum(-1).mean()
        l1_loss = self.cfg["l1_coeff"] * l1_norm
        l0_norm = (acts > 0).float().sum(-1).mean()
        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0_norm,
            "l1_norm": l1_norm,
            "aux_loss": 0,
        }
        return output

import torch
import torch.nn as nn
import torch.autograd as autograd

class RectangleFunction(autograd.Function):
    """
    Custom autograd function implementing a rectangular window function.
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass returns 1 for inputs between -0.5 and 0.5, 0 otherwise.

        Args:
            ctx: Context for saving tensors
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Binary output tensor
        """
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass zeroes gradients outside window.

        Args:
            ctx: Context containing saved tensors
            grad_output (torch.Tensor): Gradient from downstream

        Returns:
            torch.Tensor: Modified gradient
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input

class JumpReLUFunction(autograd.Function):
    """
    Custom autograd function implementing JumpReLU activation.
    """

    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass implementing JumpReLU.

        Args:
            ctx: Context for saving tensors
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of threshold parameter
            bandwidth (float): Bandwidth parameter

        Returns:
            torch.Tensor: Output tensor
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for JumpReLU.

        Args:
            ctx: Context containing saved tensors
            grad_output (torch.Tensor): Gradient from downstream

        Returns:
            tuple: (dx, dlog_threshold, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLU(nn.Module):
    """
    JumpReLU module implementing learnable threshold activation.

    Args:
        feature_size (int): Number of features
        bandwidth (float): Bandwidth parameter
    """

    def __init__(self, feature_size, bandwidth):
        super(JumpReLU, self).__init__()
        self.log_threshold = nn.Parameter(torch.zeros(feature_size))
        self.bandwidth = bandwidth

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after JumpReLU activation
        """
        return JumpReLUFunction.apply(x, self.log_threshold, self.bandwidth)

class StepFunction(autograd.Function):
    """
    Custom autograd function implementing a step function.
    """

    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass implementing step function.

        Args:
            ctx: Context for saving tensors
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of threshold parameter
            bandwidth (float): Bandwidth parameter

        Returns:
            torch.Tensor: Binary output tensor
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for step function.

        Args:
            ctx: Context containing saved tensors
            grad_output (torch.Tensor): Gradient from downstream

        Returns:
            tuple: (dx, dlog_threshold, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth

class JumpReLUSAE(BaseAutoencoder):
    """
    Sparse autoencoder using JumpReLU activation.

    Args:
        cfg (dict): Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.jumprelu = JumpReLU(feature_size=cfg["dict_size"], bandwidth=cfg["bandwidth"])

    def forward(self, x, use_pre_enc_bias=False):
        x, x_mean, x_std = self.preprocess_input(x)

        if use_pre_enc_bias:
            x = x - self.b_dec

        pre_activations = torch.relu(x @ self.W_enc + self.b_enc)
        feature_magnitudes = self.jumprelu(pre_activations)

        x_reconstructed = feature_magnitudes @ self.W_dec + self.b_dec

        return self.get_loss_dict(x, x_reconstructed, feature_magnitudes, x_mean, x_std)

    def get_loss_dict(self, x, x_reconstruct, acts, x_mean, x_std):
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean()

        l0 = StepFunction.apply(acts, self.jumprelu.log_threshold, self.cfg["bandwidth"]).sum(dim=-1).mean()
        l0_loss = self.cfg["l1_coeff"] * l0
        l1_loss = l0_loss

        loss = l2_loss + l1_loss
        num_dead_features = (
            self.num_batches_not_active > self.cfg["n_batches_to_dead"]
        ).sum()

        sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)
        output = {
            "sae_out": sae_out,
            "feature_acts": acts,
            "num_dead_features": num_dead_features,
            "loss": loss,
            "l1_loss": l1_loss,
            "l2_loss": l2_loss,
            "l0_norm": l0,
            "l1_norm": l0,
            "aux_loss": 0,
        }
        return output

import lightning.pytorch as L


class SAETraining:
    """Training wrapper class for Sparse Autoencoders using PyTorch Lightning.

    This class handles training setup and execution including callbacks, logging, and training loops.

    Args:
        cfg (dict): Configuration dictionary containing training parameters
        custom_callbacks (list, optional): List of additional PyTorch Lightning callbacks. Defaults to [].

    Attributes:
        cfg (dict): Configuration dictionary
        custom_callbacks (list): List of custom callbacks
        trainer (lightning.Trainer): PyTorch Lightning trainer instance
    """

    def __init__(self, cfg, custom_callbacks=[]):
        self.cfg = cfg
        self.custom_callbacks = custom_callbacks

        L.seed_everything(self.cfg['seed'], workers=True) # sets seeds for numpy, torch and python.random.
        callbacks = []
        
        if self.cfg['include_checkpointing']:
            checkpoint_callback = L.callbacks.ModelCheckpoint(dirpath=f"{self.cfg['outpath']}", 
                                                                filename=f"{self.cfg['name']}_{self.cfg['seed']}"+"_{epoch:02d}_{val_loss:.2f}", 
                                                                save_top_k=1, monitor="val_loss", mode="min")
            callbacks = callbacks+[checkpoint_callback]

        if self.cfg['include_early_stopping']:
            early_stop_callback = L.callbacks.EarlyStopping(monitor="val_loss", min_delta=self.cfg['min_delta'], patience=self.cfg['patience'], verbose=False, mode="min")
            callbacks = callbacks+[early_stop_callback]

        if self.cfg['track_LR']:
            LR_monitor_callback = L.callbacks.LearningRateMonitor()
            callbacks.append(LR_monitor_callback)
        
        callbacks = callbacks+self.custom_callbacks

        wandb_logger = None
        if not self.cfg['name'] is None:
            arg_dict = self.cfg
            try:
                import wandb
                wandb.login()
                wandb_logger = L.loggers.WandbLogger(project=self.cfg['wandb_project'],
                                                    group = self.cfg['name'],
                                                    config=arg_dict)
            except wandb.errors.CommError:
                print('Hit wandb init error! Retrying without logging')
                wandb_logger=None

        self.trainer = L.Trainer(deterministic=True, 
                        logger=wandb_logger,
                        accelerator = self.cfg['accelerator'], devices= self.cfg['devices'],
                        max_epochs=self.cfg['epochs'], 
                        callbacks=callbacks,)
        

    def train(self, model, train_dl, val_dl):
        """Train the model.

        Args:
            model (lightning.LightningModule): Model to train
            train_dl (DataLoader): Training data loader
            val_dl (DataLoader): Validation data loader

        Returns:
            lightning.LightningModule: Trained model
        """
        self.trainer.validate(model=model, dataloaders=val_dl)
        self.trainer.fit(model, train_dl, val_dl)
        return(model)
    

    def validate(self, val_dl, model=None):
        """Run validation on the model.

        Args:
            val_dl (DataLoader): Validation data loader
            model (lightning.LightningModule, optional): Model to validate. If None, loads from best checkpoint. Defaults to None.

        Returns:
            list: Validation metrics
        """
        if model is not None:
            val_metrics = self.trainer.validate(model=model, dataloaders=val_dl)
        else:
            val_metrics = self.trainer.validate(dataloaders = val_dl, ckpt_path='best')
        return(val_metrics)

    def test(self, test_dataloader, model=None):
        """Run testing on the model.

        Args:
            test_dataloader (DataLoader): Test data loader
            model (lightning.LightningModule, optional): Model to test. If None, loads from best checkpoint. Defaults to None.

        Returns:
            list: Test metrics
        """
        if model is not None:
            test_metrics = self.trainer.test(dataloaders = test_dataloader, model=model)
        else:
            test_metrics = self.trainer.test(dataloaders = test_dataloader, ckpt_path='best')
        return(test_metrics)


# %%
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

# -----------------------------------------------------------------------------
# 1. Generate random data for training, validation, and testing.
# -----------------------------------------------------------------------------
# Each sample is a 768-dimensional vector (act_size).
# N_train, N_val, N_test = 1024, 256, 256  # adjust sample counts as needed
# act_size = 768

# # Create random datasets
# train_data = torch.randn(N_train, act_size)
# val_data   = torch.randn(N_val, act_size)
# test_data  = torch.randn(N_test, act_size)

# # Wrap the tensors in a TensorDataset (each sample is just the data)
# train_ds = TensorDataset(train_data)
# val_ds   = TensorDataset(val_data)
# test_ds  = TensorDataset(test_data)

# Create DataLoaders
batch_size = 1
act_size = 1536
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# val_dl   = DataLoader(val_ds, batch_size=batch_size)
# test_dl  = DataLoader(test_ds, batch_size=batch_size)
train_dl = DataLoader(train_embed_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_embed_ds, batch_size=batch_size)
test_dl = DataLoader(test_embed_ds, batch_size=batch_size)

# -----------------------------------------------------------------------------
# 2. Create a configuration dictionary using get_cfg
# -----------------------------------------------------------------------------
# Here we override some defaults for troubleshooting.

cfg = get_cfg(
    seed=42,
    batch_size=batch_size,
    lr=3e-4,
    l1_coeff=0.001,      # set a small L1 coefficient for sparsity (adjust as needed)
    act_size=act_size,
    dict_size=12288,
    epochs=100,           # use fewer epochs for debugging
    wandb_project="sparse_autoencoders_debug",
    include_checkpointing=False,  # disable checkpointing for quick debugging
    include_early_stopping=False, # disable early stopping
    track_LR=False
)

# -----------------------------------------------------------------------------
# 3. Initialize wandb and instantiate the VanillaSAE model and training wrapper.
# -----------------------------------------------------------------------------
# Adjust the import paths as needed.

wandb.init(project=cfg['wandb_project'], config=cfg)
model = VanillaSAE(cfg)
trainer_wrapper = SAETraining(cfg)

# -----------------------------------------------------------------------------
# 4. Train, validate, and test the model on the random data.
# -----------------------------------------------------------------------------
# Train the model.
trained_model = trainer_wrapper.train(model, train_dl, val_dl)

# Log training completion
wandb.log({"status": "training_complete"})

# Run validation and testing.
val_metrics = trainer_wrapper.validate(val_dl, model=trained_model)
print("Validation Metrics:", val_metrics)
wandb.log({"Validation Metrics": val_metrics})

test_metrics = trainer_wrapper.test(test_dl, model=trained_model)
print("Test Metrics:", test_metrics)
wandb.log({"Test Metrics": test_metrics})

# Finish the wandb run
wandb.finish()


# %%
# shape of input and output of model
print(model.input_size)
print(model.output_size)

# %%


# %%



