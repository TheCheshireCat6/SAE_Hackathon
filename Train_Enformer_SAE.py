if __name__ == '__main__':
    ############################################################
    ################### 1. General setup #######################
    ################### Mostly not model specific ##############
    ############################################################
    import sys
    sys.path.append('./src')
    import argparse
    import torch
    from SAE_models import get_cfg, TopKSAE, VanillaSAE, JumpReLUSAE, BatchTopKSAE
    from SAE_training import SAETraining
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import json
    import os
    
    # Enformer specific imports
    from enformer_pytorch import Enformer

    try:
        from src.config_utils import get_config
        has_config_utils = True
    except ImportError:
        has_config_utils = False

    # Check if we're using config files
    if has_config_utils and '--base-config' in sys.argv:
        # If we're using config files, get the configuration
        config_dict = get_config()
        args = argparse.Namespace()
        for key, value in config_dict.items():
            setattr(args, key, value)
    else:
        # Otherwise, use the standard argparse setup
        parser = argparse.ArgumentParser()
        #General training parameters
        parser.add_argument('--seed', type=int, default=49, help='Random seed for reproducibility')
        parser.add_argument('--batch-size', type=int, default=4096, help='Batch size for training')
        parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
        parser.add_argument('--l1-coeff', type=float, default=0, help='L1 regularization coefficient')
        parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam optimizer')
        parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 parameter for Adam optimizer')
        parser.add_argument('--max-grad-norm', type=float, default=100000, help='Maximum gradient norm for clipping')
        parser.add_argument('--act-size', type=int, default=768, help='Size of activation vectors')
        parser.add_argument('--dict-size', type=int, default=12288, help='Size of the learned dictionary')
        parser.add_argument('--wandb-project', type=str, default='sparse_autoencoders', help='Weights & Biases project name')
        parser.add_argument('--input-unit-norm', action='store_true', help='Whether input embeddings are normalized to unit norm')
        parser.add_argument('--perf-log-freq', type=int, default=1000, help='Frequency of performance logging')
        parser.add_argument('--sae-type', type=str, default='topk', help='Type of sparse autoencoder (topk, vanilla, jumprelu, or batch_topk)')
        parser.add_argument('--checkpoint-freq', type=int, default=10000, help='Frequency of model checkpointing')
        parser.add_argument('--n-batches-to-dead', type=int, default=5, help='Number of batches before considering a feature dead')
        parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type (cpu, gpu, etc)')
        parser.add_argument('--devices', type=str, default='auto', help='Device configuration')
        parser.add_argument('--no-checkpointing', action='store_false', dest='include_checkpointing', help='Disable model checkpointing')
        parser.add_argument('--no-early-stopping', action='store_false', dest='include_early_stopping', help='Disable early stopping')
        parser.add_argument('--no-lr-tracking', action='store_false', dest='track_LR', help='Disable learning rate change tracking')
        parser.add_argument('--epochs', type=int, default=1000, help='maximum number of training epochs')
        parser.add_argument('--outpath', type=str, default='./out/', help='Output directory path')
        
        #Warmstart parameters
        parser.add_argument('--warmstart-batches', type=int, default=0, help='Number of warmstart batches')
        parser.add_argument('--warmstart-start-factor', type=float, default=0.0001, help='Initial factor for warmstart')
        parser.add_argument('--warmstart-end-factor', type=float, default=1, help='Final factor for warmstart')
        
        #Learning rate scheduler parameters
        parser.add_argument('--scheduler', type=str, choices=['none','RedOnPlateau', 'CosineAnnealing','OneCycleLR'], default='RedOnPlateau', help='Learning rate scheduler type')
        parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay for optimizer')
        
        #ReduceLROnPlateau parameters
        parser.add_argument('--reduceLROnPlateau-factor', type=float, default=0.1, help='Factor by which to reduce learning rate')
        parser.add_argument('--reduceLROnPlateau-patience', type=int, default=4, help='Patience for learning rate scheduler')
        parser.add_argument('--reduceLROnPlateau-threshold', type=float, default=0.0001, help='Threshold for learning rate scheduler')
        parser.add_argument('--reduceLROnPlateau-cooldown', type=int, default=0, help='Cooldown period for learning rate scheduler')
        parser.add_argument('--reduceLROnPlateau-min', type=float, default=0, help='Minimum learning rate')
        parser.add_argument('--reduceLROnPlateau-eps', type=float, default=1e-08, help='Epsilon for learning rate scheduler')
        
        #Early stopping parameters
        parser.add_argument('--min-delta', type=float, default=0, help='Minimum change in monitored quantity for early stopping')
        parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

        # (Batch)TopKSAE specific
        parser.add_argument('--top-k', type=int, default=32, help='Number of top activations to keep')
        parser.add_argument('--top-k-aux', type=int, default=512, help='Number of top activations for auxiliary loss')
        parser.add_argument('--aux-penalty', type=float, default=1/32, help='Penalty coefficient for auxiliary loss')

        # for jumprelu
        parser.add_argument('--bandwidth', type=float, default=0.001, help='Bandwidth parameter for JumpReLU activation')

        # Enformer specific
        parser.add_argument('--enformer-dim', type=int, default=1536, help='Dimension of Enformer model')
        parser.add_argument('--hook-layer', type=str, default='transformer.0', help='Enformer layer to extract activations from')
        parser.add_argument('--sequence-length', type=int, default=196608, help='Sequence length for Enformer')
        parser.add_argument('--target-length', type=int, default=896, help='Target length for Enformer')
        parser.add_argument('--npz-data-dir', type=str, default='../enformer_data_npz', help='Directory with Enformer NPZ data files')
        parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to use (None for all)')
        parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
        parser.add_argument('--test-split', type=float, default=0.1, help='Test split ratio')
        parser.add_argument('--model-name', type=str, default='Enformer', help='Name of the model')
        
        parser.add_argument('--override-config', type=str, default=None, 
                            help='Path to a YAML file with override configurations')
        
        args = parser.parse_args()

    ############################################################
    ################### 2. Training setup ######################
    ################### Not model specific #####################
    ############################################################
    
    cfg = get_cfg(**vars(args))
    trainer = SAETraining(cfg)

    ############################################################
    ################### 3. Data setup ##########################
    ################### Model specific #########################
    ############################################################

    # Set random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # Initialize Enformer model
    enformer_model = Enformer(
        dim = cfg['enformer_dim'],
        depth = 11,                    # Number of transformer layers
        heads = 8,                     # Number of attention heads
        output_heads = dict(human = 5540),
        target_length = cfg['target_length']
    )
    
    # Set model to evaluation mode
    enformer_model.eval()
    
    # Create a hook function to capture activations from a specific layer
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    # Parse the hook layer specification and register the hook
    if cfg['hook_layer'] == 'transformer.0':
        hook = enformer_model.transformer.layers[0].register_forward_hook(hook_fn)
    elif cfg['hook_layer'] == 'transformer.5':
        hook = enformer_model.transformer.layers[5].register_forward_hook(hook_fn)
    elif cfg['hook_layer'] == 'transformer.10':
        hook = enformer_model.transformer.layers[10].register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unsupported hook layer: {cfg['hook_layer']}")
    
    # Load data from NPZ files
    def load_npz_data(data_dir, max_samples=None):
        npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if max_samples is not None:
            npz_files = npz_files[:max_samples]
        
        all_sequences = []
        all_targets = []
        
        for npz_file in npz_files:
            data = np.load(os.path.join(data_dir, npz_file))
            if 'sequence' in data and 'target' in data:
                seq = data['sequence']
                target = data['target']
                all_sequences.append(seq)
                all_targets.append(target)
            else:
                print(f"Warning: Expected keys not found in {npz_file}. Keys: {list(data.keys())}")
        
        return np.array(all_sequences), np.array(all_targets)
    
    # Load data
    sequences, targets = load_npz_data(cfg['npz_data_dir'], cfg['max_samples'])
    
    # Generate activations for each sequence
    all_activations = []
    batch_size = 4  # Use a small batch size to avoid OOM
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_tensor = torch.tensor(batch_seqs, dtype=torch.float32)
            
            # Clear previous activations
            activations.clear()
            
            # Forward pass
            _ = enformer_model(batch_tensor)
            
            # Store activations
            for act in activations:
                all_activations.append(act.cpu().numpy())
    
    # Convert to numpy arrays
    all_activations = np.concatenate(all_activations, axis=0)
    
    # Remove the hook
    hook.remove()
    
    # Create train/val/test splits
    total_samples = len(all_activations)
    indices = np.random.permutation(total_samples)
    
    test_size = int(total_samples * cfg['test_split'])
    val_size = int(total_samples * cfg['val_split'])
    train_size = total_samples - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Create PyTorch datasets and dataloaders
    train_activations = torch.tensor(all_activations[train_indices], dtype=torch.float32)
    val_activations = torch.tensor(all_activations[val_indices], dtype=torch.float32)
    test_activations = torch.tensor(all_activations[test_indices], dtype=torch.float32)
    
    train_ds = TensorDataset(train_activations)
    val_ds = TensorDataset(val_activations)
    test_ds = TensorDataset(test_activations)
    
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    test_dl = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    ############################################################
    ################### 4. SAE Model setup #####################
    ################### Not Model specific #####################
    ############################################################

    cfg['training_set_batches'] = len(train_dl)

    if cfg['sae_type'] == 'topk':
        model = TopKSAE(cfg)
    elif cfg['sae_type'] == 'vanilla':
        model = VanillaSAE(cfg)
    elif cfg['sae_type'] == 'jumprelu':
        model = JumpReLUSAE(cfg)
    elif cfg['sae_type'] == 'batch_topk':
        model = BatchTopKSAE(cfg)

    ############################################################
    ################### 5. Training ############################
    ################### Not Model specific #####################
    ############################################################

    final_model = trainer.train(model, train_dl, val_dl)

    ############################################################
    ################### 6. testing/validation ##################
    ################### Not Model specific #####################
    ############################################################

    val_metrics = trainer.validate(val_dl)

    print(val_metrics)
    with open(cfg['outpath'] + f"{cfg['name']}_{cfg['seed']}_val_metrics.json", 'w') as f:
        json.dump(val_metrics, f)

    test_metrics = trainer.test(test_dl)

    print(test_metrics)
    with open(cfg['outpath'] + f"{cfg['name']}_{cfg['seed']}_test_metrics.json", 'w') as f:
        json.dump(test_metrics, f)

    # Load override configurations if specified
    if args.override_config and os.path.exists(args.override_config):
        import yaml
        with open(args.override_config, 'r') as f:
            override_cfg = yaml.safe_load(f)
            for key, value in override_cfg.items():
                setattr(args, key, value)
        print(f"Loaded override configuration from {args.override_config}") 