if __name__ == '__main__':
    ############################################################
    ################### 1. General setup #######################
    ################### Mostly not model specific ##############
    ############################################################
    import sys
    sys.path.append('../src')

    import argparse
    import torch
    import numpy as np
    import json
    import os
    import yaml

    import sys
    sys.path.append('./src')
    from enformer_temp import EnformerEmbeddingsDataLoader, NumpyFilesDataset, EnformerWithEmbeddings
    from SAE_models import get_cfg, TopKSAE, VanillaSAE, JumpReLUSAE, BatchTopKSAE
    from SAE_training import SAETraining

    # Simple argument parser to get config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', 
                        help='Path to the configuration YAML file')
    parser.add_argument('--override-config', type=str, default=None, 
                        help='Path to a YAML file with override configurations')
    args = parser.parse_args()
    
    # Load the main configuration file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
        
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Load override configurations if specified
    if args.override_config and os.path.exists(args.override_config):
        with open(args.override_config, 'r') as f:
            override_cfg = yaml.safe_load(f)
            config_dict.update(override_cfg)
        print(f"Loaded override configuration from {args.override_config}")
    
    # Convert to Namespace for compatibility with existing code
    args_namespace = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(args_namespace, key, value)
    
    ############################################################
    ################### 2. Training setup ######################
    ################### Not model specific #####################
    ############################################################
    
    cfg = get_cfg(**vars(args_namespace))
    
    trainer = SAETraining(cfg)


    ############################################################
    ################### 3. Data setup ##########################
    ################### Model specific #########################
    ############################################################

     # Set random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # Use the new EnformerEmbeddingsDataLoader for loading data and extracting embeddings
    device = 'cuda' if torch.cuda.is_available() and cfg.get('use_gpu', True) else 'cpu'
    target_layer = cfg.get('target_layer', 'conv_tower.5.2.to_attn_logits')
    batch_size = cfg.get('batch_size', 4)

    # Get train/val/test file indices from config
    train_indices = cfg.get('train_indices', None)  # e.g., "1-800" or [1, 2, 3, 4, 5]
    val_indices = cfg.get('val_indices', None)      # e.g., "801-900"
    test_indices = cfg.get('test_indices', None)    # e.g., "901-1000"

    train_dl = EnformerEmbeddingsDataLoader(
        data_config=cfg['train_loader'],
        target_layer=target_layer,
        batch_size=batch_size,
        device=device,
        file_indices=train_indices,
        file_prefix=cfg.get('file_prefix', 'test')
    )

    val_dl = EnformerEmbeddingsDataLoader(
        data_config=cfg['val_loader'],
        target_layer=target_layer,
        batch_size=batch_size,
        device=device,
        file_indices=val_indices,
        file_prefix=cfg.get('file_prefix', 'test')
    )

    test_dl = EnformerEmbeddingsDataLoader(
        data_config=cfg['test_loader'],
        target_layer=target_layer,
        batch_size=batch_size, 
        device=device,
        file_indices=test_indices,
        file_prefix=cfg.get('file_prefix', 'test')
    )

    ############################################################
    ################### 4. SAE Model setup #####################
    ################### Not Model specific #####################
    ############################################################


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

    # Ensure the output directory exists
    os.makedirs(cfg['outpath'], exist_ok=True)

    val_metrics = trainer.validate(val_dl)

    print(val_metrics)
    val_metrics_path = os.path.join(cfg['outpath'], f"{cfg['name']}_{cfg['seed']}_val_metrics.json")
    with open(val_metrics_path, 'w') as f:
        json.dump(val_metrics, f)

    test_metrics = trainer.test(test_dl)

    print(test_metrics)
    test_metrics_path = os.path.join(cfg['outpath'], f"{cfg['name']}_{cfg['seed']}_test_metrics.json")
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f)

