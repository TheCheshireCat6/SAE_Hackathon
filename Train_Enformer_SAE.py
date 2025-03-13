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
    from torch.utils.data import DataLoader
    import numpy as np
    import json
    import os
    import yaml
    
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

    class EnformerDataloader(DataLoader):
        def __init__(self, cfg):
            # cfg contains path            
            super().__init__(cfg)
            self.seqs, self.embed, self.pred = ...

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            return self.seqs[idx], self.embed[idx], self.pred[idx]


    # Load input seqs, embeddings and model predictions (special Dataloader)
    train_dl = EnformerDataloader(cfg['train_loader'])
    #seqs, embed, pred = train_dl[0]

    val_dl = EnformerDataloader(cfg['val_loader'])

    test_dl  = EnformerDataloader(cfg['test_loader'])    

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

    val_metrics = trainer.validate(val_dl)

    print(val_metrics)
    with open(cfg['outpath'] + f"{cfg['name']}_{cfg['seed']}_val_metrics.json", 'w') as f:
        json.dump(val_metrics, f)

    test_metrics = trainer.test(test_dl)

    print(test_metrics)
    with open(cfg['outpath'] + f"{cfg['name']}_{cfg['seed']}_test_metrics.json", 'w') as f:
        json.dump(test_metrics, f)

