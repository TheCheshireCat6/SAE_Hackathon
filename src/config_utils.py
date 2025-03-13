import os
import yaml
import argparse

def load_yaml_config(config_path):
    """Load a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(configs):
    """Merge multiple configuration dictionaries.
    
    Args:
        configs (list): List of configuration dictionaries.
        
    Returns:
        dict: Merged configuration dictionary.
    """
    merged_config = {}
    for config in configs:
        merged_config.update(config)
    return merged_config

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a sparse autoencoder using config files')
    parser.add_argument('--base-config', type=str, default='configs/base_config.yaml',
                        help='Base configuration file')
    parser.add_argument('--model-config', type=str, required=True,
                        help='Model-specific configuration file (e.g., configs/topk_sae.yaml)')
    parser.add_argument('--dataset-config', type=str, required=True,
                        help='Dataset-specific configuration file (e.g., configs/spliceai_80.yaml)')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Override configuration parameters (e.g., lr=0.001 batch_size=32)')
    
    return parser.parse_args()

def get_config():
    """Get configuration from command line arguments and YAML files.
    
    Returns:
        dict: Configuration dictionary.
    """
    args = parse_args()
    
    # Load config files
    configs = []
    if os.path.exists(args.base_config):
        configs.append(load_yaml_config(args.base_config))
    if os.path.exists(args.model_config):
        configs.append(load_yaml_config(args.model_config))
    if os.path.exists(args.dataset_config):
        configs.append(load_yaml_config(args.dataset_config))
    
    # Merge configs
    config = merge_configs(configs)
    
    # Apply overrides
    for override in args.override:
        if '=' in override:
            key, value = override.split('=', 1)
            try:
                # Try to convert to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'null' or value.lower() == 'none':
                    value = None
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                config[key] = value
            except Exception as e:
                print(f"Error processing override {override}: {e}")
    
    return config

if __name__ == '__main__':
    config = get_config()
    print(yaml.dump(config)) 