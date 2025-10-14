"""Configuration loader for domain adaptation inference."""

import yaml
import argparse
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], *configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    merged = base_config.copy()
    for config in configs:
        if config:
            merged.update(config)
    return merged


def get_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Get dataset-specific configuration."""
    if 'datasets' not in config or dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
    return config['datasets'][dataset_name]


def get_strategy_config(config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """Get strategy-specific configuration."""
    if 'strategies' not in config or strategy_name not in config['strategies']:
        raise ValueError(f"Strategy '{strategy_name}' not found in configuration")
    return config['strategies'][strategy_name]


def get_scenario_config(config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Get scenario-specific configuration."""
    if 'scenarios' not in config or scenario_name not in config['scenarios']:
        raise ValueError(f"Scenario '{scenario_name}' not found in configuration")
    return config['scenarios'][scenario_name]


def get_hyperparameters(config: Dict[str, Any], dataset_name: str, forward_steps: int) -> Optional[Dict[str, Any]]:
    """Get hyperparameters for specific dataset and forward steps."""
    if 'hyperparameters' not in config or dataset_name not in config['hyperparameters']:
        return None
    
    dataset_params = config['hyperparameters'][dataset_name]
    key = f"forward_steps_{forward_steps}"
    
    return dataset_params.get(key, None)


def create_args_from_config(config_path: str, dataset: str, strategy: str = "standard", 
                          scenario: str = "source_to_target", **overrides) -> argparse.Namespace:
    """Create argparse.Namespace from configuration file."""
    config = load_config(config_path)
    
    # Get base configuration
    base_config = config.get('base', {})
    
    # Get dataset-specific configuration
    dataset_config = get_dataset_config(config, dataset)
    
    # Get strategy-specific configuration
    strategy_config = get_strategy_config(config, strategy)
    
    # Get scenario-specific configuration
    scenario_config = get_scenario_config(config, scenario)
    
    # Merge all configurations
    merged_config = merge_configs(base_config, dataset_config, strategy_config, scenario_config, overrides)
    
    # Ensure all keys are strings and values are valid
    clean_config = {}
    for key, value in merged_config.items():
        if key is None:
            continue  # Skip None keys
        if isinstance(key, str):
            clean_config[key] = value
        else:
            clean_config[str(key)] = value
    
    # Convert to argparse.Namespace
    args = argparse.Namespace(**clean_config)
    
    return args


def print_config_summary(args: argparse.Namespace):
    """Print a summary of the configuration."""
    print("=" * 60)
    print("DOMAIN ADAPTATION INFERENCE CONFIGURATION")
    print("=" * 60)
    
    print(f"Dataset: {getattr(args, 'name', 'N/A')}")
    print(f"Data Directory: {getattr(args, 'data_dir', 'N/A')}")
    print(f"Model Directory: {getattr(args, 'model_dir', 'N/A')}")
    print(f"Output Directory: {getattr(args, 'image_dir', 'N/A')}")
    print()
    
    print("Model Settings:")
    print(f"  Image Size: {getattr(args, 'image_size', 'N/A')}")
    print(f"  Channels: {getattr(args, 'num_channels', 'N/A')}")
    print(f"  Channel Multipliers: {getattr(args, 'channel_mult', 'N/A')}")
    print(f"  Attention Resolutions: {getattr(args, 'attention_resolutions', 'N/A')}")
    print()
    
    print("Domain Adaptation:")
    print(f"  Enabled: {getattr(args, 'enable_domain_adaptation', False)}")
    print(f"  Target Domain: {getattr(args, 'target_domain', 'N/A')}")
    print(f"  Domain Embedding Dim: {getattr(args, 'domain_emb_dim', 'N/A')}")
    print(f"  Number of Domains: {getattr(args, 'num_domains', 'N/A')}")
    print()
    
    print("Inference Settings:")
    print(f"  Forward Steps: {getattr(args, 'forward_steps', 'N/A')}")
    print(f"  Classifier-free Weight: {getattr(args, 'w', 'N/A')}")
    print(f"  Batch Size: {getattr(args, 'batch_size', 'N/A')}")
    print(f"  Number of Batches: {getattr(args, 'num_batches', 'N/A')}")
    print()
    
    print("Advanced Strategies:")
    dual_threshold = getattr(args, 'enable_dual_threshold', False)
    snr_weighting = getattr(args, 'enable_snr_weighting', False)
    print(f"  Dual Threshold: {dual_threshold}")
    if dual_threshold:
        print(f"    Low Offset: {getattr(args, 'low_quant_offset', 'N/A')}")
        print(f"    High Offset: {getattr(args, 'high_quant_offset', 'N/A')}")
        print(f"    Entropy Weight: {getattr(args, 'entropy_weight', 'N/A')}")
    
    print(f"  SNR Weighting: {snr_weighting}")
    if snr_weighting:
        print(f"    Temporal Decay: {getattr(args, 'temporal_decay', 'N/A')}")
        print(f"    Weight Range: [{getattr(args, 'min_weight', 'N/A')}, {getattr(args, 'max_weight', 'N/A')}]")
        print(f"    Aggregation Mode: {getattr(args, 'aggregation_mode', 'N/A')}")
    
    print("=" * 60)


def validate_config(args: argparse.Namespace) -> bool:
    """Validate configuration parameters."""
    errors = []
    
    # Check required parameters
    required_params = ['name', 'data_dir', 'model_dir', 'image_dir']
    for param in required_params:
        if not hasattr(args, param) or getattr(args, param) is None:
            errors.append(f"Missing required parameter: {param}")
    
    # Check domain adaptation parameters
    if getattr(args, 'enable_domain_adaptation', False):
        if not hasattr(args, 'target_domain'):
            errors.append("target_domain is required when domain adaptation is enabled")
        elif getattr(args, 'target_domain') not in [0, 1]:
            errors.append("target_domain must be 0 or 1")
    
    # Check strategy parameters
    if getattr(args, 'enable_dual_threshold', False):
        dual_params = ['low_quant_offset', 'high_quant_offset', 'entropy_weight', 'entropy_threshold']
        for param in dual_params:
            if not hasattr(args, param):
                errors.append(f"Missing dual threshold parameter: {param}")
    
    if getattr(args, 'enable_snr_weighting', False):
        snr_params = ['snr_smoothing', 'temporal_decay', 'min_weight', 'max_weight', 
                     'consistency_weight', 'sensitivity_weight', 'aggregation_mode']
        for param in snr_params:
            if not hasattr(args, param):
                errors.append(f"Missing SNR weighting parameter: {param}")
    
    # Print errors if any
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python config_loader.py <config_path> <dataset> [strategy] [scenario]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    dataset = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else "standard"
    scenario = sys.argv[4] if len(sys.argv) > 4 else "source_to_target"
    
    try:
        args = create_args_from_config(config_path, dataset, strategy, scenario)
        print_config_summary(args)
        
        if validate_config(args):
            print("\n✓ Configuration is valid")
        else:
            print("\n✗ Configuration validation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)