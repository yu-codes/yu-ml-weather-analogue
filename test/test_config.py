#!/usr/bin/env python
"""
Test Configuration Loading

This script tests if the configuration is correctly loaded with proper types.
"""
import yaml
import sys

def main():
    """Test configuration loading"""
    # Load configuration
    with open('configs/lightning_example.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Configuration keys:", config.keys())
    
    # Check model parameters
    model_config = config.get('model', {})
    print("\nModel parameters:")
    for key, value in model_config.items():
        print(f"- {key}: {value} (type: {type(value).__name__})")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
