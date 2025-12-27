"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_event_weight(event_name: str, config: Dict[str, Any]) -> int:
    """Get weight for an event type.
    
    Args:
        event_name: Name of the event (e.g., 'View', 'AddToCart', 'Purchase')
        config: Configuration dictionary
        
    Returns:
        Event weight
    """
    event_weights = config.get('event_weights', {})
    event_name_lower = event_name.lower()
    
    # Map common event name variations
    mapping = {
        'view': 'view',
        'addtocart': 'add_to_cart',
        'add_to_cart': 'add_to_cart',
        'purchase': 'purchase',
        'buy': 'purchase'
    }
    
    mapped_name = mapping.get(event_name_lower, event_name_lower)
    return event_weights.get(mapped_name, 1)  # Default weight is 1

