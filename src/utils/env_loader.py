"""Environment variable loader utility."""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> dict:
    """Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in project root.
        
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Look for .env in project root (two levels up from src/utils/)
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)
    
    env_vars = {}
    
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Set environment variable
                    os.environ[key] = value
                    env_vars[key] = value
    
    return env_vars


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    # Try loading .env file first
    load_env_file()
    
    return os.getenv(key, default)


# Auto-load on import
load_env_file()

