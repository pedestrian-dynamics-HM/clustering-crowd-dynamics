"""Backward compatibility utilities for notebook code.

This module provides minimal utilities for older notebooks.
New code should import directly from appropriate modules:
- Config loading: from infrastructure.config_service import ...
- YAML loading: from config.configs import FullConfig
"""

import logging

import yaml

log = logging.getLogger(__name__)


def load_yaml_to_dict(value_collection_file_path) -> dict:
    """Load YAML file to dictionary (backward compatibility).

    Appends .yaml extension if not already present.
    Uses UnsafeLoader to handle Python objects like PosixPath.

    DEPRECATED: Use FullConfig._load_yaml() or FullConfig.from_yaml() instead.

    Args:
        value_collection_file_path: Path to YAML file

    Returns:
        Dictionary contents of YAML file
    """
    # Check if .yaml is part of string or not
    if value_collection_file_path[-5:] != '.yaml':
        value_collection_file_path += '.yaml'

    print(f'Loading config from {value_collection_file_path}')
    with open(value_collection_file_path) as f:
        # Use UnsafeLoader to handle Python objects like PosixPath stored in saved configs
        yaml_config = yaml.load(f.read(), Loader=yaml.UnsafeLoader)

    return yaml_config
