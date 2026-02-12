"""
Parses command line arguments
"""

import argparse
from pathlib import Path


def parse_arguments():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description="Download models from various sources")
    parser.add_argument(
        "--compiler-type",
        type=str,
        required=True,
        choices=["PLUGIN", "DRIVER"],
        help="PLUGIN|DRIVER",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        required=True,
        help="Path to a JSON configuration file describing models to validate",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Path to a directory where to save models",
    )
    args = parser.parse_args()
    return args
