"""
Parses command line arguments
"""

import argparse
from pathlib import Path


def parse_arguments():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compiling AI models from different sources specified in a json config."
    )
    parser.add_argument(
        "--compiler-type",
        type=str,
        required=True,
        choices=["MLIR", "DRIVER"],
        help="MLIR|DRIVER",
    )
    parser.add_argument(
        "--compile-tool",
        type=Path,
        required=True,
        help="Path to the compile_tool executable",
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
        help="Path to a directory with models to validate",
    )
    parser.add_argument(
        "--blobs-dir",
        type=Path,
        required=False,
        default=Path("./blobs"),
        help="Path to a directory to save compiled models",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        required=False,
        default=Path("./configs"),
        help="Path to a directory where to save compilation configs",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        required=False,
        default=Path("./logs"),
        help="Path to a directory where to save compilation logs",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        required=False,
        default=4,
        help="Number of parallel jobs",
    )
    args = parser.parse_args()
    return args
