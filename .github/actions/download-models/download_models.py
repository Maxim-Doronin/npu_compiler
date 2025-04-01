""" 
Downloading AI models from different sources specified in json config with the following layout:

{
  "models": [
    {
      "name": "model_id",
      "repository": "repository full name",
      "repository_type": "github|huggingface",
      "repository_path": "path/to/model/in/repo",
      "category": "MLIR/VPU4000/SILICON"
    }
  ]
}
"""
#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def parse_arguments():
    """ Parses command line arguments """
    parser = argparse.ArgumentParser(
        description="Download models from various sources"
    )
    parser.add_argument(
        "--compiler-type",
        type=str,
        required=True,
        choices=["MLIR", "DRIVER"],
        help="MLIR|DRIVER"
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        required=True,
        help="Path to a JSON configuration file describing models to validate"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Path to a directory where to save models"
    )
    args = parser.parse_args()
    return args

def main():
    """ Parses arguments and downloads models. """
    args = parse_arguments()

    with open(args.models_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    for model in config["models"]:
        name = model["name"]
        repository = model["repository"]
        repository_type = model["repository_type"]
        repository_path = model["repository_path"]
        compiler_type = model["extra_config"]["NPU_COMPILER_TYPE"]
        target_dir = args.models_dir / repository

        if compiler_type != args.compiler_type:
            print(f"{name}: Skipping model; NPU_COMPILER_TYPE mismatch, expected {args.compiler_type}, actual {compiler_type}")
            continue

        if os.path.exists(args.models_dir / repository / repository_path):
            print(f"{name}: Exists in {target_dir}; source: Hugging Face repository {repository}/{repository_path}")
            continue

        if repository_type == "huggingface":
            print(f"{name}: Downloading from Hugging Face repository {repository}/{repository_path} to {target_dir}")
            hf_hub_download(
                repo_id=repository,
                filename=repository_path,
                local_dir=target_dir
            )

if __name__ == "__main__":
    main()
