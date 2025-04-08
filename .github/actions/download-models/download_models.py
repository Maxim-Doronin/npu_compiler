"""
Downloading AI models from different sources specified in json config with the following layout:

{
  "models": [
    {
      "name": "model_id",
      "repository": "repository full name",
      "repository_type": "github|huggingface",
      "repository_branch": "optional_branch",
      "repository_path": "path/to/model/in/repo",
      "framework": "pytorch|onnx",
      "extra_config": {
        "NPU_COMPILER_TYPE": "MLIR|DRIVER"
      }
    }
  ]
}
"""

#!/usr/bin/env python3

import json
import os
from pathlib import Path

from tqdm import tqdm
import requests
from huggingface_hub import hf_hub_download, snapshot_download

from common.cli import parse_arguments


def download_single_file_from_huggingface(
    repository: str, repository_path: str, target_dir: Path
):
    """Download a model from HuggingFace"""
    hf_hub_download(repo_id=repository, filename=repository_path, local_dir=target_dir)


def download_repository_from_huggingface(
    repository: str, target_dir: Path
):
    """Download a model from HuggingFace"""
    snapshot_download(repo_id=repository, local_dir=target_dir)


def download_from_github(
    repository: str, repository_branch: str, repository_path: str, target_dir: Path
):
    """Download a model from GitHub LFS using raw URL"""
    raw_url = (
        f"https://github.com/{repository}/raw/{repository_branch}/{repository_path}"
    )
    r = requests.get(raw_url, stream=True, timeout=600)
    r.raise_for_status()
    target_file_path = target_dir / repository_path
    target_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    """Parses arguments and downloads models."""
    args = parse_arguments()

    with open(args.models_config, "r", encoding="utf-8") as f:
        config = json.load(f)
        models = config["models"]

    downloading_results = {}

    for model in tqdm(models, total=len(models), desc="Downloading"):
        name = model.get("name")
        repository = model.get("repository")
        repository_type = model.get("repository_type")
        repository_branch = model.get("repository_branch")
        repository_path = model.get("repository_path")
        framework = model.get("framework")
        compiler_type = model.get("extra_config").get("NPU_COMPILER_TYPE")
        target_dir = args.models_dir / repository

        if compiler_type != args.compiler_type:
            downloading_results[name] = (
                f"Skipping model; NPU_COMPILER_TYPE mismatch, expected {args.compiler_type}, actual {compiler_type}"
            )
            continue

        if os.path.exists(args.models_dir / repository / repository_path):
            downloading_results[name] = (
                f"Skipping model; Exists in {target_dir}; source: Hugging Face repository {repository}/{repository_path}"
            )
            continue

        if repository_type == "huggingface":
            if framework == "pytorch":
                downloading_results[name] = (
                    f"Downloading from Hugging Face repository {repository} to {target_dir}"
                )
                download_repository_from_huggingface(repository, target_dir)
            else:
                downloading_results[name] = (
                    f"Downloading from Hugging Face repository {repository}/{repository_path} to {target_dir}"
                )
                download_single_file_from_huggingface(repository, repository_path, target_dir)

        if repository_type == "github":
            downloading_results[name] = (
                f"Downloading from GitHub repository {repository}/{repository_path} to {target_dir}"
            )
            download_from_github(repository, repository_branch, repository_path, target_dir)

    for name, message in downloading_results.items():
        print(f"{name} - {message}")


if __name__ == "__main__":
    main()
