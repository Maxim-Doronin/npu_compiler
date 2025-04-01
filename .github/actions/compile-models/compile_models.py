""" 
Compiling AI models from different sources specified in json config with the following layout:

{
  "models": [
    {
      "name": "model_id",
      "repository": "repository full name",
      "repository_type": "github|huggingface",
      "repository_path": "path/to/model/in/repo",
      "category": "MLIR/VPU4000/SILICON",
      "extra_config": {
        "NPU_COMPILER_TYPE": "MLIR",
        "NPU_PLATFORM": "VPU4000",
        "DEVICE_ID": "4000",
        "PERFORMANCE_HINT": "LATENCY"
      },
      "Compile": {
        "input_precision": "FP16",
        "output_precision": "FP16",
        "input_layout": "NCHW",
        "output_layout": "NC",
        "model_input_layout": "NCHW",
        "model_output_layout": "NC"
      }
    }
  ]
}
"""
#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

def parse_arguments():
    """ Parses command line arguments """
    parser = argparse.ArgumentParser(
        description="Compiling AI models from different sources specified in a json config."
    )
    parser.add_argument(
        "--compiler-type",
        type=str,
        required=True,
        choices=["MLIR", "DRIVER"],
        help="MLIR|DRIVER"
    )
    parser.add_argument(
        "--compile-tool",
        type=Path,
        required=True,
        help="Path to the compile_tool executable"
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
        help="Path to a directory with models to validate"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel jobs"
    )
    args = parser.parse_args()
    return args

def write_extra_config(extra_config: dict, extra_config_path: Path):
    """ Writes pairs of config keys and values to compile_tool extra config """
    print(f"Writing extra config to {extra_config_path}")
    with open(extra_config_path, "w", encoding="utf-8") as f:
        for key, value in extra_config.items():
            print(f"\t{key} {value}")
            f.write(f"{key} {value}\n")

def compile_model(model, args):
    """ Compiles a single model using compile_tool """
    if model["extra_config"]["NPU_COMPILER_TYPE"] != args.compiler_type:
        return {"status": "SKIPPED",
                "message": f"Skipping model: {model['name']}: NPU_COMPILER_TYPE mismatch, expected {args.compiler_type}"}

    model_path = args.models_dir / model["repository"] / model["repository_path"]
    if not model_path.exists():
        return {"status": "FILE_NOT_FOUND",
                "message": f"Model file not found: {model_path}"}

    extra_config_path = Path(f"{model['name']}_config.conf")
    write_extra_config(model["extra_config"], extra_config_path)

    cmd = [
        str(args.compile_tool),
        "-m", str(model_path),
        "-d", "NPU",
        "-log_level", "LOG_INFO",
        "-ip", model["Compile"]["input_precision"],
        "-op", model["Compile"]["output_precision"],
        "-il", model["Compile"]["input_layout"],
        "-iml", model["Compile"]["model_input_layout"],
        "-ol", model["Compile"]["output_layout"],
        "-oml", model["Compile"]["model_output_layout"],
        "-c", str(extra_config_path)
    ]
    cmd_str = ' '.join(cmd)
    try:
        subprocess.run(cmd, check=True)
        return {"status": "SUCCESS",
                "message": f"{model["name"]} compiled successfully\n\t{cmd_str}"}
    except subprocess.CalledProcessError as e:
        return {"status": "FAILED",
                "message": f"{model["name"]} compilation failed\n\t{cmd_str}\n{e}"}
    except Exception as e:
        return {"status": "FAILED",
                "message": f"{model["name"]} unexpected error\n\t{cmd_str}\n{e}"}

def main():
    """ Parses arguments and compiles models. """
    args = parse_arguments()

    try:
        with open(args.models_config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    models = config.get("models", [])
    if not models:
        print(f"No models found in the config {args.models_config}")
        sys.exit(0)
    print(f"Found {len(models)} models in {args.models_config} config.")
    print(f"Starting compilation with {args.jobs} parallel jobs...")

    exit_code = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(compile_model, model, args) for model in models]
        for future in as_completed(futures):
            result = future.result()
            print(result["status"], result["message"])
            if result["status"] in ["FAILED", "FILE_NOT_FOUND"] :
                exit_code = 1

    if exit_code != 0:
        print("One or more models failed. Exiting with error.")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
