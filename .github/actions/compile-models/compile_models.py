"""
Compiling AI models from different sources specified in json config with the following layout:

{
  "models": [
    {
        "name": "model_id",
        "disabled": false,
        "repository": "repository/full_name",
        "repository_type": "github|huggingface",
        "repository_branch": "main",
        "repository_path": "path/to/model/in.repo",
        "framework": "onnx|pytorch",
        "model_type": "transormer",
        "category": "MLIR/VPU4000/SILICON",
        "extra_config": {
            "NPU_COMPILER_TYPE": "MLIR",
            "NPU_PLATFORM": "VPU4000",
            "DEVICE_ID": "4000",
            "PERFORMANCE_HINT": "LATENCY"
        },
        "Convert": {
            "text1": "Hello World!",
            "input_names": ["pixel_values"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "pixel_values": {"0": "batch_size"},
                "logits": {"0": "batch_size"}
            }
        },
        "Quantize": {
            "quant_type": "QInt8"
        },
        "Compile": {
            "input_precision": "U8",
            "output_precision": "FP32",
            "shape": "pixel_values[1,3,224,224]"
        }
    }
  ]
}
"""

#!/usr/bin/env python3

import io
import contextlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from common.cli import parse_arguments
from common.enums import Status
from common.logger import setup_console_logger, setup_thread_file_logger
from common.utils import read_models_from_config, analyze_results
from convert_pytorch_to_onnx import convert_pytorch_to_onnx
from run_compile_tool import compile_model
from quantize_onnx import quantize_onnx_model


def run_pipeline(args: dict, model: dict) -> Status:
    """Compiles a single model using compile_tool"""
    logger = setup_thread_file_logger(args.logs_dir, model.get("name"))

    if model.get("disabled", False):
        logger.info(
            '%s: compilation skipped: because "disabled" property is set True',
            model.get("name"),
        )
        return Status.DISABLED

    if model.get("extra_config").get("NPU_COMPILER_TYPE") != args.compiler_type:
        logger.info(
            "%s: compilation skipped: NPU_COMPILER_TYPE mismatch, expected %s",
            model.get("name"),
            args.compiler_type,
        )
        return Status.SKIPPED

    model_path = args.models_dir / model.get("repository") / model.get("repository_path")
    if not model_path.exists():
        logger.error(
            "%s: model file not found, expected in %s", model.get("name"), model_path
        )
        return Status.MODEL_NOT_FOUND

    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            if model.get("framework") == "pytorch":
                logger.info("PyTorch model %s will be converted to ONNX", model_path)
                model_path = convert_pytorch_to_onnx(model, model_path)
                logger.info("PyTorch model has been converted to ONNX: %s", model_path)
            if model.get("Quantize"):
                logger.info("ONNX model %s will be dynamically quantized", model_path)
                model_path = quantize_onnx_model(model, model_path)
                logger.info("ONNX model has been dynamically quantized: %s", model_path)
        logger.info("%s", buffer.getvalue().strip())
        return compile_model(args, model, model_path, logger)
    except subprocess.TimeoutExpired as e:
        logger.error("%s: compilation failed by timeout: %s", model.get("name"), e)
        return Status.TIMEOUT
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with return code %s: %s", e.returncode, e)
    except FileNotFoundError as e:
        logger.error("Command not found: %s", e)
    except OSError as e:
        logger.error("OS error occurred: %s", e)
    except BaseException as e:
        logger.error("Unrecognized error occurred: %s", e)
    return Status.FAILED


def run_parallel_pipelines(args: dict, models: list) -> dict:
    """Runs multiple compile_tool instances in parallel"""
    future_to_model = {}
    compilation_results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for model in models:
            future = executor.submit(run_pipeline, args, model)
            future_to_model[future] = model

        for future in tqdm(as_completed(future_to_model), total=len(models), desc="Compiling"):
            compilation_results[future_to_model[future].get("name")] = future.result()

    return compilation_results


def main():
    """Main pipeline"""
    args = parse_arguments()
    logger = setup_console_logger()
    models = read_models_from_config(args.models_config, logger)
    if not models:
        logger.warning("No models found in the config %s", args.models_config.resolve())
        sys.exit(0)

    logger.info("Found %s models in %s config", len(models), args.models_config.resolve())
    logger.info("Starting compilation with %s parallel jobs...", args.jobs)
    logger.info("Compilation logs will be saved in %s", args.logs_dir.resolve())
    logger.info("Extra compilation configs will be saved in %s", args.configs_dir.resolve())

    compilation_results = run_parallel_pipelines(args, models)
    if analyze_results(compilation_results, logger) in Status.get_error_statuses():
        logger.error("Some models failed. Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
