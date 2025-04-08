"""
Configures and runs multiple compile_tool instances in parallel
"""

import logging
import subprocess
from pathlib import Path

from common.enums import Status
from common.utils import write_extra_config


def build_compile_tool_command(
    args: dict, model: dict, model_path: Path, extra_config_path: Path
) -> list[str]:
    """Constructs compile_tool command line"""
    blob_path = args.blobs_dir / f"{model.get('name')}.blob"
    blob_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.compile_tool),
        "-m", str(model_path),
        "-o", str(blob_path),
        "-d", "NPU",
        "-log_level", "LOG_INFO",
        "-c", str(extra_config_path),
    ]
    compile_options_mapping = {
        "-ip": "input_precision",
        "-op": "output_precision",
        "-il": "input_layout",
        "-iml": "model_input_layout",
        "-ol": "output_layout",
        "-oml": "model_output_layout",
        "-shape": "shape",
    }
    for cmd_key, config_key in compile_options_mapping.items():
        if config_key in model.get("Compile", {}):
            cmd += [cmd_key, model.get("Compile").get(config_key)]
    return cmd


def compile_model(args: dict, model: dict, model_path: Path, logger: logging.Logger) -> Status:
    """Compiles a single model using compile_tool"""
    extra_config_path, extra_config_content = write_extra_config(args, model)
    logger.info("Content of %s:\n%s", extra_config_path, extra_config_content)

    cmd = build_compile_tool_command(args, model, model_path, extra_config_path)
    logger.info("Reproduction command line for compile_tool:")
    logger.info(" ".join(cmd))

    compile_tool_result = subprocess.run(
        cmd,
        check=False,
        timeout=600,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logger.info(compile_tool_result.stdout)
    if compile_tool_result.returncode != 0:
        logger.error("%s: compilation failed", model.get("name"))
        return Status.FAILED
    return Status.SUCCESS
