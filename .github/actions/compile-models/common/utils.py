"""
compile-models library helpers
"""

import json
import logging
from collections import Counter
from pathlib import Path

from common.enums import Status


def read_models_from_config(models_config: Path, logger: logging.Logger) -> list:
    """Parses a JSON config and returns a list of model configs as dictionaries"""
    with open(models_config, "r", encoding="utf-8") as f:
        config = json.load(f)
        models = config.get("models", [])
    return models


def write_extra_config(args: dict, model: dict) -> tuple:
    """Writes key-value pairs to compile_tool extra config"""
    extra_config_path = args.configs_dir / f"{model.get('name')}_config.conf"
    extra_config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{key} {value}\n" for key, value in model.get("extra_config", "").items()]
    content = "".join(lines)
    with open(extra_config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return extra_config_path, content


def analyze_results(compilation_results: dict, logger: logging.Logger) -> Status:
    """Prints compile_tool statistics across all configured models"""
    for model_name, result in compilation_results.items():
        if result in Status.get_ok_statuses():
            logger.info("%s - %s", result, model_name)
        else:
            logger.error("%s - %s", result, model_name)
    logger.info("Compilation summary:")
    counter = Counter(compilation_results.values())
    for status in Status:
        logger.info("\t%s: %d", status.name, counter.get(status, 0))
    if any(counter.get(status, 0) for status in Status.get_error_statuses()):
        return Status.FAILED
    return Status.SUCCESS
