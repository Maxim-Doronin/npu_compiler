"""
Compiling AI models from different sources specified in json config with the following layout:

{
  "models": [
    {
        "name": "model_id",
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

import argparse
import contextlib
import json
import io
import logging
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from tqdm import tqdm

import requests
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModel,
    ResNetForImageClassification,
    CLIPProcessor,
    CLIPModel,
    SamModel,
    SamProcessor,
    ViTImageProcessor,
    ViTForImageClassification,
)
from onnxruntime.quantization import quantize_dynamic, QuantType


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
        "--jobs", type=int, required=False, default=4, help="Number of parallel jobs"
    )
    args = parser.parse_args()
    return args


class Status(Enum):
    """Enum class defining possible compilation statuses"""

    SUCCESS = 1
    SKIPPED = 2
    FAILED = 3
    TIMEOUT = 4
    MODEL_NOT_FOUND = 5

    @classmethod
    def get_ok_statuses(cls):
        """Returns a set of statuses that don't throw an error"""
        return {cls.SUCCESS, cls.SKIPPED}

    @classmethod
    def get_error_statuses(cls):
        """Returns a set of statuses that causes the script to fail"""
        return {cls.FAILED, cls.TIMEOUT, cls.MODEL_NOT_FOUND}


def setup_thread_file_logger(logs_dir: str, log_file_name: str) -> logging.Logger:
    """Configures and returns a thread-safe file-based logger"""
    logger = logging.getLogger(f"logger_{log_file_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    log_file_path = logs_dir / f"{log_file_name}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def setup_console_logger() -> logging.Logger:
    """Configures and returns a console logger"""
    logger = logging.getLogger("console_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


def write_extra_config(args: dict, model: dict) -> tuple:
    """Writes key-value pairs to compile_tool extra config"""
    extra_config_path = args.configs_dir / f"{model.get('name')}_config.conf"
    extra_config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{key} {value}\n" for key, value in model.get("extra_config", "").items()]
    content = "".join(lines)
    with open(extra_config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return extra_config_path, content


class CLIPWrapper(torch.nn.Module):
    """
    CLIPModel returns CLIPOutput objects (dict-like).
    ONNX export expects a tuple of tensors, but get tuple of dicts.
    This class wraps CLIPModel and returns unfolded outputs as a tuple of 4 tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values):
        """Unfolds CLIPOutput into tensors"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        return (
            outputs.logits_per_image,
            outputs.logits_per_text,
            outputs.text_embeds,
            outputs.image_embeds,
        )


class PromptEncoderWrapper(torch.nn.Module):
    """
    SamPromptEncoder takes extra input_boxes optional input
    that shouldn't be counted in onnx model inputs.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_points, input_labels, input_masks):
        """Adds extra input_boxes input"""
        sparse_embeddings, dense_embeddings = self.encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_masks=input_masks,
            input_boxes=None,
        )
        return sparse_embeddings, dense_embeddings


class MaskDecoderWrapper(torch.nn.Module):
    """
    SamMaskDecoder takes extra image_positional_embeddings input
    that shouldn't be counted in onnx model inputs.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        image_embeddings,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        """Adds extra image_positional_embeddings input"""
        low_res_masks, iou_predictions, _ = self.decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=torch.randn_like(image_embeddings),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
        )
        return low_res_masks, iou_predictions


def convert_pytorch_to_onnx(model: dict, torch_model_path: str):
    """Converts a PyTorch model to ONNX format"""
    pytorch_model_dir = torch_model_path.parent
    onnx_model_path = pytorch_model_dir / f"{model.get('name')}.onnx"
    model_type = model.get("model_type", "")
    convert_config = model.get("Convert", {})

    input_names = convert_config.get("input_names", [])
    output_names = convert_config.get("output_names", [])

    dynamic_axes = {}
    for name, shape in convert_config.get("dynamic_axes", {}).items():
        dynamic_axes[name] = {int(axis): type for axis, type in shape.items()}

    if model_type == "transformer":
        tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir, use_fast=True)
        pytorch_model = AutoModel.from_pretrained(pytorch_model_dir)
        pytorch_model.eval()

        text1 = convert_config.get("text1", "")
        text2 = convert_config.get("text2", "")
        inputs = tokenizer(text1, text2, return_tensors="pt")
    elif model_type == "vision_transformer":
        processor = ViTImageProcessor.from_pretrained(pytorch_model_dir, use_fast=True)
        pytorch_model = ViTForImageClassification.from_pretrained(pytorch_model_dir)
        pytorch_model.eval()

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        )
        inputs = processor(image, return_tensors="pt")
    elif model_type == "clip":
        pytorch_model = CLIPModel.from_pretrained(pytorch_model_dir)
        pytorch_model = CLIPWrapper(pytorch_model).eval()
        processor = CLIPProcessor.from_pretrained(pytorch_model_dir)

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        )
        text1 = convert_config.get("text1", "")
        inputs = processor(
            text=text1, images=[image], return_tensors="pt", padding=True
        )
    elif model_type == "sam_image_encoder":
        sam_pytorch_model = SamModel.from_pretrained(pytorch_model_dir)
        processor = SamProcessor.from_pretrained(pytorch_model_dir)
        pytorch_model = sam_pytorch_model.vision_encoder.eval()

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        ).convert("RGB")
        inputs = processor(image, return_tensors="pt")
    elif model_type == "sam_prompt_encoder":
        sam_pytorch_model = SamModel.from_pretrained(pytorch_model_dir)
        processor = SamProcessor.from_pretrained(pytorch_model_dir)
        pytorch_model = PromptEncoderWrapper(sam_pytorch_model.prompt_encoder.eval())

        point_coords_values = [float(x) for x in convert_config.get("point_coords", [])]
        point_labels_values = [int(x) for x in convert_config.get("point_labels", [])]
        point_coords = torch.tensor([[[point_coords_values]]], dtype=torch.float)
        point_labels = torch.tensor([[point_labels_values]], dtype=torch.int)
        mask_input = torch.zeros(
            tuple(convert_config.get("mask_input", [])), dtype=torch.float
        )
        inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": mask_input,
        }
    elif model_type == "sam_mask_decoder":
        sam_pytorch_model = SamModel.from_pretrained(pytorch_model_dir)
        pytorch_model = MaskDecoderWrapper(sam_pytorch_model.mask_decoder.eval())

        image_embeddings = torch.rand(convert_config.get("image_embeddings_shape", []))
        sparse_embeddings = torch.rand(
            convert_config.get("sparse_embeddings_shape", [])
        )
        dense_embeddings = torch.rand(convert_config.get("dense_embeddings_shape", []))
        multimask_output = torch.tensor([0], dtype=torch.bool)
        inputs = {
            "image_embeddings": image_embeddings,
            "sparse_embeddings": sparse_embeddings,
            "dense_embeddings": dense_embeddings,
            "multimask_output": multimask_output,
        }
    elif model_type == "resnet":
        pytorch_model = ResNetForImageClassification.from_pretrained(pytorch_model_dir)
        pytorch_model.eval()
        processor = AutoImageProcessor.from_pretrained(pytorch_model_dir, use_fast=True)

        image = Image.open(
            requests.get(
                convert_config.get("image_url", ""), stream=True, timeout=60
            ).raw
        )
        inputs = processor(image, return_tensors="pt")

    torch.onnx.export(
        pytorch_model,
        args=tuple(inputs.get(input_name) for input_name in input_names),
        f=str(onnx_model_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
    )

    return onnx_model_path


def quantize_onnx_model(model: dict, input_model_path: Path):
    """Dynamically quantizes an ONNX model"""
    output_model_path = input_model_path.parent / f"{model.get('name')}_quantized.onnx"
    quantize_dynamic(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        weight_type=QuantType.from_string(model.get("Quantize").get("quant_type")),
    )
    return output_model_path


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


def compile_model(args: dict, model: dict) -> Status:
    """Compiles a single model using compile_tool"""
    logger = setup_thread_file_logger(args.logs_dir, model.get("name"))

    if not model.get("enabled", True):
        logger.info(
            '%s: compilation skipped: because "enable" property is False',
            model.get("name"),
        )
        return Status.SKIPPED

    if model.get("extra_config").get("NPU_COMPILER_TYPE") != args.compiler_type:
        logger.info(
            "%s: compilation skipped: NPU_COMPILER_TYPE mismatch, expected %s",
            model.get("name"),
            args.compiler_type,
        )
        return Status.SKIPPED

    model_path = (
        args.models_dir / model.get("repository") / model.get("repository_path")
    )
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
    else:
        return Status.SUCCESS
    return Status.FAILED


def compile_all_models(args: dict, models: list) -> dict:
    """Runs multiple compile_tool instances in parallel"""
    future_to_model = {}
    compilation_results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for model in models:
            future = executor.submit(compile_model, args, model)
            future_to_model[future] = model

        for future in tqdm(
            as_completed(future_to_model), total=len(models), desc="Compiling"
        ):
            compilation_results[future_to_model[future].get("name")] = future.result()

    return compilation_results


def read_models_from_config(models_config: Path, logger: logging.Logger) -> list:
    """Parses a JSON config and returns a list of model configs as dictionaries"""
    with open(models_config, "r", encoding="utf-8") as f:
        config = json.load(f)
        models = config.get("models", [])

    if not models:
        logger.warning("No models found in the config %s", models_config.resolve())
        sys.exit(0)
    return models


def print_summary(compilation_results: dict, logger: logging.Logger):
    """Prints compile_tool statistics across all configured models"""
    for model, result in compilation_results.items():
        if result in Status.get_ok_statuses():
            logger.info("%s - %s", model, result)
        else:
            logger.error("%s - %s", model, result)
    logger.info("Compilation summary:")
    counter = Counter(compilation_results.values())
    for status in Status:
        logger.info("\t%s: %d", status.name, counter.get(status, 0))
    if any(counter.get(status, 0) for status in Status.get_error_statuses()):
        logger.error("Some models failed. Exiting with error.")
        sys.exit(1)


def main():
    """Main pipeline"""
    args = parse_arguments()
    logger = setup_console_logger()
    models = read_models_from_config(args.models_config, logger)

    logger.info("Found %s models in %s config", len(models), args.models_config.resolve())
    logger.info("Starting compilation with %s parallel jobs...", args.jobs)
    logger.info("Compilation logs will be saved in %s", args.logs_dir.resolve())
    logger.info("Extra compilation configs will be saved in %s", args.configs_dir.resolve())

    compilation_results = compile_all_models(args, models)
    print_summary(compilation_results, logger)


if __name__ == "__main__":
    main()
