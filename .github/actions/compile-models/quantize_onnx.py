"""
Dynamically quantizes an ONNX model
"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(model: dict, input_model_path: Path):
    """Dynamically quantizes an ONNX model"""
    output_model_path = input_model_path.parent / f"{model.get('name')}_quantized.onnx"
    quantize_dynamic(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        weight_type=QuantType.from_string(model.get("Quantize").get("quant_type")),
    )
    return output_model_path
