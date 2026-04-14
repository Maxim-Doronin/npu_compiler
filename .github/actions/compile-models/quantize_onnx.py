"""
Dynamically quantizes an ONNX model
"""

import onnx
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(model: dict, input_model_path: Path):
    """Dynamically quantizes an ONNX model"""
    output_model_path = input_model_path.parent / f"{model.get('name')}_quantized.onnx"

    # Strip intermediate value_info that may contain stale shape annotations from
    # torch.onnx.export. onnxruntime's quantizer uses onnx.shape_inference.infer_shapes_path
    # in strict mode, which raises InferenceError when existing annotations conflict with
    # inferred shapes. Removing them lets onnxruntime infer all shapes from scratch.
    onnx_model = onnx.load(str(input_model_path))
    del onnx_model.graph.value_info[:]
    onnx.save(onnx_model, str(input_model_path))

    quantize_dynamic(
        model_input=str(input_model_path),
        model_output=str(output_model_path),
        weight_type=QuantType.from_string(model.get("Quantize").get("quant_type")),
    )
    return output_model_path
