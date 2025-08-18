# Legacy Debug Methods for Driver Compiler

>**Note**: The usage of `compilerTest` has been updated. This document describes legacy debug methods for historical reference only. For up-to-date debugging, please refer to the [main debug guide](./debug.md).

## Logs for Legacy Usage of CompilerTest

To change the compiler behavior, use a configuration file with the `compilerTest`. To change the log level, use `LOG_LEVEL="LOG_TRACE"` in the configuration file.

Example configuration for `googlenet-v1`:
```sh
--inputs_precisions="input:fp16" --inputs_layouts="input:NCHW" --outputs_precisions="InceptionV1/Logits/Predictions/Softmax:fp16" --outputs_layouts="InceptionV1/Logits/Predictions/Softmax:NC" --config NPU_PLATFORM="4000" DEVICE_ID="NPU.4000" LOG_LEVEL="LOG_TRACE" NPU_COMPILATION_MODE="DefaultHW"  NPU_COMPILATION_MODE_PARAMS="swap-transpose-with-fq=1 force-z-major-concat=1 quant-dequant-removal=1 propagate-quant-dequant=0"
```

## See Also

See the main debug guide:
* [how to debug](./debug.md)

See the main test guide:
* [how to test](./test.md)
