# How to test

## compilerTest

`compilerTest` can check the full Driver Compiler API demo. You can use the IR models to test it in Git Bash on Windows or Linux shell.
General command:
```bash
./compilerTest -m xxx.xml -d NPU.XXXX
```
Commonly used command line parameters (same command line options as [compile_tool](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/tools/compile_tool/main.cpp)):

(required) `-m model.xml`: Specifies the model path. Please ensure the model IR file is complete.

(required) `-d NPU.XXXX` : Specifies the simulated platform.

(optional) `-o output.net` : Specifies the output network name.

(optional) `-c config.file` : Uses the same configuration format as `compile_tool`.
To save the serialized IR, please use the `CID_GET_SERIALIZED_MODEL` environment variable.

## profilingTest

`profilingTest` is used to output profiling information. You can test it in Git Bash on Windows or Linux shell.

General command:
```bash
./profilingTest <blobfile>.blob profiling-0.bin
```

To get the <blobfile>.blob, please use the compilerTest or [compile_tool](https://github.com/openvinotoolkit/npu_compiler/tree/master/tools/compile_tool) of the [NPU-Plugin Project].

To get the profiling-0.bin and more profiling detail, please see **[how to use profiling.md](../../../guides/how-to-use-profiling.md)** in the [NPU-Plugin Project].


## loaderTest

`loaderTest` is used to check whether driver compiler header is available. You can test it in Git Bash on Windows or Linux shell.

General  command:
```bash
./loaderTest -v=1
./loaderTest -v=0
```
>Note: For more debug method and detail, refer to **[how to debug](../../vpux_compiler/docs/guides/how_to_debug.md)** in vpux_compiler part.


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
