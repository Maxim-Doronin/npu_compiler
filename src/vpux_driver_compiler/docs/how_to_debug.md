# How to debug

## Logs

To change the compiler behavior, a configuration file can be used with the `compilerTest` tool. For example, to change the logging level, use `-log_level LOG_TRACE` in the command line or by using a configuration file. The content for the configuration file is as follows:
```
LOG_LEVEL LOG_TRACE
```

## Other tools

One can also use the tools from [NPU-Plugin Project] and [OpenVINO Project].

### compile_tool

`compile_tool` can compile a network into a blob. If you test it for Driver Compiler, you need to set the configuration option in the configuration file.

The general command on Git Bash is:
``` bash
./compile_tool -m <model_path> -d NPU.4000 -c <config_file_path>
```

Here is an example:
```bash
./compile_tool -m path/to/googlenet-v1.xml -d NPU.4000 -c /path/to/config.txt
```
where the content of config.txt is:
```bash
NPU_COMPILER_TYPE DRIVER
```

### benchmark_app

`benchmark_app` is used to estimate inference performance. If you test it for Driver Compiler, you need to set the configuration option in the configuration file.

The general command in Git Bash:
```bash
./benchmark_app -m <model_path> -load_config=<config_file_path> -d NPU.4000
```

Here is an example:
``` bash
./benchmark_app -m /path/to/mobilenet-v2.xml -load_config=/path/to/config.txt -d NPU.4000
```
where the content of config.txt is:
```
{
    "NPU" : {
        "NPU_COMPILER_TYPE" : "DRIVER", "NPU_PLATFORM" : "4000", "LOG_LEVEL" : "LOG_INFO"
    }
}
```

### timetest suite

`timetest suite` is used to measure both total and partial execution time. You can install the timetest suite by following the [time_tests/README.md](https://github.com/openvinotoolkit/openvino/blob/master/tests/time_tests/README.md). If you test it for Driver Compiler, you need to set the configuration option in the configuration file.

The general command in Git Bash:
```bash
python3 ./scripts/run_timetest.py ../../bin/intel64/Release/timetest_infer_api_2.exe -m <model_path> -d NPU.4000 -f <config_file_path>
```

Here is an example:
```bash
python3 scripts\run_timetest.py build\src\timetests\Release\timetest_infer.exe -m googlenet-v1.xml -d NPU.4000 -f config.txt
```
where the content of config.txt is:
```
NPU_COMPILER_TYPE DRIVER
```

>Note: For more debug methods and details, refer to **[how to debug](../../vpux_compiler/docs/guides/how_to_debug.md)** in the vpux_compiler section.


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
