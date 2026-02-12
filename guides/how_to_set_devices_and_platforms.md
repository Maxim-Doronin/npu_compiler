# How to set devices and platforms

For setting platform for compilation and device for inference it has to use common config parameter `DEVICE_ID`. Config parameter `NPU_PLATFORM` is currently deprecated. Parameter `DEVICE_ID` has the following possible formats:

| Format                                 | Compilation platform | Inference device |
| :-------------------------------------------  | :--------------- | :----------------- |
| empty |   Auto detection   | Auto detection |
| `platform` |   As specified    | Auto detection (for specified platform) |
| `platform.slice`|  As specified    | As specified |

## Platform for compilation

The table below contains NPU devices and corresponding NPU platform

| NPU device                             | NPU platform |
| :------------------------------------  | :----------- |
| Intel&reg; NPU (3720VE)                |   3720       |
| Intel&reg; NPU (4000)                  |   4000       |
| Intel&reg; NPU (5000)                  |   5000       |

Here are the examples:
```
compile_tool -d NPU.3720 -m model.xml -c npu.config
```
Compilation for 3720VE device

If the platform is not specified, NPU Plugin tries to determine it by analyzing all available system devices:
```
compile_tool -d NPU -m model.xml
```

If system doesn't have any devices and platform for compilation is not provided, you will get an error `No devices found - DEVICE_ID with platform is required for compilation`

## Device for inference

Here are the examples:
```
benchmark_app -d NPU -m model.xml
```
Run inference on any available NPU device
```
benchmark_app -d NPU.3720 -m model.xml
```
Run inference on any available 3720VE device
