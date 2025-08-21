# Pre-build for Driver Compiler
## Pre-build Preparation

Before building, clone the required repositories (or download and extract the source code), and set the required environment variables. Set the cloned [OpenVINO Project] as environment variable `OPENVINO_HOME`, set the cloned [NPU-Plugin Project] as environment variable `NPU_PLUGIN_HOME`, and update their submodules. Follow these steps:

<details>
<summary>Commands (Linux shell and Windows x64 Native Tools Command Prompt)</summary>

```sh
# Set working directory
export WORKDIR=$(pwd)
cd ${WORKDIR}
git clone https://github.com/openvinotoolkit/openvino.git openvino
git clone https://github.com/openvinotoolkit/npu_compiler vpux_plugin

export OPENVINO_HOME=${WORKDIR}/openvino
export NPU_PLUGIN_HOME=${WORKDIR}/vpux_plugin
# On Windows
# set OPENVINO_HOME=%WORKDIR%\openvino
# set NPU_PLUGIN_HOME=%WORKDIR%\vpux_plugin

cd ${OPENVINO_HOME}
git checkout <OpenVINO commit ID> # Replace with your desired commit, tag, or branch
git submodule update --init --recursive
cd ${NPU_PLUGIN_HOME}
git checkout develop # Replace with your desired commit, tag, or branch
git submodule update --init --recursive
```
</details>

Before proceeding with the build, it is recommended to do the following:
> - For matching corresponding commits, refer to the [commit correspondence section of OpenVINO and NPU-Plugin](./FAQ.md#corresponding-commits-required-for-openvino-and-npu-plugin).
> - Check [common clone issues](./FAQ.md#common-clone-failure-issues) and [common build issues](./FAQ.md#build-issues).
> - See [TBB selection](./FAQ.md#choosing-the-right-tbb-linux--windows) for threading library options.


## Next Steps

Linux: build with CMake Options (Recommended for the first Driver Compiler build):
* [how to build Driver Compiler on Linux](./build/build_linux.md)

Linux: build with CMake Presets:
* [how to build Driver Compiler with CMake Presets on Linux](./build/build_with_cmake_presets_linux.md)

Windows: build with CMake Options (Recommended for the first Driver Compiler build):
* [how to build Driver Compiler on Windows](./build/build_windows.md)

Windows: build with CMake Presets:
* [how to build Driver Compiler with CMake Presets on Windows](./build/build_with_cmake_presets_windows.md)


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
