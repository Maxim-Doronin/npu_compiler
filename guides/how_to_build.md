# Build NPU Compiler

Below you can find instructions on how to build the NPU Compiler. Please note that there are instructions for release builds, but also for developer builds. If you intend to contribute to the project, it is recommended to follow the instructions for a developer build as it comes with multiple features that are helpful during development (see the [how_to_debug.md](../src/vpux_compiler/docs/guides/how_to_debug.md) for details).

> Note: The project supports two major build types: `Compiler in Plugin` (CiP) and `Compiler in Driver` (CiD). CiP is often used for compiler development, as it does not require executing the compiler via a driver, which means that it is not necessary to have an NPU device on your machine. The instructions in this document refer to the CiP build. For a CiD build, please follow the instructions from [here](../src/vpux_driver_compiler/docs/build).

## Prerequisites

The NPU Compiler is built upon the OpenVINO framework and works with the NPU plugin that is part of the [OpenVINO repository](https://github.com/openvinotoolkit/openvino). In order to build the NPU Compiler, it is necessary to first build OpenVINO and the NPU plugin. As OpenVINO is a separate project that is being developed independently, compatibility with any version cannot be guaranteed. The exact OpenVINO commit to use can be found in the [openvino_config.json](../validation/openvino_config.json) file.

In order to prepare OpenVINO for build, clone its repository, switch to the commit found in [openvino_config.json](../validation/openvino_config.json), and update the submodules:
```bash
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout COMMIT
git submodule update --init --recursive
```

Then execute the build instructions from the sections below, depending on the desired build type.

## Build instructions for Linux

### Requirements

The following are required:

- GCC 11.4+ (or Clang 17+)
- CMake 3.20+
- [Ninja](developer_tools.md#ninja)
- [ccache](developer_tools.md#ccache) (necessary when using the developer build preset mentioned below)

All of these tools (with the exception of Clang) can be installed automatically using the `install_build_dependencies.sh` script found in the root directory of OpenVINO:

```bash
sudo -E ./install_build_dependencies.sh
```

If you intend to use Clang for your build, it is necessary to install it manually.

> Note: Older versions of these tools might work as well, but the versions listed above are the ones validated.

### Release build

#### 1. Build OpenVINO

Execute the following commands from the root directory of the OpenVINO clone:

```bash
mkdir -p build-x86_64/Release
cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_INTEL_NPU=ON \
    -DENABLE_PLUGINS_XML=ON \
    -DENABLE_INTEL_NPU_COMPILER=OFF \
    -B build-x86_64/Release
cmake --build build-x86_64/Release --target ov_dev_targets openvino_intel_npu_plugin compile_tool
```

> Note: If you intend to also build `single-image-test`, it is necessary to install OpenCV locally and specify the `-DOpenCV_DIR` build option to the directory which contains the installation's CMake configuration; alternatively, call `setup_vars_opencv4` script from OpenCV package directory. Then, when building OpenVINO, add `single-image-test` to the target list.

#### 2. Build NPU Compiler

Once OpenVINO is built, we can proceed to the actual build of the NPU compiler. A CMake preset can be used, from the root directory of the compiler clone:

```bash
OPENVINO_HOME=/path/to/root/of/openvino cmake --preset build-release .
cmake --build build-x86_64/Release
```

### Developer build

For development, it is recommended to use a Debug or RelWithDebInfo build type. The instructions below describe how to obtain a Debug build of the project. If RelWithDebInfo is preferred, replace the `Debug` symbols with `RelWithDebInfo` (e.g. `build-x86_64/Debug` -> `build-x86_64/RelWithDebInfo`), and the `developer-build-debug` preset with `developer-build-relwithdebinfo`. Unless Debug-specific capabilities are needed (e.g. complete code stepping via debuggers), a RelWithDebInfo build will likely provide a better development experience, as the execution time of the compiler will be faster.

#### 1. Build OpenVINO

Execute the following commands from the root directory of the OpenVINO clone:

```bash
mkdir -p build-x86_64/Debug
cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_INTEL_NPU=ON \
    -DENABLE_PLUGINS_XML=ON \
    -DENABLE_INTEL_NPU_COMPILER=OFF \
    -DENABLE_DEBUG_CAPS=ON \
    -DENABLE_TESTS=ON \
    -DENABLE_FUNCTIONAL_TESTS=ON \
    -B build-x86_64/Debug
cmake --build build-x86_64/Debug --target ov_dev_targets openvino_intel_npu_plugin compile_tool
```

> Note: If you intend to also build `single-image-test`, it is necessary to install OpenCV locally and specify the `-DOpenCV_DIR` build option to the directory which contains the installation's CMake configuration; alternatively, call `setup_vars_opencv4` script from OpenCV package directory. Then, when building OpenVINO, add `single-image-test` to the target list.

#### 2. Build NPU Compiler

Once OpenVINO is built, we can proceed to the actual build of the NPU compiler. A CMake preset can be used, from the root directory of the compiler clone:

```bash
OPENVINO_HOME=/path/to/root/of/openvino cmake --preset developer-build-debug .
cmake --build build-x86_64/Debug
```

##### Using Clang and mold for faster builds

The compiler can also be built using [Clang](https://clang.llvm.org) and [mold](https://github.com/rui314/mold). Clang often builds faster than GCC, it produces a smaller build, and can provide more expressive diagnostics when something goes wrong. Similarly, mold is known to be faster than GNU ld / gold and even LLD. In order to build the project with this toolchain, use the following preset:

```bash
OPENVINO_HOME=/path/to/root/of/openvino cmake --preset developer-build-fast-debug .
cmake --build build-x86_64/Debug
```

> Note: The preset expects the `clang` and `clang++` executables in your system's `PATH` variable. Some package managers may install Clang in your system as versioned executables only (e.g. `clang-20`). If that is the case, you can expose the desired version as generic executables via `update-alternatives`, or via manual symbolic links etc.

It is also possible to create your own presets, for example if you prefer using LLD instead of mold. This can be done in a cusom `CMakeUserPresets.json` file placed next to the existing [CMakePresets.json](../CMakePresets.json) file. There are helper presets which can be used to select between different toolchains, such as the `clang`, `linker-lld` or `linker-mold` presets. See how the `developer-build-fast-debug` preset is created as an example.

## Build instructions for Windows

### Requirements

The following are required:

- Microsoft Visual Studio 2022 or higher
- CMake 3.20+
- [Ninja](developer_tools.md#ninja)
- [ccache](developer_tools.md#ccache) (necessary when using the developer build preset mentioned below)

> Note: Older versions of these tools might work as well, but the versions listed above are the ones validated.

> Note: All of the build commands below should be executed from a `x64 Native Tools Command Prompt for VS` console.

### Release build

#### 1. Build OpenVINO

Execute the following commands from the root directory of the OpenVINO clone:

```bat
mkdir -p build-x86_64\Release
cmake ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DENABLE_INTEL_NPU=ON ^
    -DENABLE_PLUGINS_XML=ON ^
    -DENABLE_INTEL_NPU_COMPILER=OFF ^
    -B build-x86_64/Release
cmake --build build-x86_64\Release --target ov_dev_targets openvino_intel_npu_plugin compile_tool
```

> Note: If you intend to also build `single-image-test`, it is necessary to install OpenCV locally and specify the `-DOpenCV_DIR` build option to the directory which contains the installation's CMake configuration; alternatively, call `setup_vars_opencv4` script from OpenCV package directory. Then, when building OpenVINO, add `single-image-test` to the target list.

#### 2. Build NPU Compiler

Once OpenVINO is built, we can proceed to the actual build of the NPU compiler. A CMake preset can be used, from the root directory of the compiler clone:

```bat
set OPENVINO_HOME=path\to\root\of\openvino
cmake --preset build-release .
cmake --build build-x86_64\Release
```

### Developer build

For development, it is recommended to use a Debug or RelWithDebInfo build type. The instructions below describe how to obtain a Debug build of the project. If RelWithDebInfo is preferred, replace the `Debug` symbols with `RelWithDebInfo` (e.g. `build-x86_64\Debug` -> `build-x86_64\RelWithDebInfo`), and the `developer-build-debug` preset with `developer-build-relwithdebinfo`. Unless Debug-specific capabilities are needed (e.g. complete code stepping via debuggers), a RelWithDebInfo build will likely provide a better development experience, as the execution time of the compiler will be faster.

#### 1. Build OpenVINO

Execute the following commands from the root directory of the OpenVINO clone:

```bat
mkdir -p build-x86_64\Debug
cmake ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DENABLE_INTEL_NPU=ON ^
    -DENABLE_PLUGINS_XML=ON ^
    -DENABLE_INTEL_NPU_COMPILER=OFF ^
    -DENABLE_DEBUG_CAPS=ON ^
    -DENABLE_TESTS=ON ^
    -DENABLE_FUNCTIONAL_TESTS=ON ^
    -B build-x86_64\Debug
cmake --build build-x86_64\Debug --target ov_dev_targets openvino_intel_npu_plugin compile_tool
```

> Note: If you intend to also build `single-image-test`, it is necessary to install OpenCV locally and specify the `-DOpenCV_DIR` build option to the directory which contains the installation's CMake configuration; alternatively, call `setup_vars_opencv4` script from OpenCV package directory. Then, when building OpenVINO, add `single-image-test` to the target list.

#### 2. Build NPU Compiler

Once OpenVINO is built, we can proceed to the actual build of the NPU compiler. A CMake preset can be used, from the root directory of the compiler clone:

```bat
set OPENVINO_HOME=path\to\root\of\openvino
cmake --preset developer-build-debug .
cmake --build build-x86_64\Debug
```

> Note: If you are getting an error that "the system cannot find the file specified", please make sure that `ccache` is not available. Please make sure to add it to the system's `PATH` variable.

### Visual Studio project

When working on Windows, you may want to use Visual Studio as IDE. If that is the case, you need to configure and build the project using Visual Studio as the CMake generator instead of Ninja. The end-result will be a `.sln` file created into the build directory which can be open the project in Visual Studio.

To change the generator, you need to change the value passed to the `-G` argument of CMake. For example, in the OpenVINO command above, change `-G Ninja` to `-G ""Visual Studio 17 2022""` when configuring the project. Building the project remains unchanged (`cmake --build path\to\build`).

That being said, the instructions above for the NPU Compiler mention using a CMake preset. The presets will always use Ninja as the generator, so they cannot be used. It is however very simple to configure the project based on a preset, but make some alterations. CMake offers a view mode which will list all of the variables that are set by a preset, without actually configuring or building the project. It can be used with the `-N` argument. For example:

```bat
set OPENVINO_HOME=path\to\root\of\openvino
cmake --preset developer-build-debug -N

# Will print (at the time of writing):
Preset CMake variables:

  CMAKE_BUILD_TYPE:STRING="Debug"
  CMAKE_CXX_COMPILER_LAUNCHER:STRING="ccache"
  CMAKE_C_COMPILER_LAUNCHER:STRING="ccache"
  CMAKE_EXPORT_COMPILE_COMMANDS:BOOL="TRUE"
  ENABLE_CPPLINT:BOOL="FALSE"
  ENABLE_DEVELOPER_BUILD:BOOL="TRUE"
  ...
  OpenVINODeveloperPackage_DIR:FILEPATH="path\to\root\of\openvino/build-x86_64/Debug"
```

Based on these variables, the project can be manually configured and built to use Visual Studio as the generator:

```bat
mkdir -p build-x86_64\Debug
cmake ^
    -DCMAKE_BUILD_TYPE:STRING="Debug" ^
    -DCMAKE_CXX_COMPILER_LAUNCHER:STRING="ccache" ^
    -DCMAKE_C_COMPILER_LAUNCHER:STRING="ccache" ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL="TRUE" ^
    -DENABLE_CPPLINT:BOOL="FALSE" ^
    -DENABLE_DEVELOPER_BUILD:BOOL="TRUE" ^
    ...
    -DOpenVINODeveloperPackage_DIR:FILEPATH="path\to\root\of\openvino/build-x86_64/Debug" ^
    -G "Visual Studio 17 2022" ^
    -B build-x86_64\Debug
cmake --build build-x86_64\Debug
```

## Verifying your build

The libraries and executables created during the build will be placed in the `openvino/bin/intel64/<build-type>/` directory. You should be able to find executables such as `compile_tool` or `vpux-opt` in there.

In case you went with a developer build, tests will also be included. To check if the build was successful, you can execute some tests from this directory by running:
- `npuUnitTests` for unit tests
- `lit-tests/run_all_lit_tests.sh` for lit-tests (executable on Linux)

## Notable build options

- `ENABLE_DEVELOPER_BUILD`: This build option is enabled by the presets associated with developer builds. It enables multiple features that are useful for development, such as those seen in the [how_to_debug.md](../src/vpux_compiler/docs/guides/how_to_debug.md) document.
- `ENABLE_SPLIT_DWARF`: This build option can be set for Debug / RelWithDebInfo builds, when the Clang or GCC toolchains are used. This will separate the debug information from the compiled binaries, which can have a positive effect on the incremental build time, memory usage during build and the total storage used by the project build. This option is enabled by default by developer presets.
- `BUILD_LOG_LEVEL`: This build option can restrict the log level, so that logs below the given level are removed from the build. This eliminates the call overhead and reduces the binary size. The option takes the same string values as the `LOG_LEVEL` configuration used during execution. The default values are `LOG_INFO` for non-developer Release builds and `LOG_TRACE` for all other builds.
