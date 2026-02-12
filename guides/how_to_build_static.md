# How to build static VPUX Plugin

## Requirements
- Latest Windows SDK and spectre libraries
- CMake version 3.17 or higher
- OpenMPI Portable Hardware Locality (hwloc) version 2.5
- Intel® Threading Building Blocks (TBB) version 2020.3.2 or higher  
    **OR**      
    Intel® oneAPI Threading Building Blocks (oneTBB) version 2021.8 or higher
- To build IE Python API 1.0, you need to make sure the following python packages are installed:
  - cython>=0.29.22

## Static build for X86_64 - Windows
Static library configuration to be built as an extra module for OpenVINO.
The only available backend for the static build is `npu_level_zero_backend`.
The available compilers are `openvino_intel_npu_compiler` and `npu_driver_compiler`.
   
To select a compiler to build, use the `BUILD_COMPILER_FOR_DRIVER` CMake options. When `BUILD_COMPILER_FOR_DRIVER` is set to OFF, `openvino_intel_npu_compiler` is used, otherwise, `npu_driver_compiler` is used. Recommend using `openvino_intel_npu_compiler` when `BUILD_SHARED_LIBS` is set to `ON`, and `npu_driver_compiler` when `BUILD_SHARED_LIBS` is set to `OFF`.
To select a compiler at runtime, use the `NPU_COMPILER_TYPE` config option with values `PLUGIN` or `DRIVER`.
For a successful build please use the `Developer Command Prompt for VS 2019` or execute the following command in essential `Command Prompt` or `PowerShell` before building:

`"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"`

Actual path to the `vcvars64.bat` may vary depending on your installation ("Professional", "Community" and so on).

1. Clone OpenVINO to `%OPENVINO_HOME%` directory. Check out the branch to a specific commit, if required.  
   After that, clone OpenVINO submodules.
   ```bat
        ::set OPENVINO_HOME to appropriate location
        git clone https://github.com/openvinotoolkit/openvino.git %OPENVINO_HOME%
        cd %OPENVINO_HOME%
        git checkout <fixed-commit-id>
        git submodule update --init --recursive
   ```

2. Clone VPUXPlugin to `%VPUX_PLUGIN_HOME%` directory **OR** unpack VPUXPlugin source package to `%VPUX_PLUGIN_HOME%`.
    - set VPUX_PLUGIN_HOME to appropriate location

3. **Optional**: <a id="oneTBB-build"></a>

    To use a static library of Intel® oneAPI Threading Building Blocks (oneTBB)
    (instead of a shared version of Intel® TBB that OpenVINO provides by default):

    Clone oneTBB and build
    ```bat
        ::set TBB_HOME to appropriate location
        git clone https://github.com/oneapi-src/oneTBB.git %TBB_HOME%
        cd %TBB_HOME%
        git checkout v2021.8.0

        ::set OPENMPI_HWLOC_HOME to appropriate location
        mkdir %OPENMPI_HWLOC_HOME%
        cd %OPENMPI_HWLOC_HOME%

        set OPENMPI_HWLOC_URL=https://download.open-mpi.org/release/hwloc/v2.5/
        set OPENMPI_HWLOC_FILE=hwloc-win64-build-2.5.0
        curl --fail %OPENMPI_HWLOC_URL%%OPENMPI_HWLOC_FILE%.zip -O
        ::extract %OPENMPI_HWLOC_FILE%.zip

        set TBB_BUILD_DIR=build
        set TBB=tbb
        mkdir %TBB_HOME%\\%TBB_BUILD_DIR%
        cd %TBB_HOME%\\%TBB_BUILD_DIR%

        cmake ^
              -D TBB_TEST=OFF ^
              -D TBB_WINDOWS_DRIVER=ON ^
              -D CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
              -D CMAKE_INSTALL_PREFIX=%TBB_HOME%\\%TBB_BUILD_DIR%\\%TBB% ^
              -D TBB_ENABLE_IPO=ON ^
              -D CMAKE_HWLOC_2_5_LIBRARY_PATH=%OPENMPI_HWLOC_HOME%\\%OPENMPI_HWLOC_FILE%\\lib\\libhwloc.lib ^
              -D CMAKE_HWLOC_2_5_INCLUDE_PATH=%OPENMPI_HWLOC_HOME%\\%OPENMPI_HWLOC_FILE%\\include ^
              -D CMAKE_HWLOC_2_5_DLL_PATH=%OPENMPI_HWLOC_HOME%\\%OPENMPI_HWLOC_FILE%\\bin\\libhwloc-15.dll ^
              -D TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON ^
              -D BUILD_SHARED_LIBS=OFF ^
              ..
              cmake --build . --config Release --parallel --clean-first
              cmake --install . --config Release
    ```

4. Go to the build directory in `%OPENVINO_HOME%`:
    ```bat
    mkdir %OPENVINO_HOME%\build-x86_64
    cd %OPENVINO_HOME%\build-x86_64
    ```

5. In `Developer Command Prompt for Visual Studio`:

    **Optional**: in case of using a custom library of oneTBB (see [item #3](#oneTBB-build)):
    ```bat
    ::set TBB_HOME to the location previously set
    set TBB_BUILD_DIR=build
    set TBB=tbb
    set TBBROOT=%TBB_HOME%\\%TBB_BUILD_DIR%\\%TBB%
    set TBB_CMAKE_DIR=%TBBROOT%\\lib\\cmake\\TBB\\
    set TBB_DIR=%TBB_CMAKE_DIR%
    ```

    Build OpenVINO and VPUXPlugin:
    ```bat
    cmake ^
        -D BUILD_SHARED_LIBS=OFF ^
        -D OPENVINO_EXTRA_MODULES=%VPUX_PLUGIN_HOME% ^
        -D ENABLE_LTO=ON ^
        -D ENABLE_TESTS=OFF ^
        -D ENABLE_FUNCTIONAL_TESTS=OFF ^
        -D ENABLE_INTEL_CPU=OFF ^
        -D ENABLE_INTEL_GPU=OFF ^
        -D ENABLE_AUTO=OFF ^
        -D ENABLE_AUTO_BATCH=OFF ^
        -D ENABLE_HETERO=OFF ^
        -D ENABLE_MULTI=OFF ^
        -D ENABLE_TEMPLATE=OFF ^
        -D ENABLE_OV_ONNX_FRONTEND=OFF ^
        -D ENABLE_OV_PYTORCH_FRONTEND=OFF ^
        -D ENABLE_OV_PADDLE_FRONTEND=OFF ^
        -D ENABLE_OV_TF_FRONTEND=OFF ^
        -D ENABLE_OV_IR_FRONTEND=ON ^
        -D BUILD_COMPILER_FOR_DRIVER=OFF ^
        -D ENABLE_DRIVER_COMPILER_ADAPTER=ON ^
        -D ENABLE_ZEROAPI_BACKEND=ON ^
        -D ENABLE_TBBBIND_2_5=OFF ^
        -D CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
        -D ENABLE_OPENCV=OFF ^
        ..
        cmake --build . --target inference_engine_snippets --config Release --parallel
        cmake --build . --config Release --parallel
    ```
	For using the Accuracy Checker or another Python-based application it is recommended to set this option: `-D ENABLE_PYTHON=ON`.

6. Install built OpenVINO and VPUXPlugin libraries to `%INSTALL_DIR%` directory using CMake:
    ```bat
    ::set INSTALL_DIR to appropriate location
    cmake --install %OPENVINO_HOME%\build-x86_64 --prefix %INSTALL_DIR%
    ```

7. Link the installed OpenVINO and VPUXPlugin libraries to your application.  
    ```cmake
    find_package(OpenVINO REQUIRED)
    target_link_libraries(<application> PRIVATE openvino::runtime)
    ```
