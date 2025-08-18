# Driver Compiler Build FAQ & Troubleshooting Guide
## Corresponding Commits Required for OpenVINO and NPU-Plugin

When building the Driver Compiler for Linux, ensure you use a supported OpenVINO version.

You can find the required OpenVINO version in the [release notes](https://github.com/intel/linux-npu-driver/releases) under "OpenVINO built from source" or in the [Linux NPU driver source](https://github.com/intel/linux-npu-driver/blob/main/compiler/compiler_source.cmake#L20).

### Common Clone Failure Issues

If you encounter errors like the following when cloning repositories, you may need to configure your proxy:

```sh
fatal: unable to access 'https://github.com/your-repo.git/': Could not resolve host: github.com
fatal: unable to access 'https://github.com/your-repo.git/': Failed to connect to github.com port 443: Connection timed out
fatal: unable to access 'https://github.com/your-repo.git/': Received HTTP code 407 from proxy after CONNECT
```

<details>
<summary>Set up proxy</summary>

```sh
# Linux
export http_proxy=<http://proxy_link:port>
export https_proxy=<http://proxy_link:port>
export no_proxy=<no_proxy>

# Windows
set http_proxy=<http://proxy_link:port>
set https_proxy=<http://proxy_link:port>
set no_proxy=<no_proxy>
```
</details>


## Common Build and Install Issues

For installation, it is **not recommended** to use `cmake --install . --prefix /usr --component CiD` as this will also install elf, compilerTest, and other CiD targets which are unnecessary.

### Linux Build Issues

- **Error:** `c++: internal compiler error: Killed (program cc1plus)`  
  **Cause:** Usually due to insufficient memory.  
  **Solution:** Reduce the number of parallel build jobs (e.g., use `-j4` instead of `-j8`), or increase swap space.

### Windows Build Issues

1. **Clone error: `filename too long`**  
   Run: `git config --global core.longpaths true` and enable the long path feature on Windows as follows or this [image](./imgs/long_path_enable.png):
   Open the Registry Editor, go to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`, and set the DWORD value `LongPathsEnabled` to `1`.
2. **MT/MD mismatch error:**  
   If your commit is earlier than `d0719b79c5847` (2024-08-28), do **not** add `-D CMAKE_TOOLCHAIN_FILE=%OPENVINO_HOME%\cmake\toolchains\onecoreuap.toolchain.cmake`.
3. **Ccache is required for Windows build preset:**  
   If ccache is not installed, you may see `CreateProcess failed: The system cannot find the file specified during the cmake --build ... step.`


## TBB-Related Questions
### Choosing the Right TBB (Linux & Windows)
- **OpenVINO auto-download:** OpenVINO will automatically download a prebuilt oneTBB (see `${OPENVINO_HOME}/temp/tbb`), if
    1. `ENABLE_SYSTEM_TBB` is set to `OFF`, **and** 
    2. CMake option `-D TBBROOT=/path/to/alternative/oneTBB` is **not passed**, **and** 
    3. Environment variable `TBBROOT` is **not set**.
- **System oneTBB:** To use the system TBB, set `-D ENABLE_SYSTEM_TBB=ON` for CMake.
- **Custom oneTBB:** Download or build your own oneTBB from the [oneTBB Project](https://github.com/oneapi-src/oneTBB) and set `TBBROOT` environment variable.

>Note: If you plan to use the sideloading feature on Windows, we recommend using the OneCore version of TBB. The official release version of TBB does not support OneCore, so please manually compile it according to the documentation. For instructions on how to build OneCore TBB, please see [how to build OneCore TBB](#windows-onecore-tbb-build).
- **No TBB:** Not recommended, as build speed will decrease. You can set `-D THREADING=SEQ` for CMake to disable TBB.

#### Windows OneCore TBB Build
- Build OneCore oneTBB by yourself. You must build hwloc first, then build oneTBB. See the detailed steps below.

    <details>
    <summary>1. Clone repository and build hwloc</summary>

    The `hwloc` library is a dependency of the `tbbbind_2_5` binary, so its version must be determined before proceeding with the build. The correct version can be confirmed by dumping the `tbbbind_2_5_debug.lib` binary. The corresponding `hwloc` version is `2.8.0` for oneTBB library `v2021.2.5`.
    
    Download the ZIP folder from [here](https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.8.0.zip), unzip, build the library or follow the commands below. Next, clone oneTBB and build oneTBB with MultiThreaded,

    ```bat
    set WORKDIR=%cd%

    curl -L -o hwloc-2.8.0.zip https://github.com/open-mpi/hwloc/archive/refs/tags/hwloc-2.8.0.zip
    tar -xf hwloc-2.8.0.zip
    cd hwloc-hwloc-2.8.0\contrib\windows-cmake
    cmake -A X64 --install-prefix=%cd%\install -DHWLOC_SKIP_TOOLS=ON -DHWLOC_WITH_LIBXML2=OFF -DBUILD_SHARED_LIBS=ON -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" -B build
    cmake --build build --parallel --config Release
    cmake --install build --config Release

    set HWLOC_INSTALL_DIR=%WORKDIR%\hwloc-hwloc-2.8.0\contrib\windows-cmake\install

    cd %WORKDIR%
    git clone https://github.com/uxlfoundation/oneTBB.git oneTBB-2021.2.5-static
    cd oneTBB-2021.2.5-static
    git checkout v2021.2.5
    @REM Use another directory for dynamic build
    cd ..
    xcopy oneTBB-2021.2.5-static oneTBB-2021.2.5-dynamic /E /H /I /Q

    @REM set command build option environment variable.
    @REM CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded" build with MT
    set buildOption=-D CMAKE_INSTALL_PREFIX=%cd%\install ^
        -D CMAKE_HWLOC_2_5_LIBRARY_PATH=%HWLOC_INSTALL_DIR%\lib\hwloc.lib ^
        -D CMAKE_HWLOC_2_5_INCLUDE_PATH=%HWLOC_INSTALL_DIR%\include ^
        -D CMAKE_HWLOC_2_5_DLL_PATH=%HWLOC_INSTALL_DIR%\bin\hwloc.dll ^
        -D TBB_TEST=OFF ^
        -D CMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded"
    ```
    </details>

    <details>
    <summary>2. Static oneTBB build</summary>

    ```bat
    set TBB_HOME=%WORKDIR%\oneTBB-2021.2.5-static
    cd %TBB_HOME%

    mkdir build_static
    cd build_static

    cmake ^
        -G "Visual Studio 16 2019" -A X64 ^
        -D BUILD_SHARED_LIBS=OFF ^
        %buildOption% ^
        ..

    cmake --build . --config Release
    cmake --install . --config Release
    cmake --build . --config Debug
    cmake --install . --config Debug

    copy ..\LICENSE.txt install\LICENSE
    copy %HWLOC_INSTALL_DIR%\bin\hwloc.dll install\lib
    ```
    All static oneTBB `.lib` libraries will be found in the `build_static\install` folder.
    </details>

    <details>
    <summary>3. Dynamic oneTBB build</summary>

    ```bat
    set TBB_HOME=%WORKDIR%\oneTBB-2021.2.5-dynamic

    cd %TBB_HOME%
    mkdir build_dynamic
    cd build_dynamic

    cmake ^
        -G "Visual Studio 16 2019" -A X64 ^
        -D BUILD_SHARED_LIBS=ON ^
        %buildOption% ^
        ..

    cmake --build . --config Release
    cmake --install . --config Release
    cmake --build . --config Debug
    cmake --install . --config Debug

    copy ..\LICENSE.txt install\LICENSE
    copy %HWLOC_INSTALL_DIR%\bin\hwloc.dll install\bin
    ```
    All dynamic oneTBB `.lib` and `.dll` libraries will be found in the `build_dynamic\install` folder.
    </details>

#### TBB Settings for CMake Presets

- You can set `TBBROOT` via environment variable or in the preset's `cacheVariables`.

    <details>
    <summary>Add TBBROOT option to Cmake Preset</summary>

    Adding the path to `TBBROOT` in `cacheVariables`, [under `cid` preset](../../../CMakePresets.json#L223), as shown below:

    ```json
    "name": "cid",
    "cacheVariables": {
        "TBBROOT": {
            "type": "FILEPATH",
            "value": "/path/to/alternative/oneTBB"
        }
    }
    ```
    Linux Cmake Preset usage is listed [here](./build_with_cmake_presets_linux.md).
    Windows Cmake Preset usage is listed [here](./build_with_cmake_presets_windows.md).
    </details>

- To disable TBB, set `THREADING=SEQ`.
    <details>
    <summary>Disable oneTBB commands</summary>

    To build Driver Compiler without using oneTBB library (longer model compile time), replace `"value": "TBB"` with `"value": "SEQ"` for `THREADING` in `cacheVariables` under `cid` ([see here](../../../CMakePresets.json#L228)).

    ```json
    "name": "cid",
    "cacheVariables": {
        "THREADING": {
            "type": "STRING",
            "value": "SEQ"
        },
    },
    ```
    Refer to [this document](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/cmake_options_for_custom_compilation.md#options-affecting-binary-size) for information related to SEQ threading.
    </details>

#### TBB Usage with LLVM Cache

- Any TBB version can be used when generating the LLVM cache, but when building with the cache, the parameters must match those in `build_manifest.txt`. 
- Please note that when using sideloading on Windows, you must use the OneCore version of oneTBB.


## Additional Notes

- For detailed CMake options, refer to the [CMake documentation](https://cmake.org/cmake/help/latest/index.html) and the respective `features.cmake` files in [OpenVINO](https://github.com/openvinotoolkit/openvino/blob/master/cmake/features.cmake) and [NPU-Plugin](https://github.com/openvinotoolkit/npu_compiler/blob/develop/cmake/features.cmake) repositories.


If you need further clarification or have additional questions, please refer to the main documentation in this repo or contact the development team.


[OpenVINO Project]: https://github.com/openvinotoolkit/openvino
[NPU-Plugin Project]: https://github.com/openvinotoolkit/npu_compiler
