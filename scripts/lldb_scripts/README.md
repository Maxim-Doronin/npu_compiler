
`NPUDataFormatters.py` is a script designed to enhance the debugging experience for NPU (Neural Processing Unit) structures within LLDB. It provides custom pretty printers for specific data types used in NPU development, allowing developers to easily interpret complex data structures during debugging sessions.

This script is intended to be used alongside existing pretty printers for LLVM and MLIR, which are available in the LLVM project repository. By integrating these scripts, developers can achieve a comprehensive debugging setup tailored for NPU-related projects.

## Prerequisites

To use `NPUDataFormatters.py`, ensure you have the following:

- LLDB installed on your system.
- Access to the LLVM project repository, specifically the pretty printers for LLVM and MLIR:
  - [LLVM Pretty Printers](https://github.com/llvm/llvm-project/blob/release/19.x/llvm/utils/lldbDataFormatters.py)
  - [MLIR Pretty Printers](https://github.com/llvm/llvm-project/blob/release/19.x/mlir/utils/lldb-scripts/mlirDataFormatters.py)

## Example of usage:

Option formatter:
![image](https://github.com/user-attachments/assets/3a8af6fd-39d9-424d-83a5-62702efb7996)
Shape/ShapeRef formatter:

![image](https://github.com/user-attachments/assets/44567caa-de12-4e11-9c23-966a9fba5ba7)

DimsOrder formatter:

![image](https://github.com/user-attachments/assets/cfe2a066-7f27-432f-b15b-7c07e8a9463d)

## How to setup environment/lldb debugger
- create ~/.lldbinit file with the following content:
```python
script print("LLDB is using .lldbinit!")
script print("LLVMSupport...")
command script import <path_to_applications.ai.vpu-accelerators.vpux-plugin>/thirdparty/llvm-project/llvm/utils/lldbDataFormatters.py
script print("MLIRSupport...")
command script import <path_to_applications.ai.vpu-accelerators.vpux-plugin>/thirdparty/llvm-project/mlir/utils/lldb-scripts/mlirDataFormatters.py
script print("NPUSupport...")
command script import <path_to_applications.ai.vpu-accelerators.vpux-plugin>/scripts/lldb_scripts/NPUDataFormatters.py
script print("\tDone")
```
- configure `launch.json` to use `~/.lldbinit`:   
[CodeLLDB VSCode extention will not import ~/.lldbinit by default](https://github.com/vadimcn/codelldb/issues/367#issuecomment-706774357), but you can achieve that, by adding to VSCode configuration:
```
"lldb.launch.initCommands": ["command source ${env:HOME}/.lldbinit"]
```
or by adding the same command in `initCommands` section for every confuguration from `launch.json` config file:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "vpuxOpt-lldb",
            "type": "lldb",
            "request": "launch",
            "program": "~/src/openvino/bin/intel64/RelWithDebInfo/vpux-opt",
            "args": [
                "--vpu-arch=NPU40XX",
                "--mlir-disable-threading",
                "--host-compile",
                "path_to_file.mlir"
            ],
            "cwd": "${fileDirname}",
            "preLaunchTask": "",
            "postDebugTask": "",
            "sourceLanguages": [
                "cpp"
            ],
            "initCommands": [
                "command source ${env:HOME}/.lldbinit"
            ]
        }
    ]
}
