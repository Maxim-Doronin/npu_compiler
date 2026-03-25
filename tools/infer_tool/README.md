# Infer Tool

## Overview

`infer_tool` is a command-line utility for running inference on OpenVINO models. It supports both OpenVINO IR models and pre-compiled blobs, and can run on various devices including CPU or NPU. For inferences on NPU, the IMD plugin can also be used.

### How It Works

1. **Model Loading**: the tool either reads an OpenVINO IR model or imports a pre-compiled blob
2. **Preprocessing**: applies user-specified precision and layout configurations via OpenVINO's `PrePostProcessor` API (if such a configuration is given)
3. **Compilation**: compiles the model for the target device (if not using a pre-compiled blob)
4. **Input Preparation**: loads input data from files or generates random inputs when no files are provided
5. **Inference**: executes an inference on the compiled model
6. **Output Dumping**: saves the output tensors to binary files

## Usage

### Basic Syntax

```bash
infer_tool -m <model_path> -d <device> [options]
```

### Required Arguments

- `-m <path>` - path to the OpenVINO XML model or pre-compiled blob (`.blob`)
- `-d <device>` - target device for inference (e.g., CPU, NPU, IMD)

> Note: When using IMD, extra environment variables need to be set when using the application. See [how_to_use_imd_plugin.md](../../guides/how_to_use_imd_plugin.md) for details.

### Optional Arguments

#### Input/Output Files

- `-i <paths>` - path(s) to input tensor file(s), separated by comma
  - if not provided, random values will be generated
  - the number of files must match the number of model inputs
- `-o <directory>` - directory where outputs will be saved
  - defaults to current directory if not specified

#### Configuration

- `-c <path>` - path to configuration file for the plugin
  - format: `KEY VALUE` (one per line)
  - lines starting with `#` are treated as comments

#### Precision Configuration

- `-ip <precision>` - set precision for all input layers
- `-op <precision>` - set precision for all output layers
- `-iop <mapping>` - set precision for specific input/output layers
  - format: `"input_name:PRECISION, output_name:PRECISION"`
  - example: `-iop "input:FP16, output:FP16"`

#### Layout Configuration

- `-il <layout>` - set tensor layout for all input layers
- `-ol <layout>` - set tensor layout for all output layers
- `-iol <mapping>` - set tensor layout for specific input/output layers
  - format: `"input_name:LAYOUT, output_name:LAYOUT"`
  - example: `-iol "input:NCHW, output:NHWC"`

#### Model Layout Configuration

- `-iml <layout>` - set model layout for all input layers
- `-oml <layout>` - set model layout for all output layers
- `-ioml <mapping>` - set model layout for specific input/output tensors
  - format: `"input_name:LAYOUT, output_name:LAYOUT"`
  - example: `-ioml "input:NCHW, output:NHWC"`

> Note: The precision and layout arguments are only applicable when the model is being compiled by the tool. In other words, it is not applicable when using a pre-compiled `.blob`.

## Examples

```bash
# Inference with random input on NPU plugin
infer_tool -m model.xml -d NPU

# Inference with random input on IMD plugin, targeting platform 5010
infer_tool -m model.xml -d IMD.5010

# Inference with custom inputs
infer_tool -m model.xml -d IMD.5010 -i input1.bin,input2.bin

# Inference with outputs stored in the directory `./results`
infer_tool -m model.xml -d IMD.5010 -o ./results

# Inference with a pre-compiled blob
infer_tool -m model.blob -d IMD.5010

# Inference with a custom precision and layout configuration
infer_tool -m model.xml -d IMD.5010 -ip FP16 -op FP16 -il NCHW

# Inference with a configuration file, where `config.txt` contains:
# NPU_COMPILER_TYPE PLUGIN
infer_tool -m model.xml -d IMD.5010 -c config.txt
```
