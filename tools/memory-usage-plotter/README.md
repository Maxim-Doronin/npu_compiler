# Memory Usage Plotter

## Summary

Memory usage plotter is a visualization tool to plot DDR heap utilization over time from logs added in the StaticAllocation pass.

## Prerequisites

As an input the script requires logs from model compilation with log level set to TRACE. 
Minimum amount of required logs can be produced by setting the environment variables as presented:
```
OV_NPU_LOG_LEVEL="LOG_TRACE"
IE_NPU_LOG_FILTER="vpux-compiler|memory-usage-info"
```

## Usage

The tool requires two command line arguments:
`python3 plot_memory_usage.py INPUT_LOG OUTPUT_PNG`
* `INPUT_LOG` - path to the log file with name and extension
* `OUTPUT_PNG` - path to the output png file with name and extension

## Example

`python3 plot_memory_usage.py logs.txt memory_usage_plot.png`

`logs.txt` contains an example, hand crafted snippet of logs from which the plot can be generated.
