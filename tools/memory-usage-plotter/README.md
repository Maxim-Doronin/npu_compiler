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

## Graph Description

The generated graph shows DDR heap memory usage throughout the StaticAllocation pass. The X-axis represents allocation steps and the Y-axis represents memory size in an automatically chosen unit (B, KB, MB, or GB). One allocation step corresponds to a single `allocNewBuffers` invocation in the linear scan, where all output buffers of one scheduled async operation are allocated together. The step counter increments once per invocation, so each unit on the X-axis is one such batch allocation.

The following data series are plotted:
* **Max Allocated Size** (blue line) — the high-water mark of the allocated memory address space, shown as a non-decreasing curve.
* **Max Allocated Size Increased** (magenta dots) — points where the high-water mark grew compared to the previous value.
* **Max Allocated Size Increased Due To Fragmentation** (red triangles) — a subset of the above where the increase was caused by memory fragmentation rather than by the total buffer size exceeding the previously available free space.
* **Used Memory** (red line) — the amount of memory currently occupied by live buffers.
* **Free Memory** (green line) — the amount of free memory within the currently allocated range.

Vertical dotted lines separate the timeline into regions labeled with function names. During model compilation the compiler may outline the model into multiple functions. The labels correspond to the function names:
* **part1**, **part2**, … — individual parts of the model produced by the outlining pass, sorted by part number.
* **main** — the top-level entry-point function, placed last on the timeline.
