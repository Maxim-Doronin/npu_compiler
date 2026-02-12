# How to use neural network profiling

Note: NPU inference profiling functionality is EXPERIMENTAL and may be limited or incomplete depending on the NPU generation.<br />
See [below](#limitations-and-caveats) for more details.

## Overview

NPU implements OpenVINO inference time profiling also referred to as performance counters and controlled with
`PERF_COUNT` property. The OpenVINO API returns compute and real execution times per OpenVINO operation (NN model
layer).

For development purposes task-level profiling information can be produced as well.

## Theory of operation

If requested the NPU compiler will instrument the compiled model with profiling primitives utilizing NPU HW and FW
components to collect runtime information into a memory buffer constituting an implicit model output. Compiler will also
append a profiling metadata section describing the layout of the profiling output, layer and task names and auxiliary
information.

After inference, if the profiling information is requested by the application, post-processing code will be invoked to
decode the profiling output buffer and return layer execution information. Depending on `NPU_PRINT_PROFILING`
environment variable a task profiling report may be generated on disk at the same time.

Since OpenVINO operations are not directly executed on the NPU and may be lowered into multiple NPU tasks, the operation
(layer) execution times are re-constructed from the task timestamps collected during the inference.

## Configuration

### NPU plugin properties

Boolean `PERF_COUNT` property must be set to `YES` during model compilation to enable the profiling instrumentation.<br />
Note: This is implied when compiling a model using `benchmark_app` or `single-image-test` if `-pc` flag is given.

### Environment variables

#### `NPU_PRINT_PROFILING`
Will write the task profiling output to a file if set. The variable accepts the following values:
* `JSON`. Generates a Perfetto report in Trace Event Format.
* `RAW`. Stores raw profiling output into a binary file (see *prof_parser* below).

Note that for the profiling output to be generated `PERF_COUNT` property must be set to `YES` and
*InferRequest::get_profiling_info()* API must be called by the application.

#### `NPU_PROFILING_OUTPUT_FILE`
Specifies a path to profiling output file to be written to. By default this is *profiling.json* in current working
directory if `NPU_PRINT_PROFILING` is set to JSON, and *profiling.out* otherwise.

#### `NPU_PROFILING_VERBOSITY`
Only relevant for JSON report. Will include every DPU variant information if set to `HIGH`.

### Compiler options

`NPU_COMPILATION_MODE_PARAMS` plugin property can be used to set the following compiler options for advanced control
over profiling instrumentation, used mostly for debug:

 Option         | Type | Default | Description
----------------|------|---------|-------------
`profiling`     | bool | false   | Enables profiling output and passes (equivalent to *PERF_COUNT YES*).
`dpu-profiling` | bool | true    | Enables DPU profiling.
`sw-profiling`  | bool | true    | Enables Shave (SW layer) profiling.
`dma-profiling` | bool | true*   | Enables DMA profiling.**
`m2i-profiling` | bool | true    | Enables M2I task profiling (available starting NPU50XX)

*&nbsp;except NPU40xx where DMA profiling is disabled by default
**&nbsp;DMA profiling is required for timestamp synchronization on NPU37xx

Disabling any of the above options may lead to incomplete layer execution information.<br />
Note: profiling instrumentation is supported only in DefaultHW compilation pipeline.

## How to use with *benchmark_app*

```sh
export NPU_PRINT_PROFILING=JSON
$ benchmark_app -d NPU -m $MODEL -pc -niter 1 -nireq 1
```
Layer profiling information will be printed by *benchmark_app* in step 11.<br />
Set *NPU_PRINT_PROFILING* variable to generate task-level Perfetto report in *profiling.json*.

## How to view task profiling report

Open JSON file either in:

* [Perfetto UI](https://ui.perfetto.dev/),
* [`chrome://tracing`](chrome://tracing/) build-in the Google Chrome browser.

### Format details

The JSON report includes NPU task schedule and reconstructed layer timing.
See [Schedule trace file](./how_to_get_schedule_trace_and_analysis.md#schedule-trace-file) for task format description.
The report is augmented with task statistic, see [Tasks statistics](./how_to_get_schedule_trace_and_analysis.md#tasks-statistics).

JSON report follows [Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview).

## prof_parser
Profiling parser is a standalone tool for decoding the profiling output based on the profiling metadata embedded in the
compiled blob.

Command line syntax:
`prof_parser -b BLOB  OPTIONS`

The following OPTIONS are available:

* `-h` prints usage information,
* `-p FILENAME` parses given profiling output binary file (e.g. *profiling.out* or *profiling-0.bin*)
* `-f FORMAT` selects the output format, where FORMAT is:
  * `json` (default) Perfetto JSON task profiling report, see [format details](#format-details),
  * `debug` raw task counters (debug purpose only),
  * `text` text output.
* `-o OUTPUT` output file, default is *stdout*.
* `-m` decodes the profiling metadata section from the provided blob (mutually exclusive with `-p`)

Note: *prof_parser* build should match the compiler used to generate the compiled blob.

## Limitations and caveats

### Performance impact

Enabling profiling introduces overhead, that stems mainly from two sources:

* timestamp and counters collection (mostly negligible in platform with hardware profiling support)
* buffer management, as profiling data must be transferred into the profiling output buffer

Profiling requires CMX allocations and additional DMA operations inserted in the schedule to collect the profiling data.
In some cases these DMA undergo regular scheduling and may not incur additional latency, but in unfavorable cases may
lead to extra spills or cause contention due to extra dependencies. In some cases like for DMA profiling the buffer
management DMA may be inserted on the critical path inducing fixed overhead.

### Timestamp collection and synchronization

Depending on the platform and specific execution engine the task timestamps may come from different source, it's either
a fixed frequency free running counter or local cycle counter being a subject to dynamic frequency scaling.

In the platforms where dynamic frequency scaling is affecting the DPU timestamps (NPU37xx or NPU40xx), the frequency
workpoint is captured at the beginning and at the end of the inference and used to calibrate the frequency, but may
result in inaccurate data if frequency changes during the inference.

Even if frequency is fixed or known in given platform the timestamps that come from different counters require
synchronization. An imperfect heuristic approach based on barriers which are common among the tasks is used to calibrate
the timers offset, therefore some bias may be introduced.

### DMA concurrency

The distribution of the DMA tasks across the threads in Perfetto report is arbitrary and is not aligned with hardware
channels.

Also the task profiling report may show more concurrent DMA operations than the number of hardware DMA channels, for the
following reasons:

* Consecutive DMA operations scheduled on the same channel may overlap when scheduled out-of-order (ORD bit clear). Use
  `dma-ooo=false` to disable this optimization.
* Due to HW limitation in some scenarios the DMA task end timestamp may be captured with a delay leading to an overlap
  with a subsequent task.

### DPU overlap

In some cases consecutive DPU tasks executed on the same DPU engine may slightly overlap. That is due to pipelining,
where input data unit picks subsequent task before output data unit completes prior task. This may result in number of
DPU threads appear greater than the number of DPUs in given cluster.

### Layer duration

Layers correspond to original model operations which are broken down into multiple hardware tasks by the compiler.
Because the tasks that belong to given layer may be executed by various engines in non-contiguous manner, the layer
duration reported may be significantly larger from the total compute time spent on the individual tasks, especially in
the presence of data prefetch. Per-engine compute time is included in the report.

### Platform specific constraints

#### NPU37xx

* DMA profiling injects timestamp DMAs, therefore incur overhead and is not precise especially in concurrent scenarios.
* DPU timestamps are derived from variable frequency clock and may be biased in presence of dynamic frequency scaling.
* Shave and DPU timestamps are derived from different hardware timers. Synchronization is heuristic (best-effort) and
  relies on DMA profiling data.

#### NPU40xx

* There is no support for DMA profiling in the driver stack. In full-stack a custom FW build with experimental FORCE_ENABLE_DMA_HWP config could be used in limited
  scenarios.
* M2I tasks are not profiled.
* DPU duration and start timestamp are derived from variable frequency clock and may be inaccurate in presence of
  dynamic frequency scaling.

#### NPU50xx

The NPU40xx limitations no longer apply. All timestamps are derived from a single fixed frequency timer.

### benchmark_app output

Reported total time and CPU total time may be significantly larger from the inference duration. That is because:
* CPU time is accumulated over the hardware tiles,
* multiple model layers may be executed in parallel in a non-contiguous manner (e.g. due to data prefetch).

Note: not-executed (optimized-out or fused) operations are not included in the output.

## Profiling development

### Location information
MLIR location is essential for profiling utility as it is used to identify a specific NPU task in the report and also to
map the tasks to OpenVINO model operation for reporting its execution time. The serialized tasks name will have the
following format:

`layer?t_type/subtask/...`

where *layer* is the friendly name of the original OpenVINO operation and *type* the operator name. Each NPU task should
have a unique and meaningful location so it can be tracked back to its origin in the compiler pipeline.

### Profiling metadata

Profiling metadata uses flat-buffers for serialization and follows [the profiling
schema](../src/vpux_utils/src/profiling/schema/profiling.fbs)

### Profiling SprLUT dummy DPU tasks

On NPU5+ a dummy DPU task is injected by the `insert-delay-dpu-variant` pass to ensure the correct initialization
of the invariant. Since the injection occurs after DPU profiling -- there's no profiling buffer allocated for the
profiling data of such task, but it still has to write its profiling data somewhere. The DPU profiling currently works
around it by assigning the dummy task the same workload_id as the next DPU task in the invariant, which makes the next
task overwrite the profiling data of the dummy task.

### Tools
`vpux-translate` offers `--vpux-profiling` command line option that enables profiling output during model import, e.g.:

`vpux-translate --vpu-arch=$ARCH --import-IE model.xml --vpux-profiling --mlir-print-debuginfo`

When using `vpux-opt` profiling passes can be enabled with a pipeline option `profiling=true`, e.g.:

`vpux-opt --vpu-arch=$ARCH --default-hw-mode="profiling=true" --mlir-print-debuginfo`

In either case `--mlir-print-debuginfo` is required to preserve location information.
