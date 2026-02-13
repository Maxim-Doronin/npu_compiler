# [Compiler Schedule Trace](#compiler-schedule-trace)

Schedule trace functionality is handled by `inference-execution-analysis` pass at end of compilation which performs simulation of inference execution with knowledge of each task cycle cost and understanding of different types of HW engines (e.g. DPU, DMA, ActShave, ...) and their instances (e.g. cluster 0, cluster 1). As a result of this simulation for each task start and end cycle is determined.
By knowing the system frequency cycles can be converted to time units and total inference latency can be predicted

Operation cost is taken from VPUNN cost model so how close provided simulation is to real execution is heavily dependant on accurate cost model.

**IMPORTANT:**

There are still issues with cost provided from VPUNN. Until they are resolved determined cycles might not reflect real HW. Some known limitations:
* Not all ActShave layers are supported by VPUNN (E#89808)
* No accurate predictions for DMAs with multiple strides (e.g. Permute DMA) (E#89933)
* Not all generations are supported by VPUNN
* Estimated cost can be far from real one. This can happen for all types of engines (example: E#90279)
* No modelling for out-of-order DMA execution
* Model layers are not included in the JSON report

If for some layers returned cost was invalid (error code from VPUNN or layer not supported by VPUNN) it will be assigned cost = 1. Information about number of layers with no valid cost and their type is provided in the log, example:

```
[WARNING] [inference-execution-analysis]    Invalid cost for:
[WARNING] [inference-execution-analysis]      VPUIP.SW.Kernel.builtin_Convert
```

## Enabling

`inference-execution-analysis` pass is disabled by default. There is a dedicated compiler option that needs to be turned on:

```
NPU_COMPILATION_MODE_PARAMS enable-schedule-trace=true
```
When model is compiled trace file `compileTimeScheduleTrace.json` is generated. Structure of this file allows it to be easily compared with result from HW profiling.

*NOTE: Compiler schedule trace is not intended to be enabled with HW profiling (PERF_COUNT YES) simultaneously.*

Name of the generated trace file can be changed via:

```
NPU_COMPILATION_MODE_PARAMS schedule-trace-file-name=compileTimeScheduleTrace.json
```

## Latency estimation and tasks statistics
### Latency estimation
Besides generating trace file **inference latency estimation** is provided in the log (`LOG_INFO`), example:

```
[INFO] 16:30:10.463 [inference-execution-analysis]    Estimated inference latency: 942.78us
```

### Tasks statistics
The compiler schedule trace (when enabled) invokes calculation of basic tasks statistics that are logged and added to compileTimeScheduleTrace.json file.

Example of task statistics logs:
```
[INFO] 16:30:10.465 [inference-execution-analysis]    Tasks statistics:
[INFO] 16:30:10.465 [inference-execution-analysis]      - total duration [ns]: 942784
[INFO] 16:30:10.465 [inference-execution-analysis]      - DMA duration [ns]: 517849 (54.93 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - DPU duration [ns]: 378580 (40.16 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - SW duration [ns]: 241148 (25.58 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - DMA-DPU overlap [ns]: 188631 (20.01 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - DMA-SW overlap [ns]: 6231 (0.66 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - SW-DPU overlap [ns]: 0 (0.00 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - all tasks union [ns]: 942715 (99.99 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - total idle [ns]: 69 (0.01 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - SW duration without DPU overlap [ns]: 241148 (25.58 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - DMA duration without overlaps [ns]: 322987 (34.26 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - Sum of DMA task durations [ns]: 887417 (94.13 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - Sum of DPU task durations [ns]: 752563 (79.82 %)
[INFO] 16:30:10.465 [inference-execution-analysis]      - Sum of SW task durations [ns]: 250715 (26.59 %)

```

The individual entries are defined as follows:

* *total duration* - wall time duration of inference. This should match the *Estimated inference latency*.
  If the two don't match it should indicate that some tasks may have not been selected for statistics calculation.
* *DMA/DPU/SW duration* - sum of interval durations of a union of DMA/DPU/SW tasks. The naming convention for these parameters follows the one assumed in VPU-EM simulations
* *DMA-DPU overlap* - sum of interval durations of intersection of a union of DMA and a union of DPU tasks. Hence, this parameter measures 
the overlap between all DMA and all DPU tasks projected on time axis.
* *DMA-SW overlap* - sum of interval durations of intersection of a union of DMA and a union of SW tasks
* *SW-DPU overlap* - sum of interval durations of intersection of a union of SW and a union of DPU tasks
* *all tasks union* - provided for self-consistency: should match *total duration* - *idle duration*
* *total idle* - sum all no-operation intervals during which neither of DMA, DPU or SW tasks are executed
* *SW duration without DPU overlap* - total time of SW tasks union that does not intersect with a union of DPU tasks. This parameter
  is also measured in VPU-EM simulations.
* *DMA duration without overlaps* - total time of DMA tasks union that does not intersect with a union of all other tasks (DPU and SW). This parameter is also measured in VPU-EM simulations but it's value may differ because it is not biased by possible 
non-vanishing "total idle" and non-vanishing *SW-DPU overlap*.
* *Sum of DMA/DPU/SW task durations* - sum of all DMA/DPU/SW tasks durations (in all clusters). For DPU tasks we calculate this statistic 
using invariants and otherwise we use all tasks of given type.

The values provided in parentheses are percentages of *total duration*.

## Schedule trace file

The generated JSON trace file uses Google Trace Event Format that can be opened in [perfetto](https://ui.perfetto.dev/). 
The file contains two main sections *traceEvents* and *taskStatistics*.
The same file format is used for [HW profiling](./how_to_use_profiling.md#how-to-use-neural-network-profiling).

### traceEvents

In *traceEvents* section each task contains the following keys:
* *name* - name of operation matching its `Loc` at the end of `VPURT` dialect, amended by additional post-processing suffixes
* *cat* - a task category (DPU, SW, DMA)
* *ts* - task start time [us]
* *dur* - task duration [us]
* *pid* and *tid* - assignment to process and thread that correspond to HW hierarchical layout of clusters and multiple engines

Example trace view is structured in following way:
```
DMA:
|-- DMA 0
|-- DMA 1
Cluster 0:
|-- DPU 0
|-- SW / Shave 1
|-- SW / Shave 2
Cluster 1:
|-- DPU 0
|-- SW / Shave 1
|-- SW / Shave 2
```

This way all independent engines like DMA ports, NCE clusters and multiple ActShave engines in single clusters are shown as separate queues. Similarly structured trace is generated by HW profiling feature.

### taskStatistics

The *taskStatistics* section contains the tasks statistics in us. 
Individual fields correspond to those from the compiler logs (see [Tasks Statistics](#tasks-statistics))
