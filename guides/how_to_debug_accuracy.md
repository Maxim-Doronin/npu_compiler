# How to debug accuracy issues

There are scenarios where changes or new features in the compiler may result in accuracy issues. Debugging and finding root cause of accuracy issues can be a difficult task, below are some guides on how to approach such problem.

## Generic Accuracy Debug

Generic approach is to cut the network before compilation and try to determine the place where accuracy is broken and investigate the region for any issues which potentially could cause accuracy issues. Model can be cut using op model cutter or it can be cut manually by editing the .xml file. 

However, some issues may not be reproducible after modification to the network - this can be debugged with the tool below.

## Accuracy issues due to Scheduling (memory | barrier)

When changes in scheduling (memory addresses, barriers) fix the accuracy issue. E.g. issue is not reproduced with a cut model. Debugging with larger models can get very complex.

Intermediate buffer output tool was introduced to reduce complexity. It can dump intermediate buffer to output after all scheduling was performed and dump specified operation buffer as output at a specified location.

### Intermediate Buffer Output

Requires DEVELOPER build and is suggested to use with:
* `export IE_NPU_LOG_FILTER="intermediate-buffer-output"` - enable logs from the buffer dumping pass.
* `export IE_NPU_IR_PRINTING_FILTER="IntermediateBufferOutput"` - enable IR dump after buffer dumping pass.

Requires compiler flag to be enabled:
* `enable-intermediate-buffer-output=true` - flag to enable the pass in the compiler.

Required 3 environmental variables:
* `export IE_NPU_DEBUG_OP_INDEX=10` - index of operation of which buffer is to be dumped.
* `export IE_NPU_DEBUG_BUFFER_INDEX=1` - index of buffer from selected operation which is to be dumped.
* `export IE_NPU_DEBUG_INSERTION_INDEX=10` - index of insertion for the DMA which dumps the target buffer.

#### How to use ?

1. Compile the target model with `enable-intermediate-buffer-output=true` compilation option and the `IE_NPU_IR_PRINTING_FILTER="IntermediateBufferOutput"` environment variable. This will print the IR at the pass which can introduce extra DMAs for dumping intermediate results.
2. From the IR, extract the index of the operation that should be dumped and re-compile the model while setting the `IE_NPU_DEBUG_OP_INDEX` and `IE_NPU_DEBUG_INSERTION_INDEX` to the chosen index. `IE_NPU_LOG_FILTER="intermediate-buffer-output"` should also be set, so that the logs of the pass will be displayed. In the printed logs, each buffer of the operation will have an index.
3. Finally, to dump a specific buffer, also set `IE_NPU_DEBUG_BUFFER_INDEX` to one of the indexes seen in the logs. The position of the DMA to be inserted can also be controlled via `IE_NPU_DEBUG_INSERTION_INDEX`.

#### Example usage:

1. Compile model with `enable-intermediate-buffer-output=true` and use`export IE_NPU_IR_PRINTING_FILTER="IntermediateBufferOutput"` for OP_INDEX INSERTION_INDEX selection
2. From IR dump select index of operation to cut
```
  VPURT.Task waits(%27 : !VPURT.Barrier) updates(%28 : !VPURT.Barrier) attributes {opIndex = 164 : i64} {
    %537 = VPUIP.NNDMA inputs(%288) outputs(%521)
  }
  VPURT.Task waits(%27 : !VPURT.Barrier) updates(%28 : !VPURT.Barrier) attributes {opIndex = 165 : i64} {
    %537 = VPUIP.NNDMA inputs(%287) outputs(%522)
  }
  VPURT.Task waits(%28 : !VPURT.Barrier) updates(%29 : !VPURT.Barrier) attributes {opIndex = 166 : i64} {
    %537 = VPUIP.NNDMA inputs(%294) outputs(%299)
  }
  VPURT.Task waits(%29 : !VPURT.Barrier) updates(%31 : !VPURT.Barrier) attributes {opIndex = 167 : i64} {
    %537 = VPUIP.NCEClusterTask input(%295) weights(%300) weight_table(%308) parent_input(%295) parent_output(%317)
  }

```
3. Set `export IE_NPU_DEBUG_OP_INDEX=167` and `export IE_NPU_DEBUG_INSERTION_INDEX=167` then use `export IE_NPU_LOG_FILTER="intermediate-buffer-output"` for BUFFER_INDEX selection
```
[TRACE] 09:25:43.652 [intermediate-buffer-output]     targetTaskOp %537 = VPUIP.NCEClusterTask input(%295) weights(%300) weight_table(%308) parent_input(%295) parent_output(%317)

Operation buffers with indices:
[TRACE] 09:25:43.659 [intermediate-buffer-output]       Index=0, buffer %295
[TRACE] 09:25:43.666 [intermediate-buffer-output]       Index=1, buffer %300
[TRACE] 09:25:43.673 [intermediate-buffer-output]       Index=2, buffer %308
[TRACE] 09:25:43.680 [intermediate-buffer-output]       Index=4, buffer %317

```
4. Select buffer index, `export IE_NPU_DEBUG_BUFFER_INDEX=4`. Now we can see a DMA after `167` and result types updated.
```
  VPURT.Task waits(%31 : !VPURT.Barrier) {
    %559 = VPUIP.NNDMA inputs(%319) outputs(%318)
  }
  return %arg1
```
5. If DMA is to be inserted in a different place use `export IE_NPU_DEBUG_INSERTION_INDEX=154`. New DMA out is now after `154`
```
  VPURT.Task waits(%26 : !VPURT.Barrier) updates(%27 : !VPURT.Barrier) attributes {opIndex = 154 : i64} {
    %555 = VPUIP.NNDMA inputs(%274) outputs(%282 )
  }
  VPURT.Task waits(%27 : !VPURT.Barrier) {
    %555 = VPUIP.NNDMA inputs(%315) outputs(%314)
  }
  return %arg1
```

This intermediate output can be used to find the operation / place which causes accuracy issue. 

##### How to find the target corresponding operation in a modified schedule ?

Currently indices of operations need to be found - which can be difficult in certain scenarios. Aiming to move to a unique operation name or identifier when `E#92445` will be implemented.

Tips for now: use the same index but ensure that the `targetTaskOp` in logs is the same, `numOpsToPrint=10` is also used to log a defined number of previous operations which could used to find the corresponding same operation with different scheduling to compare accuracy.
