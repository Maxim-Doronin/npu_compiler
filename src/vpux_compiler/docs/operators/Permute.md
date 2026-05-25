## NCE Permute Operation

 `NCE.Permute` operation is an operation executed on DPU with the main goal of efficiently permuting the given input from one layout to another. Currently, the operation supports only `NCHW` to `NHWC` transition, but it could be further extended in the future.

### Background info

The main issue here is that `DPU` only processes input data with `NHWC` layout. But there are networks that have inputs stored in `NCHW` layout and must be converted to `NHWC` layout before running an inference.

In order to do this permute operation on HW, the DPU can be tricked by making it interpret `NCHW` data as `NHWC`.

The ideal permute will be the following :
    NCHW -> NHWC

In order to do this permutation on HW op we trick the HW that we have the following permutation, but memory still looks the same:

- Permutation we trick HW to do : `NHWC -> NWCH`
- How `memory side` looks like : `NCHW -> NHWC`
- Eg:
    - Original configuration:   input: `1x3x25x24, NCHW` -> output: `1x4x25x24, NHWC`
        - in memory:    `1x3x25x24` -> `1x25x24x4`
    - `NCE.Permute` configuration: input: `1x24x3x25, NHWC` -> output: `1x24x4x25, NWCH`
        - in memory:    `1x3x25x24` -> `1x25x24x4`

The output has 4 channels instead of 3 because of channel expansion — one of NCE.Permute's capabilities. The input has 3 channels, but the DPU hardware requires channel dimensions to be aligned to specific values (typically multiples of 4 or 16).

Why are we able to do this?

We use a HW Eltwise operation which operates on the whole tensor and does not care how spatial or depth info is organized in memory. It will move data as specified, the permutation itself being done by the ODU. Technically, this could be applied by any NCE op, as the ODU is common between them. For example, Pooling operations could be used also, but we use Eltwise because it shortcuts the MPE, which we don't need to do the Permute and thus it's more efficient.

For awareness, `NCHW` compiler layout can also be called `Planar format (XYZ)`, while `NHWC` compiler layout can be found as `ZMajor (ZM) format (ZXY)`.

NCHW layout memory representation:

![NCHW layout data representation](../assets/NCHW-layout-data-representation.png)

NHWC layout memory representation:

![NHWC layout data representation](../assets/NHWC-layout-data-representation.png)

### NCE Permute flow inside compiler

#### IE dialect implementation

At `IE` dialect level, the first time where permutation from `NCHW` to `NHWC` layout can be found after `AdjustLayouts` pass. This pass inserts multiple `IE.Reorder` operations to wrap each op in order to satisfy layout requirements. Unnecessary Reorder ops are optimized with `Canonicalizer` pass which follows `AdjustLayouts` pass, and the only ones left are `IE.Reorder` ops at the beginning and the end of the IR. `IE.Reorder` op IR example below:

```mlir
%0 = IE.Reorder(%input) {dstOrder = #NHWC} : tensor<1x3x224x224xf16> -> tensor<1x3x224x224xf16, {order = #NHWC}>
```

Next step in `IE` dialect is to convert `IE.Reorder` op to `IE.PermuteQuantize` op. This conversion takes place inside `ConvertReorderToPermuteQuantize` pass that converts to `IE.PermuteQuantize` only `IE.Reorder` operation with `input order NCHW` and `output order NHWC`. `IE.PermuteQuantize` have the following representation:

```mlir
%0 = IE.PermuteQuantize(%input) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>
```

Only some isolated cases are lowered to `VPU.NCEPermute` operation, which will execute the permutation on DPU. The other IE.PermuteQuantize ops end up being executed on SHAVE. The isolated cases are presented in the chapter below.

#### Lowering to VPU NCE Permute

IEPermuteQuantize operation is lowered to `NCEPermute` operation which has the main characteristics below, extracted from op definition:

```
def VPU_NCEPermuteOp :
        ...{
    let summary = "More abstract version of combined NCE Permute and Quantization layers";

    let description = [{
        Used to perform a datatype conversion, relayout of data and shape expansion,
        all using a single NCE HW op.

        * expandedChannels - target size of output channels after expansion, usual values are 4 and 16
        * dstElemType - output tensor datatype
        * dstOrder - output tensor layout, NCHW input to NHWC output relayout is supported
    }];

    let arguments = (ins
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$input,

        IntAttr:$expandedChannels,
        TypeAttr:$dstElemType,
        AffineMapAttr:$dstOrder,
        VPU_PPEAttrInterface:$ppe,

        OptionalAttr<VPU_MultiClusterStrategyAttr>:$multiClusterStrategy
    );

    let results = (outs
        AnyTypeOf<[4DTensorOf<[F16, BF16, quant_QuantizedType]>, VPU_SparseTensor, VPU_DistributedTensor]>:$output
    );
    ...
}
```

From the op definition slice listed above we can extract the following key aspects:

- the supported layout permute done by `NCEPermute` op on HW is the following : `NCHW -> NHWC`
- operation can expand channels
- `NCEPermute` can also change tensor datatype. Supported input data type : FP16 ->  output data type : FP16, FP32, UniformQuantizedType
- full op definition can be found in ops.td file corresponding to VPU dialect.

Lowering to VPU dialect is done as shown below:

- IR before lowering:

```mlir
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @ConvertPermuteQuantize(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}
```

- IR after lowering:

```mlir
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

func.func @ConvertPermuteQuantize(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}
```

As it can be seen above, `dstElemType` and `dstOrder` remain unchanged from previous dialect's `IE.PermuteQuantizeOp`. The only supported type of padding for HW Permute op is on channels and is converted into `expandedChannels` attribute that stores integer value for the output channel size. The above examples have expandedChannels to 4 because the following op would be NCECompressConv operation.

#### VPU dialect implementation

`VPU.NCEPermute` operation is an abstract representation of the permutation that is done on HW, retaining its original `NCHW` input and `NHWC` output. It's cleaner to handle the op with the original configuration while in the higher-level dialect, and operation will be lowered into the actual hardware config at a later stage.

Regarding strategy assignment, `NCE.Permute` operation has the following supported strategies: `SplitOverHeightOverlapped`, `SplitOverHeight` and `SplitOverKernel`. For a better understanding, below are some examples:

- First example is for `SplitOverHeightOverlapped` strategy NPU37XX.
    - this strategy is assigned only for NPU37XX platform when the next operation is a CompressConv. CompressConv operation has the following requirement: input must be `OVERLAPPED` because it cannot read data from other clusters and this is why we assign `SplitOverHeightOverlapped` strategy in this case.
    - per cluster distribution between Permute distributed out and CompressConv distributed in is the same, so there is no need for a spill between them. Also equal_memory_and_compute_view attribute is set which means that the compute shapes/offsets & memory shapes/offsets are the same; without this, the output workload computation for permute will not be correct.
    - `NCE.Permute` input distributed type:
        ```mlir
            !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
        ```
    - `NCE.Permute` output distributed type :
        ```mlir
            !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>
        ```
    - NCECompressConv input distributed type :
        ```mlir
            !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, strides = [2, 2], num_clusters = 2 : i64}>
        ```
- Second example is for `SplitOverHeight` strategy NPU40XX.
    - NCEPermute input distributed type - no line overlap between clusters:
        ```mlir
            !VPU.DistributedTensor<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]], compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]], memory_shapes = [[1, 3, 112, 224], [1, 3, 112, 224]], memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
        ```
    - NCEPermute output distributed type - with overlap lines, to satisfy the next consumer requirements:
        ```mlir
            !VPU.DistributedTensor<1x4x224x224x!qElemType, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 4, 112, 224], [1, 4, 112, 224]], compute_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]], memory_shapes = [[1, 4, 113, 224], [1, 4, 112, 224]], memory_offsets = [[0, 0, 0, 0], [0, 0, 112, 0]]}>
        ```
    - Next NCE operation input distributed type, in this case was NCECompressConv but it could be any other DPU op and NCE.Permute will treat it the same:
        ```mlir
            !VPU.DistributedTensor<1x16x112x112xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 16, 56, 112], [1, 16, 56, 112]], compute_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]], memory_shapes = [[1, 16, 56, 112], [1, 16, 56, 112]], memory_offsets = [[0, 0, 0, 0], [0, 0, 56, 0]]}>
        ```
- Third example is for `SplitOverKernel`
    - this strategy is assigned in the cases where NCEPermute operation is followed by another operation with SOK strategy and NCEPermute output channels are large enough to be splitted. This is done in order to avoid spills between this ops.
    - NCEPermute output distributed type:
        ```mlir
            !VPU.DistributedTensor<1x128x32x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]], compute_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]], memory_shapes = [[1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64], [1, 32, 32, 64]], memory_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]}>
        ```

#### Lowering to VPUIP NCE Cluster task

When lowering to VPUIP dialect we move from an abstract representation to the actual hardware configuration as shown in the mlir test below:

- IR before lowering:

```mlir
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

    !qElemType = !quant.uniform<u8:f16, 1.000000e+00>

    func.func @NcePermute(%arg0: tensor<1x3x225x224xf16, {mem_space = @CMX_NN}>)
            -> tensor<1x4x225x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {

        %0 = VPU.NCE.Permute(%arg0) {
            dstElemType = !qElemType,
            dstOrder = #NHWC,
            expandedChannels = 4 : i64
        } -> tensor<1x4x225x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 225, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
        }

        return %0 : tensor<1x4x225x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>
    }
```

- IR after lowering:

```mlir
    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
    #NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

    !qElemType = !quant.uniform<u8:f16, 1.000000e+00>

    func.func @NcePermute(%arg0: memref<1x3x225x224xf16, @CMX_NN>)
            -> memref<1x4x225x224x!qElemType, #NHWC, @CMX_NN> {

        %0 = VPUIP.ViewOp %arg0 : memref<1x3x225x224xf16, @CMX_NN> to memref<1x224x3x225xf16, #NHWC, @CMX_NN>
        %alloc = memref.alloc() : memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN>

        %1 = VPUIP.NCEClusterTask <{
            is_permute_quantize,
            is_superdense,
            task_type = #VPUIP.nce_task_type<ELTWISE>
        }> input(%0 : memref<1x224x3x225xf16, #NHWC, @CMX_NN>)
          weights(%0 : memref<1x224x3x225xf16, #NHWC, @CMX_NN>)
          parent_input(%0 : memref<1x224x3x225xf16, #NHWC, @CMX_NN>)
          parent_output(%alloc : memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN>)
          outputs(%alloc : memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN>)
        -> memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN> variants : {
            DPUTask {mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
            outEnd = [224, 2, 223], outStart = [0, 0, 0],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask <ADD> {clamp_high = 255 : i64, clamp_low = 0 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01]}
        }

        %2 = VPUIP.ViewOp %1 : memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN> to memref<1x4x225x224x!qElemType, #NHWC, @CMX_NN>

        return %2 : memref<1x4x225x224x!qElemType, #NHWC, @CMX_NN>
  }
```

The lowering stage for `NCEPermute` op is done as following:

- There is a `ViewOp` introduced for the input tensor that `changes shape view and layout view of the memref`. Taking the above example, `input shape view` is changed from `1x3x225x224` to `1x224x3x225`, and `input layout` view from `NCHW` to `NHWC`. The reason why View op is introduced, is to represent the same memory placement, but with another layout. So 1x3x225x224 NCHW and 1x224x3x225 NHWC have their data placed in CMX in exactly the same way, but we need the DPU op to see the second representation, while the parent op must see the NCHW representation.
- `VPU.NCE.Permute` operation is lowered to `VPUIP.NCEClusterTask` operation with `ELTWISE` nce task type and does also memory permutation as shown in the example : `input(memref<1x224x3x225xf16, #NHWC, @CMX_NN>)` ->  `output(memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN>)`.
- Last operation inserted for this subgraph is another `ViewOp` that also `changes shape view and layout view` in the following way: from the above example, `output shape view` is changed from `1x224x4x225` to `1x4x225x224`, and `output layout view` from `NWCH` to `NHWC`. The reason why View op is introduced, is to represent the same memory placement, but with another layout. So NCEPermute must see this buffer configuration at output `memref<1x224x4x225x!qElemType, #NWCH, @CMX_NN>`, but the consumer op must see `memref<1x4x225x224x!qElemType, #NHWC, @CMX_NN>`.
