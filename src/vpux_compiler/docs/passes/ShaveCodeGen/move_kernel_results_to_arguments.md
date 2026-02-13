# MoveKernelResultsToArguments

## Brief

The pass enforces the calling convention for outputs of ShaveCodeGen generated kernels. We discuss alternatives and the rationale behind performing this transformation before bufferization.

## Problem statement

Let's assume the following function:

```MLIR
func.func @foo(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
  %0 = tensor.empty() : tensor<1x3x224x224xf16>
  %1 = linalg.generic {
     indexing_maps = [#NCHW, #NCHW, #NCHW],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
     ins(%arg0 : tensor<1x3x224x224xf16>) outs(%0 : tensor<1x3x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = math.log %in fastmath<afn> : f16
    linalg.yield %1 : f16
  } -> tensor<1x3x224x224xf16>
  return %1 : tensor<1x3x224x224xf16>
}
```

The calling convention for the function is that the result (output) buffers are allocated and owned by the caller, and that result (output) buffers are passed to the callee by appending the correspondent memrefs to the end of its argument list.

At some point after bufferization our codegen is expected to transform this IR into:

```MLIR
func.func @foo(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) {
  linalg.generic {
     indexing_maps = [#NCHW, #NCHW, #NCHW],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
     ins(%arg0 : memref<1x3x224x224xf16>) outs(%arg1 : memref<1x3x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %1 = math.log %in fastmath<afn> : f16
    linalg.yield %1 : f16
  }
  return
}
```

The tensor.empty() or equivalent memref.alloc should be removed in the process as its purpose was to model the allocation/creation of the returned tensor. We now have an additional argument for the caller-allocated return buffer (%arg1).

Note that this is a very simple example and we expect this to generalize to random IR containing a mix of linalg/tensor/scf ops.

## Interactions with SHAVE specific optimizations

ShaveCodeGen uses various optimizations to remove intermediate tensor values and improve execution speed. After these transformations are performed the IR is substantially more complex, with added SCF and tensor ops (including reshapes). Leftover tensor.empty() ops after this optimizations become memref.alloc, which are not supported and need to be handled in some other way (otherwise these get lowered to calls to malloc which is not supported in SHAVE kernels).

These optimizations impact the transformation required for the calling convention enforcement. If we're performing it post-optimizations then we need to handle the additional complexities of the IR (e.g. reshape ops).

## Approaches

There are two competing approaches. First, the post-bufferization one, already implemented in VPUX which leverages copy optimizations to remove the memref.alloc. Second the currently proposed pre-bufferization one uses bufferization ops and empty tensor elimination to enforce the semantics of the calling convention pre-bufferization and remove the tensor.empty() op.

### Post-bufferization (add-buffers-for-net-results)

We enforce the calling convention post-bufferization by adding the additional memref arguments. On return copies are added from the currently returned memrefs to the previously added arguments. We rely on copy elimination optimizations for removal of the callee-side memref.alloc ops for the the returned values. This is similar to add-buffers-for-net-results.

Due to the SHAVE-specific optimizations, the IR for the simple example looks like the following after performing this transformation:

```MLIR
#C = affine_map<(d0) -> (d0)>
func.func @foo(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) {
  %c1 = arith.constant 1 : index
  %c150528 = arith.constant 150528 : index
  %c0 = arith.constant 0 : index
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<1x3x224x224xf16> into memref<150528xf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<150528xf16>
  scf.for %arg2 = %c0 to %c150528 step %c1 {
    %subview = memref.subview %collapse_shape[%arg2] [1] [1] : memref<150528xf16> to memref<1xf16, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[%arg2] [1] [1] : memref<150528xf16> to memref<1xf16, strided<[1], offset: ?>>
    linalg.generic { indexing_maps = [#C, #C], iterator_types = ["parallel"]}
        ins(%subview : memref<1xf16, strided<[1], offset: ?>>)
        outs(%subview_0 : memref<1xf16, strided<[1], offset: ?>>) {
    ^bb0(%in: f16, %out: f16):
      %0 = math.log %in fastmath<afn> : f16
      linalg.yield %0 : f16
    }
    %subview_1 = memref.subview %alloc[%arg2] [1] [1] : memref<150528xf16> to memref<1xf16, strided<[1], offset: ?>>
    memref.copy %subview_0, %subview_1 : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
  }
  %expand_shape = memref.expand_shape %alloc [[0, 1, 2, 3]] output_shape [1, 3, 224, 224] : memref<150528xf16> into memref<1x3x224x224xf16>
  memref.copy %expand_shape, %arg1 : memref<1x3x224x224xf16> to memref<1x3x224x224xf16>
  return
}
```

Because the original tensor.empty() has folded with tensor.collapse_shape in the optimization process we now have a memref.alloc of memref<150528xf16> which doesn't match the original output shape.

The copy elimination algorithm would need to be able to look through memref.expand_shape to find %alloc, emit a memref.collapse_shape for %arg1 from memref<1x3x224x224xf16> to memref<150528xf16>, then we can finally replace %alloc. More generally, a robust copy optimization needs to:

- look through memref.reshape, memref.view, memref.collapse_shape, memref.expand_shape, etc
- emit a sequence of ops to reshape the output argument memref to the allocation memref

It might be tempting to simplify the reshapes by changing the type of the returned memref. This looks suspicious and likely won't work as this can break the calling convention between the caller and callee.

Overall this looks possible, however it would be complex to get to something that universally works.

Note: memref.copy was used for the copy. A VPUIP.Copy would not be desirable since we don't have lowering to LLVM for it.

### Pre-bufferization approach

Inspired by the [LLVM Dev Meeting 2023 tutorial slides](https://m-sp.org/downloads/llvm_dev_2023.pdf) (see slide 34-39) and the [bufferization documentation](https://mlir.llvm.org/docs/Bufferization/#tensor--buffer-boundary) this attempts to perform the transformation using available upstream infrastructure, in the hope of producing a generic and maintainable solution. This is the approach adopted by the MoveKernelResultsToArguments pass.

We run this transformation before the ShaveCodeGen optimization phase (pre-bufferization), as we would like to avoid any reshape ops introduced there. It's also less computationally intensive to run this on simpler IR.

The transformation adds the result buffers as memrefs to the function signature and enforces that the result is written to the added result buffers with bufferization::MaterializeInDestinationOp ops.

We remove the possibility of getting memref.allocs after bufferization for the result tensor by running empty tensor elimination. Note that this requires that the added bufferization.materialize_in_destination ops to have restrict/writable set for the result buffers.

After this transformation:

```MLIR
func.func @foo(%arg0: tensor<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) {
  %0 = tensor.empty() : tensor<1x3x224x224xf16>
  %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
           ins(%arg0 : tensor<1x3x224x224xf16>)
           outs(%0 : tensor<1x3x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %2 = math.log %in fastmath<afn> : f16
    linalg.yield %2 : f16
  } -> tensor<1x3x224x224xf16>
  bufferization.materialize_in_destination %1 in restrict writable %arg1 : (tensor<1x3x224x224xf16>, memref<1x3x224x224xf16>) -> ()
  return
}
```

Running empty tensor elimination removes the tensor.empty() that would correspond to the returned buffer, replacing it with a bufferization.to_tensor of our added result memref:

```MLIR
func.func @foo(%arg0: tensor<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) {
  %0 = bufferization.to_tensor %arg1 restrict writable : memref<1x3x224x224xf16> to tensor<1x3x224x224xf16>
  %1 = linalg.generic {indexing_maps = [#NCHW, #NCHW], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
           ins(%arg0 : tensor<1x3x224x224xf16>)
           outs(%0 : tensor<1x3x224x224xf16>) {
  ^bb0(%in: f16, %out: f16):
    %3 = math.log %in fastmath<afn> : f16
    linalg.yield %3 : f16
  } -> tensor<1x3x224x224xf16>
  bufferization.materialize_in_destination %1 in writable %arg1 : (tensor<1x3x224x224xf16>, memref<1x3x224x224xf16>) -> ()
  return
}
```

This form has the desired semantics already available in the IR. Since the semantics are already baked in, after the various ShaveCodeGen transformations, one-shot bufferization will produce the IR in the desired form without the need of other optimizations:

```MLIR
func.func @foo(%arg0: memref<1x3x224x224xf16>, %arg1: memref<1x3x224x224xf16>) {
  %c1 = arith.constant 1 : index
  %c150528 = arith.constant 150528 : index
  %c0 = arith.constant 0 : index
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<1x3x224x224xf16> into memref<150528xf16>
  %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1, 2, 3]] : memref<1x3x224x224xf16> into memref<150528xf16>
  scf.for %arg2 = %c0 to %c150528 step %c1 {
    %subview = memref.subview %collapse_shape[%arg2] [1] [1] : memref<150528xf16> to memref<1xf16, strided<[1], offset: ?>>
    %subview_1 = memref.subview %collapse_shape_0[%arg2] [1] [1] : memref<150528xf16> to memref<1xf16, strided<[1], offset: ?>>
    linalg.generic {indexing_maps = [#C, #C], iterator_types = ["parallel"]}
        ins(%subview : memref<1xf16, strided<[1], offset: ?>>)
        outs(%subview_1 : memref<1xf16, strided<[1], offset: ?>>) {
    ^bb0(%in: f16, %out: f16):
      %0 = math.log %in fastmath<afn> : f16
      linalg.yield %0 : f16
    }
    %subview_2 = memref.subview %collapse_shape_0[%arg2] [1] [1] : memref<150528xf16> to memref<1xf16, strided<[1], offset: ?>>
    memref.copy %subview_1, %subview_2 : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16, strided<[1], offset: ?>>
  }
  return
}
```

## Why do this pre-bufferization?

Operations used by ShaveCodeGen are either DestinationPassingStyle or views/subviews which don't copy memory after bufferization. Because of this we can shape the IR pre-bufferization to something that's already optimal for bufferization and avoids copies. This avoids the need to have a copy optimization pass. If there is an issue with generated copies in this case the fix would be adjusting the IR pre-bufferization rather than supporting copy optimization. This also seems to be reflected upstream as there doesn't seem to be any copy optimization support that we could use out of the box.

This is fundamentally different from the rest of VPUX, where DestinationPassingStyle is not used and there is a reliance on copy optimizations/FeasibleAllocation for optimal buffer allocation. For this case the post-bufferization approach seems natural, as there is infrastructure available for it, but no infrastructure exists for the pre-bufferization approach.

Note that the VPUIP copy optimization pass does not work for the case described above and would likely need substantial work to get the required support for the generic case.

Both approaches are implementable, however the post-bufferization approach requires custom logic that likely isn't useful outside of this usecase. This feels forced and against the spirit of one-shot bufferization producing optimal allocations.

## Conclusion

The post-bufferization approach works with the VPUIP case because there is already some working post-bufferization copy elimination pass/pipeline operating on the existing set of ops.

There's no obvious, urgent need for post-bufferization copy optimizations for ShaveCodeGen, at least at the moment, since we expect the DPS ops to give us something that is as close as possible to optimal allocation. This makes the pre-bufferization approach preferable for ShaveCodeGen.
