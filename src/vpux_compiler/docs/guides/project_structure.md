# Project structure

This document is written on the basis of discussions taken as part of the task of preparing the compiler for development in open-source. This means that (debatable) decisions on the structure of the project structure were made based on the following conditions:
- Enable/disable a specific platform by a CMake option and a corresponding define;
- Make a clear, convenient process of adding a new device;
- Ensure code reuse for different device generations.

## Compiler overview

### Dialects

<p align="center">
  <img src="images/compilation_flow.png" width="50%"><br>
  <em>NPU compilation pipeline</em>
</p>

Regardless of the device version, the compilation flow has the same appearance at the dialect level. These dialects represent different levels of detail. The IR is lowered from high level abstractions to more detailed representation step-by-step during compilation. The compilation pipeline consists of the "atomic“ passes. Each pass in compilation pipeline must represent one single transformation to reach one specific goal (either IR adaptation or IR optimization). More information is available from the [Compiler HLD](../npu_compiler_hld.md#generic-architecture).

It is also necessary to mention that only a low-level dialect might depend on a high-level one. Otherwise, it may lead to circular dependencies that break the build, make the system harder to maintain, limit testability, and cause other negative effects.

### Libraries

It makes sense to divide libraries into two "types": common and platform-specific. The diagram below illustrates connections between the libraries:

<p align="center">
  <img src="images/libraries.png" width="80%"><br>
  <em>Library dependencies</em>
</p>

Please note that this diagram does not show all libraries and dependencies in the compiler. The common part(dark rectangles) contains only open (non-embargoed) code reused across multiple platforms. It consists of:
- dialects: `const`, `core`, `net` and `config` which are completely common and widely used in the compiler;
- `core` – data structures required by compiler;
- `utils` – helpers to work with core data structures;
- `profiling_utils` – contains several model profiling infrastructure components within `::vpux::profiling` namespace;
- etc.

The common part can also be split into components where the core piece is the dialect library. This scheme is used for the `IE`, `VPU`, `VPUIP`, `VPURT`, and `ELF` dialects:
- `[dialectName]_IR` – dialect operations, attributes and types;
- `[dialectName]_transforms` – passes to perform transformations over IR;
- `[dialectName]_interfaces` – interfaces and base classes on which passes may depend;
- `[dialectName]_utils` – helpers to work with dialect types.

The platform-specific part includes both open and closed code: constants (e.g., number of tiles, CMX memory size), passes, pipelines, interface implementations, and other device-specific details/utilities. There is one library for each IP generation.

## Passes

### Common passes

These are fully platform-agnostic passes. This means you will get the same result for any input IR regardless of the platform version. Such passes have to be placed in a common part. Please refer to [primer_mlir](primer_mlir.md#passes) to get more information.

### Platform-specific passes

Hardware specific passes are designed to work on a particular platform. From the development perspective, the only difference is that it is necessary to use the appropriate folder.

You are allowed to reuse passes from an older IP generation for a newer one if the required feature is a strict superset:

```C++
// 50XX
void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions50XX& options, Logger log) {
    // ...
    // Use pass from previous version here
    pm.addPass(IE::arch40xx::createHwSpecific1Pass(log));
    IE::buildName1Pipeline(pm, log);
    // ...
}
```

Platform-specific passes must also be registered in [vpux-opt](../../../../tools/vpux-opt/vpux-opt.cpp) for validation purposes. In order to do this private platform should implement `IPassesRegistry` interface:

Then `registerPasses` method of the appropriate implementation will be called:

```C++
const auto passesRegistry = vpux::createPassesRegistry(archKind);
passesRegistry->registerPasses();
```

### "Mixed" passes

Mixed passes share a common core algorithm but utilise platform-specific information to make decisions. Simply put, the same input IR produces different results across platforms.

If the pass logic differs between public platforms, you can use "standard" `if/else` or `switch/case` statements:

```cpp
// Assume:
//  - 37XX, 40XX and 50XX are public

// OK: special logic for a specific public platform
if(archKind == ArchKind::NPU37XX) {
    // do something for 37XX
}

// OK: special logic for multiple public platforms
if(archKind <= ArchKind::NPU40XX) {
    // do something for 37XX and 40XX
} else {
    // do something for 50XX
}
```

It’s worth noting that the information in the sections below is still useful when working with public platforms, as it can offer insights into writing flexible, platform-independent code. The core concept is to rely on [OOP](https://en.wikipedia.org/wiki/Object-oriented_programming) principles and, in particular, the [Strategy behavioral pattern](https://refactoring.guru/design-patterns/strategy). The main advantage of this pattern is that when a new platform is introduced, you don’t need to modify the pass code to change its behavior — adding a new implementation of the interface is enough. This approach helps maintain compliance with the Open/Closed Principle ([OCP](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle)).

#### "Classic" strategy

TODO-#196176

#### MLIR-based strategy

Consider `FuseActivationOps` pass in `IE` dialect that can be applied for all HW generations: it fuses activation functions(or `post-ops`) (e.g. `ReLU`, leaky `ReLU`) with operations that support post-processing. For different platforms the same operation can support different `post-ops` or can't support `post-op` at all. This behavior can be expressed using the following interface:

```MLIR
def IE_LayerWithPostOpInterface : OpInterface<"LayerWithPostOpInterface"> {
    let description = "Interface for operations that support post-processing";

    let cppNamespace = "vpux::IE";

    let methods = [
        InterfaceMethod<
            "Returns the post-processing operation attribute",
            "vpux::IE::PostOpAttr", "getPostOp", (ins)
        >,

        InterfaceMethod<
            "Set post-processing operation attribute from an operation",
            "void", "setPostOp", (ins "mlir::Operation*":$postOp)
        >,

        InterfaceMethod<
            "Checks if the operation supports a given post-processing operation",
            "bool", "isSupportedPostOp", (ins "mlir::Operation*":$postOp, "const FuncRef<void(const formatv_object_base&)>&":$logCb)
        >,

        //...
    ];
}
```

This abstraction decouples platform-dependent behavior from the pass code that uses it: instead of matching concrete operations depending on platform, the pass can determine whether producer of the `post-op` supports post-processing or not by casting to the interface:

```C++
// postOp is an activation functions (e.g. ReLU, leaky ReLU)
auto producerOp = mlir::dyn_cast_or_null<IE::LayerWithPostOpInterface>(
                                    postOp->getOperand(0).getDefiningOp());
if (producerOp == nullptr) {
    return matchFailed(
            _log, rewriter, postOp,
            "PostOp input is a block argument or the producer does not support post-processing");
}
```

Now we move on to the key point — the method of registering the interface. The intuitive approach would be to use the `ODS Framework`, meaning to change operation definition and attach the interface to the operation statically at project compile time. You can find more details in [primer_mlir](./primer_mlir.md#interfaces):

```MLIR
def IE_MultiplyOp :
        IE_LayerOp<"Multiply", [
                DeclareOpInterfaceMethods<IE_LayerWithPostOpInterface>
            ]
        > { ... }
```

This approach has a couple of drawback: 
- in particular, in the above example, where the interface is declared for `MultiplyOp`, this automatically means that the operation supports `post-ops` for all platforms, since it can be casted to `IE::LayerWithPostOpInterface`. At the same time it is possible that `MultiplyOp` supports a specific `post-op` type(e.g. `ReLu`) for the `50XX+`, but not for `37XX` and `40XX`;
- operation can support defferent `post-ops` depending on platform, so we will have to use `ArchKind` again to implement the interface.

To eliminate these drawbacks, we can use another option provided by `MLIR` and attach models(implementation of the interface) dynamically during network compilation, without modifying the .td files:

```C++
void vpux::VPU::arch50xx::registerLayerWithPostOpModelInterface(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, IE::IEDialect*) {
        // LayerWithPostOpModel inherits IE::LayerWithPostOpInterface::ExternalModel
        IE::MultiplyOp::attachInterface<LayerWithPostOpModel<IE::MultiplyOp>>(*ctx);
        //...
    });
}
```

For detailed information on how to attach an interface to an operation, please visit [MLIR Overview](https://mlir.llvm.org/docs/Interfaces). Consequently, for each platform we can specify which operations need the interface attached using a concrete model. For example, `LayerWithPostOpModel` can differ between `37XX`/`40XX` and `50XX`. The diagram below illustrates the dependencies between these classes: 

<p align="center">
  <img src="images/op_interface.png" width="70%"><br>
  <em>IE::LayerWithPostOpInterface dependency</em>
</p>

In order to registry necessary interfaces each platform should implement `IInterfaceRegistry`:

<p align="center">
  <img src="images/interface_registry.png" width="70%"><br>
  <em>Interface registry class diagram</em>
</p>

Before starting the compilation pipeline, the `registerInterfaces` method of the appropriate implementation will be called:

```C++
auto interfacesRegistry = createInterfacesRegistry(archKind);

// registry is a mlir::DialectRegistry
interfacesRegistry->registerInterfaces(registry);
```

#### Rewriter-based approach

In this example, the different behavior of the pass for different platforms is achieved by adding special rewriters. In the following example `FuseQuantizedOps` pass uses an instance of [`IGreedilyPassStrategy`](../../include/vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp) in order to retrieve patterns:

```C++
void FuseQuantizedOpsPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&ctx);

    // creating an instance of IGreedilyPassStrategy
    auto strategy = vpux::IE::createFuseQuantizedOpsStrategy(
                    &ctx, func, _seOpsEnabled, _seExperimentalOpsEnabled);
    // register platform-specific rewriters using the strategy
    strategy->addPatterns(patterns, _log);

    if (mlir::failed(applyPatternsGreedily(func, 
            std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
```

The `IGreedilyPassStrategy` can be implemented in different ways, depending on the version of the device:

```C++

// 37XX
void FuseQuantizedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    // ...
    patterns.add<FuseWithSlice>(ctx, log);
    patterns.add<FuseWithMaxPool>(ctx, /*isPerAxesQuantSupported=*/false, log);
    patterns.add<FuseWithTile>(ctx, log);
    // There are no FuseWithReduce rewriters
    patterns.add<FuseWithAveragePool>(ctx, false, log);
    patterns.add<FuseWithConcat>(ctx, log);
    patterns.add<FuseWithDepth2Space>(ctx, log);
    // ...
}

// 50XX
void FuseQuantizedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    // ...
    patterns.add<FuseWithSlice>(ctx, log);
    // FuseWithMaxPool has different value for the isPerAxesQuantSupported parameter
    patterns.add<FuseWithMaxPool>(ctx, /*isPerAxesQuantSupported=*/true, log);
    patterns.add<FuseWithTile>(ctx, log);
    patterns.add<FuseWithReduce<IE::ReduceMeanOp>>(ctx, log);
    patterns.add<FuseWithReduce<IE::ReduceSumOp>>(ctx, log);
    patterns.add<FuseWithAveragePool>(ctx, false, log);
    patterns.add<FuseWithConcat>(ctx, log);
    // There is no FuseWithDepth2Space
    // ...
}
```

The diagram below illustrates the dependencies between these classes: 

<p align="center">
  <img src="images/rewriter_base.png" width="70%"><br>
  <em>Rewriter-based approach scheme</em>
</p>

Another [`IConversionPassStrategy`](../../include/vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp) interface also provides the `markOpLegality` method, useful for setting up operation legality in passes which rely on the dialect conversion driver.

Rewriters can also depend on interfaces to write them in the most general form — kind of combination with [MLIR-based strategy](#mlir-based-strategy).

```MLIR
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=NPU40XX allow-custom-values=true" --unroll-distributed-ops-VPUX40XX  %s | FileCheck %s
```

More detailed information about vpux-opt can be found in the [how_to_test.md](../../../../guides/how_to_test.md) document.

### Canonicalization

TODO: #-86282

## Pipelines

Compiler has different pipeline for different HW generation. These pipelines are stored in appropriate HW folders: [NPU37XX](../include/vpux/compiler/NPU37XX/pipelines.cpp), etc. To build a pipeline, it is also necessary to implement `IPipelineStrategy` interface for each device:

<p align="center">
  <img src="images/pipeline.png" width="70%"><br>
  <em>Pipeline strategy class diagram</em>
</p>

Then it is used in this way:

```C++
auto pipelineFactory = createPipelineStrategy(arch);
// pm is mlir::PassManager
pipelineFactory->buildPipeline(pm, config, rootTiming, log);
```

The main advantage of this approach is that we can easily hide the pipeline for a new device containing platform-specific passes. The consequence of this separation is that there is no need to add passes to the pipeline that do not work with this device. Therefore, the size of the pipeline becomes smaller, only the necessary passes are involved. And it is possible to get rid of such code:

```C++
void MyPass::safeRunOnFunc() {
    // ...
    if (arch != config::ArchKind::NPU37XX) {
        return mlir::failure();
    }
    // ...
}
```

This approach also has a downside. It is not clear why this or that pass participates in one pipeline, but not in another. Are there HW restrictions or did developer forget to add it? A possible solution is to introduce as many sub-pipelines as possible to bring the main pipeline to a similar form:

```C++
// Only sub-pipelines and platform-specific passages should remain in the main pipeline

// 37XX
void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options, Logger log) {
    // ...
    IE::buildName1Pipeline(pm, log);
    IE::buildName2Pipeline(pm, log);
    pm.addPass(IE::arch37xx::createHwSpecific1Pass(log));
    IE::buildName3Pipeline(pm, log);
    // ...
}

// 40XX
void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions40XX& options, Logger log) {
    // ...
    IE::buildName1Pipeline(pm, log);
    pm.addPass(IE::arch40xx::createHwSpecific2Pass(log));
    IE::buildName2Pipeline(pm, log);
    IE::buildName3Pipeline(pm, log);
    // ...
}
```

Some [recommendations](../code_style.md#pipelines-and-passes) are already written in code style.

## Properties

TODO: #-196170

## Operations

<p align="center">
  <img src="images/operations.png" width="25%"><br>
  <em>Sets of operationse</em>
</p>

TODO: #-86281

There is no complex solution here yet. As a first step, operations are devided between several `ops.td` files depending on the HW version. And the logic of transformations again is based on op-interfaces.

In future we could proceed with platform-specific dialects if necessary:
- VPUIP37XX_SwKernelOp
- VPUIP40XX_ConvertDMAOp
- ..

For example, we already have platform-specific dialects like [VPUMI37XX](../../tblgen/vpux/compiler/dialect/VPUMI37XX/dialect.td).

## Attributes

TODO: #-88494

## Dispatched Inlining

### Motivation

MLIR's inliner will be called as part of the inliner pass. It does a lot behind the scenes, but when it comes to deciding if an operation can be inlined and how it shall be inlined, it is quite simple.

As an example, we take a look at `isLegalToInline()`: If it encounters some operation, it will lookup the dialect. Internally, the inliner saves a mapping from `mlir::Dialect*` to `mlir::DialectInlinerInterface`. If the map contains no such interface for the requested dialect, `false` will be returned. Otherwise the inliner dispatches to that particular inliner interface and return `interface->isLegalToInline(op)`. There are a handful of other functions that work in the same fashion.

We see two important things here: The inliner can only decide on a per-operation basis which interface to choose and the inliner only supports at most one interface per dialect. This means that the inliner **cannot** support multiple different inlining semantics for a particular operation out of the box! This is the motivation for our dispatched inliner interface system.

Our solution is to implement a `func` inliner interface that can then dynamically dispatch to other interfaces. The idea is that the user can add a special attribute, namely `{inliner_dispatch = #MyDialect.MyInlinerDispatchAttr}`, to `func` operations. `UnifiedFuncInlinerInterface` then dispatches to an inliner interface that is associatated with `#MyDialect.MyInlinerDispatchAttr`.

### Tutorial

#### Supporting operations of a custom dialect in the inliner

The common approach here is extending `mlir::DialectInlinerInterface` and implementing `isLegalToInline()`. The most trivial implementation looks like this:
```cpp
struct MyDialectInlinerInterface : public mlir::DialectInlinerInterface {
    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }
};
```
This then has to be registered during `MyDialect::initialize()`:
```cpp
void MyDialect::initialize() {
    // ...
    addInterface<MyDialectInlinerInterface>();
    // ...
}
```

This is enough to enable inlining in `MyDialect` if the default inlining behaviour (see `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp`) is enough.

#### Supporting custom call (and func, return) operations

Assume we want to implement a custom `MyDialect.Call` operation. It extends `CallOpInterface` and will therefore be handled by `UnifiedFuncInlinerInterface`. If we **don't** want to have the default behaviour (see `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp`) for that kind of operation, we can extend `mlir::DialectInlinerInterface`.
```cpp
struct MyDialectDispatchedInlinerInterface : public mlir::DialectInlinerInterface {
    bool isLegalToInline(mlir::Operation*, mlir::Operation*, bool) const final {
        return true;
    }

    bool isLegalToInline(mlir::Region*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    bool isLegalToInline(mlir::Operation*, mlir::Region*, bool, mlir::IRMapping&) const final {
        return true;
    }

    void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
        // custom logic
    }

    void processInlinedCallBlocks(mlir::Operation* call,
                                  mlir::iterator_range<mlir::Region::iterator> inlinedBlocks) const final {
        // custom logic
    }

    std::tuple<mlir::Block*, mlir::Block::iterator> getInlineBlockAndPoint(mlir::Operation* call) const final {
        // custom logic
    }

    void eraseCall(mlir::Operation* call) const final {
        // custom logic
    }
};
```
Additionally, we have to add an attribute to `MyDialect`. This attribute will be added to the func-like operations in `MyDialect` to tell `UnifiedFuncInlinerInterface` which interface to dispatch to.
```tablegen
def MyDialectInlinerDispatchAttr : InlinerDispatchAttr<MyDialect, "MyDialectInlinerDispatch">;
```
```mlir
MyDialect.Call {inliner_dispatch = #MyDialect.MyInlinerDispatchAttr} @someFunction() -> ()
// or if we just want to have different semantics for func ops
func.call {inliner_dispatch = #MyDialect.MyInlinerDispatchAttr} @someFunction() -> ()
```
We then have to register this interface in the `UnifiedFuncInlinerInterface`:
```cpp
void MyDialect::initialize() {
    // ...

    // support inlining for "normal" ops in MyDialect
    addInterfaces<MyDialectInlinerInterface>();

    // support for func-like ops in MyDialect
    auto funcDialect = getContext()->getLoadedDialect<mlir::func::FuncDialect>();
    assert(funcDialect != nullptr);

    auto interface = funcDialect->getRegisteredInterface<Core::UnifiedFuncInlinerInterface>();
    assert(interface != nullptr);

    interface->registerDispatchedInlinerInterface<MyDialect::MyDialectInlinerDispatchAttr, MyDialect::FuncInlinerInterface>();
}
```

Note: If no dispatched inliner interface is provided via `registerDispatchedInlinerInterface`, a fallback implementation which mirrors `mlir/lib/Dialect/Func/Extensions/InlinerExtension.cpp` is used! For a lot of use-cases this is enough as the default inlining behaviour is the desired one.

## HostCompile Compilation Pipeline

The HostCompile pipeline is a specialized compilation mode designed to partition a neural network into multiple independently compilable functions. Each function contains NPU code, which is subsequently compiled into separate ELF blobs, along with the main function, which contains CPU host code that manages these compiled blobs using the LevelZero API.

<img src="images/1_ir_levels_HostCompile.drawio.svg" alt="drawing" width="600"/>

### Overview

In the HostCompile pipeline, the network is divided into kernel functions and host functions. Kernel functions contain the NPU-specific code and are compiled into ELF blobs as usual. Host function, on the other hand, consist of operations from MLIR upstream dialects such as tensor, scf, async, memref and arith. This code is later compiled into CPU code responsible for orchestrating the execution of kernel functions, passing various input data, and aggregating their outputs.

### Example

Consider the following example:

```mlir
module @StaticEltwiseNHWC attributes {config.arch = #config.arch_kind<NPU40XX>, config.revisionID = #config.revision_id<REVISION_NONE>, config.compilationMode = #config.compilation_mode<HostCompile>} {
  module @Module_1 {
    // function which contains the NPU-specific code and supposed to be compiled into ELF blobs
    func.func private @main_func0(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
        return %0 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
    }
  }
  // Host function which will be compiled into CPU code responsible for orchestrating the execution of kernel functions
  func.func @main(%arg0: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %0 = tensor.empty(%dim) : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %dim_0 = tensor.dim %arg0, %c3 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    %1 = scf.for %arg2 = %c0 to %dim_0 step %c100 iter_args(%arg3 = %0) -> (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>) {
      %2 = affine.min #map(%arg2)[%dim_0]
      %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %extracted_slice_1 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}> to tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %3 = Core.NestedCall @Module_1::@main_func0(%extracted_slice, %extracted_slice_1) : (tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>, tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}>
      %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 720, %2] [1, 1, 1, 1] : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 100]> : tensor<4xsi64>, order = #NHWC}> into tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
      scf.yield %inserted_slice : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }
    return %1 : tensor<1x16x720x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 720, 1000]> : tensor<4xsi64>, order = #NHWC}>
  }
}
```
In the HostCompile pipeline, as demonstrated in the example, the code is distinctly divided into NPU kernel functions, such as `@Module_1::@main_func0`, and the host function `@main`, which orchestrates their execution.
This contrasts with the `DefaultHW` pipeline, where the compilation process generates a single ELF blob that encapsulates all NPU code, without including any CPU code for orchestration.

### How to use HostCompile mode

#### Locally

To compile a network end to end with `HostCompile` pipeline use one of the following options:

- Provide `NPU_COMPILATION_MODE` environment variable while using compile_tool:

`NPU_COMPILATION_MODE="HostCompile" ./compile_tool -d NPU.4000 -m ./net.onnx -o ./net.blob  -shape [1,3,4..6,7..10]`
- Create config file, specify compilation mode and use this config for compilation:

`./compile_tool -d NPU.4000 -m ./net.onnx -o ./net.blob -c ./extra_config_net.conf -shape [1,3,4..6,7..10]`

Below is the content of the `extra_config_net.conf` file

```plaintext
NPU_COMPILATION_MODE HostCompile
```

#### In CI
Specify "NPU_COMPILATION_MODE": "HostCompile" in `extra_config` section of JSON config file:
```json
  "networks": [
    {
      "name": "Model",
      "ir": "net.onnx",
      "category": "CID/precommit/VPU4000",
      "extra_config": {
        "NPU_PLATFORM": "VPU4000",
        "NPU_COMPILATION_MODE": "HostCompile"
      },
      "Compile": {
        "shape": "[1,3,10..1600,10..2560]"
      },
      "BenchmarkApp": {
        "disabled": true
      },
      "NetTest-CalcRef": {
        "disabled": true
      },
      "NetTest-Validate": {
        "disabled": true
      },
      "AccuracyCheck": {
        "disabled": true
      }
    }
  ]
```
