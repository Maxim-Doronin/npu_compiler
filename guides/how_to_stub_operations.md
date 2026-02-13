# How to stub operations

## Introduction 

When compiling a network, they can fail because they contain operations / layers which are not (fully) supported by the compiler. 

Operation Stubbing is a mechanism with which we can overcome this problem by replacing the non-supported operations with StubOps. StubOps are placeholder operators, without any side effects and respecting the shape / type consistency of the original stubbed ops. This feature helps to better identificate the operations which need to be enabled and to run the models end-to-end with these StubOps included.

As ```StubOp``` is only a placeholder operation, no executable code will be generated for this operator. Accuracy information is lost because of the stubbed layer.

StubOp is part of these dialects:
- IE
- VPU
- VPUIP

OperationStubbingPass is part of the VPUIP dialect.

## Usage

### Stubbing by operation type

The OperationStubbingPass is needed to be included into the compilation pipeline with its creation function, which accepts a condition function:

```cpp
std::unique_ptr<mlir::Pass> vpux::VPUIP::createOperationStubbingPass(std::function<bool(mlir::Operation*)> condition, Logger log)
```

In order to stub an operation by operation type, we need to place a stubbing pass into the pipeline in `src/vpux_compiler/src/pipelines.cpp`.

For example, if we want to stub every ```IE::PReluOp``` in the network, we will add the pass the following way:

```cpp
pm.addPass(VPUIP::createOperationStubbingPass(condition, log));
```

where the condition function is defined as:

```cpp
std::function<bool(mlir::Operation*)> condition = [](mlir::Operation* op) -> bool {
    return mlir::isa<IE::PReluOp>(op);
};
```

After recompiling the compiler, we can check the newly generated IR. In the IR after the OperationStubbing pass we can see that every occurence of ```IE::PReluOp``` will be converted to ```IE.Stub```.
```cpp
%0 = IE.PRelu(%1, %cst_0) : tensor<1x56x320x180xf16>, tensor<56xf16> -> tensor<1x56x320x180xf16>
```
```cpp
%0 = IE.Stub(%1, %cst_0) : tensor<1x56x320x180xf16>, tensor<56xf16> -> tensor<1x56x320x180xf16>
```

### Stubbing by operation type + operation condition
If we want to stub operations which have specific characteristics, we can check these in the condition function. For example, if we want to stub all ```VPU::MemPermuteOp```s which have input with rank greater than 4D, we will introduce this condition into the pipeline as:

```cpp
std::function<bool(mlir::Operation*)> condition3 = [](mlir::Operation* op) -> bool {
    if (mlir::isa<VPU::MemPermuteOp>(op)) {
        auto memPermuteOp = ::llvm::dyn_cast<VPU::MemPermuteOp>(op);
        if (mlir::cast<vpux::NDTypeInterface>(memPermuteOp.input().getType()).getRank() > 4) {
            return true;
        }
    }
    return false;
};
pm.addPass(VPUIP::createOperationStubbingPass(condition3, log));
```

# How to stub unsupported SW operations

If the cause of the failure is related to unsupported software kernels, they can be replaced with a Dummy SW kernel.

Enable replacement with `NPU_COMPILATION_MODE_PARAMS dummy-op-replacement=true` in the configuration file.

The config file can be provided to the compile_tool as:

```cpp
./compile_tool -d VPUX.3720 -c config.conf [other options]
```
