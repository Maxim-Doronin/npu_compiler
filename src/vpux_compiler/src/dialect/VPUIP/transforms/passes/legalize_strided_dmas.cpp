//
// Copyright (C) 2025-2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/utils/func_dialect.hpp>
#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/version.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <algorithm>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_LEGALIZESTRIDEDDMAS
#define GEN_PASS_DEF_LEGALIZESTRIDEDDMAS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

namespace {
class LegalizeStridedDmasPass final : public VPUIP::impl::LegalizeStridedDmasBase<LegalizeStridedDmasPass> {
public:
    explicit LegalizeStridedDmasPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void insertLegalizationInputDma(mlir::Value ioVal, mlir::Operation* incompatibleOp);
    void legalizeOutputDmas(mlir::func::FuncOp func, mlir::Value outputArgument,
                            SmallVector<VPUIP::DMATypeOpInterface>& dmaSet);
    bool areStridesCompatible(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType);
    void legalizeFunctionInputOutputDmas(mlir::func::FuncOp func, int operandIdx, bool isInput);
};

// Strides are considered compatible when they are the same save for difference in final dimension.
// For example following strides are compatible
// in [4, 2, 1] out [4, 4, 2, 1]
// Because the second one is just an expanded shape on a final dimension.
// Strides need to be compatible to ensure we can generate correct DMA descriptor later on
// which can only transfer a multiples of a stride on each dimension.
bool LegalizeStridedDmasPass::areStridesCompatible(NDTypeInterface inType, NDTypeInterface outType) {
    auto inStrides = inType.getMemStrides();
    auto outStrides = outType.getMemStrides();

    auto iterationLimit = inStrides.size() < outStrides.size() ? outStrides.size() : inStrides.size();
    for (size_t idx = 0; idx < iterationLimit; idx++) {
        auto inStride = idx < inStrides.size() ? inStrides[MemDim(inStrides.size() - 1 - idx)] : inStrides[MemDim(0)];
        auto outStride =
                idx < outStrides.size() ? outStrides[MemDim(outStrides.size() - 1 - idx)] : outStrides[MemDim(0)];
        if (inStride != outStride) {
            return false;
        }
    }
    return true;
}

void LegalizeStridedDmasPass::insertLegalizationInputDma(mlir::Value ioVal, mlir::Operation* incompatibleOp) {
    auto opBuilder = mlir::OpBuilder(incompatibleOp);
    auto newLocation = takeOpLoc(incompatibleOp, "stridedInputDmaLegalization");
    auto newMemref =
            opBuilder.create<mlir::memref::AllocOp>(newLocation, mlir::cast<mlir::MemRefType>(ioVal.getType()));
    auto newStridedDmaOp = opBuilder.create<VPUIP::NNDMAOp>(newLocation, ioVal, newMemref.getMemref());
    newStridedDmaOp->setAttr(vpux::stridedInputAttrName, mlir::UnitAttr::get(&getContext()));
    ioVal.replaceUsesWithIf(newStridedDmaOp->getResult(0), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == incompatibleOp;
    });
}

void LegalizeStridedDmasPass::legalizeOutputDmas(mlir::func::FuncOp func, mlir::Value outputArgument,
                                                 SmallVector<VPUIP::DMATypeOpInterface>& dmaSet) {
    bool isLegal = true;
    for (auto& dmaOp : dmaSet) {
        auto dmaType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getOutput().getType());
        auto argType = mlir::cast<vpux::NDTypeInterface>(outputArgument.getType());
        if (!areStridesCompatible(dmaType, argType)) {
            isLegal = false;
            break;
        }
    }

    if (isLegal) {
        for (auto& dmaOp : dmaSet) {
            dmaOp->setAttr(vpux::stridedOutputAttrName, mlir::UnitAttr::get(&getContext()));
        }
    } else {
        mlir::Operation* lastOutputDmaOp = nullptr;
        SmallVector<mlir::Value> concatInputs;
        for (auto dmaOp : dmaSet) {
            concatInputs.push_back(dmaOp.getOutput());
            if (lastOutputDmaOp == nullptr || !dmaOp->isBeforeInBlock(lastOutputDmaOp)) {
                lastOutputDmaOp = dmaOp.getOperation();
            }
        }

        auto opBuilder = mlir::OpBuilder(getOperation());
        opBuilder.setInsertionPointToStart(&func.getBody().front());
        auto newLocation = takeOpLoc(lastOutputDmaOp, "stridedOutputDmaLegalization");
        auto newMemref = opBuilder.create<mlir::memref::AllocOp>(
                newLocation, mlir::cast<mlir::MemRefType>(outputArgument.getType()));
        outputArgument.replaceUsesWithIf(newMemref.getMemref(), [&](mlir::OpOperand& opOperand) {
            return !mlir::isa<mlir::func::ReturnOp>(opOperand.getOwner());
        });

        opBuilder.setInsertionPointAfter(lastOutputDmaOp);
        auto newConcat = opBuilder.create<VPUIP::ConcatViewOp>(newLocation, concatInputs, newMemref.getMemref());
        auto newStridedDmaOp = opBuilder.create<VPUIP::NNDMAOp>(newLocation, newConcat.getOutput(), outputArgument);
        newStridedDmaOp->setAttr(vpux::stridedOutputAttrName, mlir::UnitAttr::get(&getContext()));
    }
}

/*
  The function below traverses the graph in a BFS fashion and searches for transformations
  incompatible with dynamic-strides DMAs.
  The IR is assumed to meet the following constraints:
  1. There is a DMA between a function argument and a compute layer.
  2. There can be any number of ViewLikeOps between the DMA operation and the function argument.
  3. There can be a function call operation between the function argument and a DMA.

  The algorithm works differently for inputs and outputs.
  For inputs, the algorithm will search for incompatible ViewLikeOps, and if one is found,
  it will allocate a new DDR buffer and a new DMA which will transfer the ViewLikeOp source
  to this new buffer, which will serve as the input to the incompatible op. Example:

  func(%input_arg) {
    %0 = IncompatibleViewLikeOp(%input_arg)
    %1 = FurtherViewLikeOp(%0)
  }

  is transformed into:

  func(%input_arg) {
    %new_alloc = memref.alloc()
    %new_dma = NNDMA {stridedInput} input(%input_arg) output(%new_alloc)
    %0 = IncompatibleViewLikeOp(%new_dma)
    %1 = FurtherViewLikeOp(%0)
  }

  For outputs, the algorithm will first search for all DMAs that are transferring data to
  the function output. Once we have a DMA set, the algorithm will check if all DMAs are
  compatible, and if even one of them isn't, a new DDR buffer is created to store compact
  data and a new DMA is added to transfer data from this new compact buffer to the strided
  function output.

  func(%output_arg) {
    %slice_0 = IncompatibleViewLike(%output_arg)
    %slice_1 = IncompatibleViewLike(%output_arg)
    %dma_0 = NNDMA inputs(%some_data) output(%slice_0)
    %dma_1 = NNDMA inputs(%some_other_data) output(%slice_1)
    return %output_arg
  }

  is transformed into:

  func(%output_arg) {
    %new_alloc = memref.alloc()
    %slice_0 = IncompatibleViewLike(%new_alloc)
    %slice_1 = IncompatibleViewLike(%new_alloc)
    %dma_0 = NNDMA inputs(%some_data) output(%slice_0)
    %dma_1 = NNDMA inputs(%some_other_data) output(%slice_1)
    %concat = ConcatOp inputs(%dma_0, %dma_1)
    %new_dma = NNDMA {stridedOutput} input(%concat) output(%output_arg)
    return %output_arg
  }

  Function calls are handled by calling the same algorithm recursively. This means
  that functions can't call each other (which can't happen in the current IR). The algorithm
  will also update a function even if it is called from multiple call sites, which can be
  inefficient in cases where the same function is later called with an input that is not
  function I/O. Those cases should be rare, as we mostly get multiple functions from
  vertical fusion outlining.
*/
void LegalizeStridedDmasPass::legalizeFunctionInputOutputDmas(mlir::func::FuncOp func, int operandIdx, bool isInput) {
    std::queue<mlir::OpOperand*> valuesToVisit;
    llvm::DenseSet<mlir::OpOperand*> visitedValues;
    mlir::SmallVector<VPUIP::DMATypeOpInterface> outputDmas;
    auto funcArgument = func.getBlocks().front().getArguments()[operandIdx];
    for (auto& use : funcArgument.getUses()) {
        valuesToVisit.push(&use);
    }
    while (!valuesToVisit.empty()) {
        auto use = valuesToVisit.front();
        valuesToVisit.pop();
        if (visitedValues.contains(use)) {
            continue;
        }
        visitedValues.insert(use);
        mlir::Operation* user = use->getOwner();
        llvm::TypeSwitch<mlir::Operation*, void>(user)
                .Case<VPUIP::DMATypeOpInterface>([&](VPUIP::DMATypeOpInterface dmaOp) {
                    if (isInput) {
                        if (mlir::isa<VPUIP::NNDMAOp, VPUIP::ConvertDMAOp, VPUIP::PermuteDMAOp>(dmaOp.getOperation())) {
                            dmaOp->setAttr(vpux::stridedInputAttrName, mlir::UnitAttr::get(&getContext()));
                        } else {
                            insertLegalizationInputDma(dmaOp.getInput(), dmaOp.getOperation());
                        }
                    } else {
                        outputDmas.push_back(dmaOp);
                    }
                })
                .Case<mlir::ViewLikeOpInterface>([&](mlir::ViewLikeOpInterface viewLikeOp) {
                    if (isInput) {
                        auto inType = mlir::cast<vpux::NDTypeInterface>(viewLikeOp.getViewSource().getType());
                        auto outType = mlir::cast<vpux::NDTypeInterface>(viewLikeOp->getResult(0).getType());
                        if (areStridesCompatible(inType, outType)) {
                            for (auto& viewUse : viewLikeOp->getUses()) {
                                valuesToVisit.push(&viewUse);
                            }
                        } else {
                            insertLegalizationInputDma(viewLikeOp.getViewSource(), viewLikeOp.getOperation());
                        }
                    } else {
                        for (auto& viewUse : viewLikeOp->getUses()) {
                            valuesToVisit.push(&viewUse);
                        }
                    }
                })
                .Case<mlir::func::CallOp>([&](mlir::func::CallOp callOp) {
                    auto calledFunc = getCalledFunction(callOp);
                    legalizeFunctionInputOutputDmas(calledFunc, use->getOperandNumber(), isInput);
                })
                .Case<mlir::func::ReturnOp>([](mlir::func::ReturnOp) {
                    return;
                })
                .Default([&](mlir::Operation*) {
                    VPUX_THROW("Unknown operation encountered");
                });
    }
    if (!isInput) {
        legalizeOutputDmas(func, funcArgument, outputDmas);
    }
}

void LegalizeStridedDmasPass::safeRunOnModule() {
    auto module = getOperation();
    net::NetworkInfoOp netInfo = *module.getOps<net::NetworkInfoOp>().begin();
    auto func = module.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPoint());
    auto funcArguments = func.getBlocks().front().getArguments();
    bool hasStridedIO = false;

    auto isArgInput = [&](size_t idx) {
        auto inputs = to_small_vector(netInfo.getInputsInfo().getOps<net::DataInfoOp>());
        return idx < inputs.size();
    };

    for (auto it : funcArguments | indexed) {
        auto argument = it.value();
        auto index = it.index();
        auto isInput = isArgInput(index);

        if (!vpux::net::isArgStrided(module, index)) {
            continue;
        }

        auto argumentElementSize = mlir::cast<vpux::NDTypeInterface>(argument.getType()).getElemTypeSize();
        if (argumentElementSize < vpux::Bit(8)) {
            continue;
        }

        hasStridedIO = true;
        legalizeFunctionInputOutputDmas(func, index, isInput);
    }

    if (hasStridedIO) {
        const auto currentElfVersion = config::getElfAbiVersion(module);
        const auto minimumElfVersion = config::getNPUConstraints(module.getContext()).dynamicStridesMinElfAbiVersion;
        if (currentElfVersion.has_value() && minimumElfVersion.has_value() &&
            minimumElfVersion.value() > currentElfVersion.value()) {
            config::setElfAbiVersion(module, minimumElfVersion.value());
            _log.warning("Updating ELF version to {0} due to presence of dynamic strides", minimumElfVersion.value());
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createLegalizeStridedDMAsPass(Logger log) {
    return std::make_unique<LegalizeStridedDmasPass>(log);
}
