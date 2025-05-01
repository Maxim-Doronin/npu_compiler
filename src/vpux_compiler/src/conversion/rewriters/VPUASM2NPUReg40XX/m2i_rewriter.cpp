//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUASM2NPUReg40XX/m2i_rewriter.hpp"

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux::VPURegMapped;
using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace vpux {
namespace vpuasm2npureg40xx {

namespace {
void setNormFactor(VpuMediaTask& initValues, ::mlir::ArrayAttr normFactor) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = vpux::type::float16(val);
        return valFP16.to_bits();
    };

    auto normArr = parseFPArrayAttr<double>(normFactor);
    VPUX_THROW_UNLESS(normArr.size() == MEDIA_MAX_NUM_PLANES * 4 /*MEDIA_MAX_NUM_NORM_FACT*/,
                      "Normalization array is invalid");

    initValues.write<Registers::NormFactor_0, Fields::NormFact0>(getRawFP16(normArr[0]));
    initValues.write<Registers::NormFactor_0, Fields::NormFact1>(getRawFP16(normArr[1]));
    initValues.write<Registers::NormFactor_0, Fields::NormFact2>(getRawFP16(normArr[2]));
    initValues.write<Registers::NormFactor_0, Fields::NormFact3>(getRawFP16(normArr[3]));

    initValues.write<Registers::NormFactor_1, Fields::NormFact0>(getRawFP16(normArr[4]));
    initValues.write<Registers::NormFactor_1, Fields::NormFact1>(getRawFP16(normArr[5]));
    initValues.write<Registers::NormFactor_1, Fields::NormFact2>(getRawFP16(normArr[6]));
    initValues.write<Registers::NormFactor_1, Fields::NormFact3>(getRawFP16(normArr[7]));

    initValues.write<Registers::NormFactor_2, Fields::NormFact0>(getRawFP16(normArr[8]));
    initValues.write<Registers::NormFactor_2, Fields::NormFact1>(getRawFP16(normArr[9]));
    initValues.write<Registers::NormFactor_2, Fields::NormFact2>(getRawFP16(normArr[10]));
    initValues.write<Registers::NormFactor_2, Fields::NormFact3>(getRawFP16(normArr[11]));
}

uint8_t getBytesOfPackOfPixels(VPU::M2iColorFmt inFormat) {
    switch (inFormat) {
    case VPU::M2iColorFmt::PL_FP16_RGB:
    case VPU::M2iColorFmt::PL_FP16_YUV:
    case VPU::M2iColorFmt::SP_NV12_10:
    case VPU::M2iColorFmt::SP_P010:
        return 2;
    case VPU::M2iColorFmt::IL_RGB888:
        return 3;
    case VPU::M2iColorFmt::IL_RGB8888:
    case VPU::M2iColorFmt::IL_RGB30:
        return 4;
    default:
        return 1;
    };
}

void setMediaDimensions(VPUASM::DeclareBufferOp bufferOp, VPU::M2iColorFmt format, uint64_t& width, uint64_t& height) {
    auto elemShape = mlir::cast<vpux::NDTypeInterface>(bufferOp.getBufferType().getMemref()).getShape();

    switch (format) {
    case VPU::M2iColorFmt::PL_YUV420_8:
    case VPU::M2iColorFmt::SP_NV12_8:  // dims[] = N(0),H(1),W(2),C(3)
        // H / 3 * 2 -- These YUV formats have a full sized Y plane, and weaved U,V values,
        // hence we need to extract the height of the Y plane from the concatenated height
        height = elemShape[Dims4D::Act::C] / 3 * 2;
        width = elemShape[Dims4D::Act::H];
        break;

    case VPU::M2iColorFmt::IL_RGB888:  // dims[] = N(0),H(1),W(2),C(3)
        height = elemShape[Dims4D::Act::C];
        width = elemShape[Dims4D::Act::H];
        break;

    case VPU::M2iColorFmt::PL_RGB24:     // dims[] = N(0),C(1),H(2),W(3)
    case VPU::M2iColorFmt::PL_FP16_RGB:  // dims[] = N(0),C(1),H(2),W(3)
        height = elemShape[Dims4D::Act::H];
        width = elemShape[Dims4D::Act::W];
        break;

    default:
        VPUX_THROW("{0} format is not supported", format);
        break;
    }
}

void setInSizeDescription(VpuMediaTask& initValues, VPU::M2iColorFmt inFormat, uint64_t width, uint64_t height,
                          uint64_t m2iIndex) {
    uint64_t inSize0_ls(0), PSOB_inPS(0), inSize1_width(0), inSize1_height(0);
    uint64_t inSize1_ls(0), inSize2_width(0), inSize2_height(0), inSize2_ls(0);

    auto inSize0_width = width - 1;
    auto inSize0_height = height - 1;

    auto inSize0_PID = m2iIndex;

    switch (inFormat) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_YUV444_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width;
        break;

    case VPU::M2iColorFmt::PL_FP16_RGB:
        inSize0_ls = width * 2;
        PSOB_inPS = width * height * 2;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width * 2;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width * 2;
        break;

    case VPU::M2iColorFmt::PL_GRAY8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width;
        break;

    case VPU::M2iColorFmt::SP_NV12_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width - 1;
        inSize1_height = height / 2 - 1;
        inSize1_ls = width;
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width / 2 - 1;
        inSize1_height = height / 2 - 1;
        inSize1_ls = width / 2;
        inSize2_width = width / 2 - 1;
        inSize2_height = height / 2 - 1;
        inSize2_ls = width / 2;
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width / 2 - 1;
        inSize1_height = height - 1;
        inSize1_ls = width / 2;
        inSize2_width = width / 2 - 1;
        inSize2_height = height - 1;
        inSize2_ls = width / 2;
        break;

    case VPU::M2iColorFmt::IL_RGB888:
        inSize0_ls = width * 3;
        PSOB_inPS = width * height * 3;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width * 3;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width * 3;
        break;

    default:
        VPUX_THROW("invalid input format {0}", inFormat);
        break;
    }

    initValues.write<Registers::inSize0, Fields::ls>(inSize0_ls);
    initValues.write<Registers::inSize0, Fields::width>(inSize0_width);
    initValues.write<Registers::inSize0, Fields::height>(inSize0_height);
    initValues.write<Fields::pid>(inSize0_PID);

    initValues.write<Registers::inSize1, Fields::ls>(inSize1_ls);
    initValues.write<Registers::inSize1, Fields::width>(inSize1_width);
    initValues.write<Registers::inSize1, Fields::height>(inSize1_height);

    initValues.write<Registers::inSize2, Fields::ls>(inSize2_ls);
    initValues.write<Registers::inSize2, Fields::width>(inSize2_width);
    initValues.write<Registers::inSize2, Fields::height>(inSize2_height);

    initValues.write<Fields::inPS>(PSOB_inPS);
}

void setOutDescription(VpuMediaTask& initValues, VPU::M2iColorFmt outFormat, uint64_t outWidth, uint64_t outHeight) {
    uint64_t outScale0_width(0), outScale0_height(0);
    uint64_t psSc0Y(0), psSc0UV(0), lsSc0Y(0), lsSc0UV(0);

    switch (outFormat) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_GRAY8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        lsSc0Y = outWidth;
        break;

    case VPU::M2iColorFmt::PL_FP16_YUV:
    case VPU::M2iColorFmt::PL_FP16_RGB:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight * 2;
        lsSc0Y = outWidth * 2;
        break;

    case VPU::M2iColorFmt::SP_NV12_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight / 2;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth;
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight / 4;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth / 2;
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight / 2;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth / 2;
        break;

    case VPU::M2iColorFmt::PL_YUV444_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth;
        break;

    case VPU::M2iColorFmt::IL_RGB888:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight * 3;
        lsSc0Y = outWidth * 3;
        break;

    default:
        VPUX_THROW("invalid output format {0}", outFormat);
        break;
    }

    initValues.write<Fields::outScale0_width>(outScale0_width);
    initValues.write<Fields::outScale0_height>(outScale0_height);
    initValues.write<Fields::psSc0Y>(psSc0Y);
    initValues.write<Fields::psSc0UV>(psSc0UV);
    initValues.write<Fields::lsSc0Y>(lsSc0Y);
    initValues.write<Fields::lsSc0UV>(lsSc0UV);
}

bool isCscRequired(VPU::M2iColorFmt inFormat, VPU::M2iColorFmt outFormat) {
    // Automatically switch CSC on when input format and output format are different
    // and they are found in a viable conversion list
    llvm::DenseMap<VPU::M2iColorFmt, llvm::DenseSet<VPU::M2iColorFmt>> supportedInOutFormatMap = {
            {VPU::M2iColorFmt::SP_NV12_8,
             {VPU::M2iColorFmt::PL_RGB24, VPU::M2iColorFmt::IL_RGB888, VPU::M2iColorFmt::PL_FP16_RGB}},
            {VPU::M2iColorFmt::PL_RGB24,
             {VPU::M2iColorFmt::SP_NV12_8, VPU::M2iColorFmt::PL_YUV444_8, VPU::M2iColorFmt::PL_YUV422_8,
              VPU::M2iColorFmt::PL_GRAY8, VPU::M2iColorFmt::PL_YUV420_8}},
            {VPU::M2iColorFmt::IL_RGB888, {VPU::M2iColorFmt::SP_NV12_8}},
            {VPU::M2iColorFmt::PL_YUV444_8, {VPU::M2iColorFmt::PL_RGB24}},
            {VPU::M2iColorFmt::PL_YUV422_8, {VPU::M2iColorFmt::PL_RGB24}},
            {VPU::M2iColorFmt::PL_YUV420_8,
             {VPU::M2iColorFmt::PL_RGB24, VPU::M2iColorFmt::IL_RGB888, VPU::M2iColorFmt::PL_FP16_RGB}}};

    return (supportedInOutFormatMap.find(inFormat) != supportedInOutFormatMap.end() &&
            supportedInOutFormatMap[inFormat].find(outFormat) != supportedInOutFormatMap[inFormat].end());
}

}  // namespace
mlir::LogicalResult M2IRewriter::matchAndRewrite(VPUASM::M2IOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // // prepare MediaRegister
    VpuMediaTask descriptor;

    auto outFormat = static_cast<uint64_t>(origOp.getOutFmt());
    auto sampleType = static_cast<uint64_t>(origOp.getInterp());

    const auto chromaInRC = static_cast<uint64_t>(origOp.getChromaInReverseChannels());
    const auto lumaInRC = static_cast<uint64_t>(origOp.getLumaInReverseChannels());
    auto ifc = ((chromaInRC & 0x1) << 5) | ((lumaInRC & 0x1) << 4) | (getBytesOfPackOfPixels(origOp.getInFmt()) & 0xF);
    uint64_t irqMask = 1 << 15;

    const auto chromaOutRC = static_cast<uint64_t>(origOp.getChromaOutReverseChannels());
    const auto lumaOutRC = static_cast<uint64_t>(origOp.getLumaOutReverseChannels());
    auto ofc = ((chromaOutRC & 0x1) << 1) | (lumaOutRC & 0x1);

    uint64_t nextDescTileMask = 0;
    if (origOp.getNextLink().has_value()) {
        auto nextM2IRef = _symRefMap.lookupSymbol(origOp.getNextLink().value());
        if (auto nextM2ITaskBuffer = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(nextM2IRef)) {
            nextDescTileMask = VPUASM::getTileSelectMaskForBuffer(nextM2ITaskBuffer);
        }
    }

    uint64_t width(0), height(0), inputTileMask(0);
    auto m2iIndex = origOp.getTaskIndex().getValue();
    auto inBufferRef = _symRefMap.lookupSymbol(origOp.getInput());
    auto inBufferOp = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(inBufferRef);
    VPUX_THROW_UNLESS(inBufferOp, "Could not find symbol name entry for {0}", inBufferRef);
    inputTileMask = VPUASM::getTileSelectMaskForBuffer(inBufferOp);
    setMediaDimensions(inBufferOp, origOp.getInFmt(), width, height);
    setInSizeDescription(descriptor, origOp.getInFmt(), width, height, m2iIndex);
    auto roiWidth = width - 1;
    auto roiHeight = height - 1;

    uint64_t outWidth(0), outHeight(0), outputTileMask(0);
    auto outBufferRef = _symRefMap.lookupSymbol(origOp.getOutputBuff());
    auto outBufferOp = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(outBufferRef);
    VPUX_THROW_UNLESS(outBufferOp, "Could not find symbol name entry for {0}", outBufferRef);
    outputTileMask = VPUASM::getTileSelectMaskForBuffer(outBufferOp);
    setMediaDimensions(outBufferOp, origOp.getOutFmt(), outWidth, outHeight);
    setOutDescription(descriptor, origOp.getOutFmt(), outWidth, outHeight);
    outWidth = outWidth - 1;
    outHeight = outHeight - 1;

    if (origOp.getNorm().has_value()) {
        setNormFactor(descriptor, origOp.getNorm().value());
    }

    uint64_t operations(0);
    operations |= origOp.getDoCsc() ? (1 << 0) : 0;
    operations |= isCscRequired(origOp.getInFmt(), origOp.getOutFmt()) ? (1 << 3 | 1 << 0) : 0;  // CLAMP bit always set
    operations |= origOp.getDoNorm() ? 1 << 1 : 0;

    descriptor.write<Registers::inAddr0, Fields::inAddr>(inputTileMask);
    descriptor.write<Registers::inAddr1, Fields::inAddr>(inputTileMask);
    descriptor.write<Registers::inAddr2, Fields::inAddr>(inputTileMask);
    descriptor.write<Fields::inFormat>(static_cast<uint64_t>(origOp.getInFmt()));
    descriptor.write<Fields::outFormat>(outFormat);
    descriptor.write<Fields::sampleType>(sampleType);
    descriptor.write<Fields::numRois>(1);
    descriptor.write<Fields::IFC>(ifc);
    descriptor.write<Fields::IRQMask>(irqMask);
    descriptor.write<Fields::operations>(operations);
    descriptor.write<Fields::roiBase>(outputTileMask);
    descriptor.write<Fields::OFC>(ofc);
    descriptor.write<Fields::outFormatLocal>(outFormat);
    descriptor.write<Fields::samlingTypeLocal>(sampleType);
    descriptor.write<Fields::outScale0_width>(outWidth);
    descriptor.write<Fields::outScale0_height>(outHeight);
    descriptor.write<Fields::roiWidth>(roiWidth);
    descriptor.write<Fields::roiHeight>(roiHeight);
    descriptor.write<Fields::vSc_offset>(origOp.getTileOffsetY().value_or(0));
    descriptor.write<Fields::hSc_offset>(origOp.getTileOffsetX().value_or(0));
    descriptor.write<Fields::vSc_factor>(origOp.getScaleFactorY());
    descriptor.write<Fields::hSc_factor>(origOp.getScaleFactorX());
    descriptor.write<Fields::nextDesc>(nextDescTileMask);
    descriptor.write<Fields::barGateMaskLO>(VPUMI40XX::computeMaskLo(origOp.getWaitBarriers()));
    descriptor.write<Fields::barGateMaskHI>(VPUMI40XX::computeMaskHi(origOp.getWaitBarriers()));
    descriptor.write<Fields::barUpdateLO>(VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers()));
    descriptor.write<Fields::barUpdateHI>(VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers()));
    descriptor.write<Registers::media_barriers_sched_, Fields::start_after_>(origOp.getStartAfter());
    descriptor.write<Registers::media_barriers_sched_, Fields::clean_after_>(origOp.getCleanAfter());

    auto regM2IDescriptorAttr = VpuMediaTaskAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::M2IOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getInputAttr(),
                                       origOp.getOutputBuffAttr(), origOp.getProfilingDataAttr(),
                                       origOp.getNextLinkAttr(), regM2IDescriptorAttr);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

}  // namespace vpuasm2npureg40xx
}  // namespace vpux
