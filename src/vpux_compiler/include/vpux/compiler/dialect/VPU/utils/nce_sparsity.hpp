//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/utils/core/algo.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/optional.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

namespace NCESparsity {

// base_ptr is 9bits size
const int BASE_PTR_SIZE = 9;

const VPU::SparsitySupport FULLY_SUPPORTED_SPARSITY_MODE =
        SparsitySupport::SPARSE_INPUTS | SparsitySupport::SPARSE_OUTPUTS | SparsitySupport::SPARSE_WEIGHTS;

constexpr int32_t SPARSITY_PTR_WHEN_NO_SPARSITY = 0xFFFFFF;

const unsigned int DEFAULT_SPARSIFIABLE_INPUT_OPERAND_ID = 0;
const unsigned int ELTWISE_SPARSIFIABLE_SECOND_INPUT_OPERAND_ID = 1;

constexpr std::int32_t ALIGNMENT_REQUIREMENT_IN_ELEMENTS = 16;
constexpr std::int32_t WEIGHTS_TABLE_READER_COUNT = 8;
// each weight reader receives up to 32 bytes of data. for data/sparsity pointer tables, each element has 4 bytes, which
// means that there are (at most) 8 elements to be read by each weight reader. so, by multiplying
// WEIGHTS_TABLE_READER_COUNT with this value (8), we obtain WEIGHTS_TABLE_READER_ALIGNMENT = 64. For each workload, its
// size has to be rounded up to the nearest multiple of WEIGHTS_TABLE_READER_ALIGNMENT = 64, the values to be added
// representing the padding.
constexpr std::int32_t WEIGHTS_TABLE_READER_ALIGNMENT = WEIGHTS_TABLE_READER_COUNT * 8;
// in each final (and padded) group of 64 weights table sets there has to be a multiple of 16 valid sets (16, 32, 48 or
// 64), that can be read from memory
constexpr std::int32_t WEIGHTS_TABLE_SETS_MIN_ALIGNMENT = 16;

enum class Mode { DW_CONV, POOL };

template <typename ScaleElemType>
llvm::unique_function<ScaleElemType(size_t)> getMultShiftFunc(mlir::Type inElemType, mlir::Type outElemType,
                                                              mlir::Type weightsType,
                                                              VPU::NCESparsity::PPEConverterCb ppeConverter, size_t OC,
                                                              mlir::FloatAttr constMultiplyFpScale,
                                                              mlir::Type scalesType = nullptr) {
    if (weightsType != nullptr) {
        auto inStorageType = mlir::quant::QuantizedType::castToStorageType(inElemType);
        if ((mlir::isa<mlir::quant::QuantizedType>(inElemType) && !mlir::isa<mlir::quant::QuantizedType>(weightsType) &&
             !inStorageType.isFloat8E5M2() && !inStorageType.isFloat8E4M3FN())) {
            VPUX_THROW("Unsupported In/Wt mixed precision. Got: in type {0}, wt type {1}", inElemType, weightsType);
        }
    }

    auto inQuantScales = extractScalesOrDefault(inElemType, 1.0);
    auto weightsQuantScales = extractScalesOrDefault(weightsType, 1.0);
    auto outQuantScales = extractScalesOrDefault(outElemType, 1.0);
    const auto constMultiplyScale = constMultiplyFpScale ? constMultiplyFpScale.getValueAsDouble() : 1.0;

    broadcast(inQuantScales, OC);
    broadcast(weightsQuantScales, OC);
    broadcast(outQuantScales, OC);

    std::vector<double> rescale;
    rescale.reserve(OC);
    for (size_t i = 0; i < OC; ++i) {
        const auto scale = (weightsQuantScales[i] * inQuantScales[i]) / outQuantScales[i] * constMultiplyScale;
        rescale.push_back(scale);
    }

    return [rescale = std::move(rescale), inElemType, scalesType, ppeConverter](size_t oc) {
        const auto quantScale = rescale[oc];

        // scalesType is set only when using the new weights table format, as in this case
        // the scale data type is independent of the input data type
        if (scalesType != nullptr) {
            auto multShift = ppeConverter(0, 0, rescale[oc], scalesType);
            return std::get<ScaleElemType>(multShift);
        } else {
            const QuantizationApproximation scaleApproximation(quantScale);
            auto multShift = ppeConverter(checked_cast<uint8_t>(scaleApproximation.shift()),
                                          checked_cast<int16_t>(scaleApproximation.mult()), rescale[oc], inElemType);

            return std::get<ScaleElemType>(multShift);
        }
    };
}

// this function is used for formatting both data pointers and sparsity pointers
// for getting a formatted sparsity pointer, provide only the pointer (as there is no zeropoint in this case)
template <typename T>
int32_t getDataOrSparsityPointer(int32_t pointer, T zeroPoint = 0) {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "Invalid zero-point type, expected int8_t or uint8_t");

    constexpr int32_t ZP_OFFSET = 24;
    T ZP_VALUE = zeroPoint;

    int32_t PTR_VALUE = pointer;
    constexpr int32_t PTR_VALIDATION_MASK = 0xFF00000F;

    if (PTR_VALUE & PTR_VALIDATION_MASK) {
        VPUX_THROW("The value that's stored for the pointer ({0}) has to fit on 24 bits, having the last 4 bits equal "
                   "to 0",
                   PTR_VALUE);
    }

    return (ZP_VALUE << ZP_OFFSET) | PTR_VALUE;
}

int64_t getBitPatternSize(Mode mode, ShapeRef kernelSize, int64_t SX, mlir::Type elemType, int64_t IC);

int32_t getWeightPtrStep(mlir::Value weights);

std::vector<int32_t> getExpandedWeightsTable(ArrayRef<int32_t> weightsTableVector, int64_t OC);

std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType,
                                     std::optional<int32_t> weightsPtrOffset, int32_t weightsPtrStep,
                                     std::optional<int32_t> sparsityPtrOffset, int32_t sparsityPtrStep,
                                     VPU::NCESparsity::PPEConverterCb ppeConverter,
                                     VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {},
                                     mlir::FloatAttr constScale = nullptr);
std::vector<int32_t> getWeightsTable(mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> weightPtrs,
                                     ArrayRef<int32_t> sparsityPtrs, VPU::NCESparsity::PPEConverterCb ppeConverter,
                                     VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                     mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {},
                                     mlir::FloatAttr constScale = nullptr);

//
// NewWeightsTableFormatMapper
//

class NewWeightsTableFormatMapper {
public:
    // utility function
    static int32_t normalizeKAndReturnCurrentGroupOf128ZeroPoints(int32_t index, int32_t& k);

    // encoding and decoding functions between the zero point initial index and its new index in the new format
    // these two functions use mathematical computations in order to compute the result
    static int32_t mathematicallyEncodePositionInNewZeroPointOnlyTableLayout(int32_t zeroPointIndex, int32_t k);
    static int32_t mathematicallyDecodePositionInNewZeroPointOnlyTableLayout(int32_t position, int32_t k);

    // utility functions
    static std::vector<int32_t> computeInversePermutation(std::vector<int32_t> v);
    static std::vector<int32_t> computePointerTable(const std::vector<int32_t>& v);
    static std::vector<int32_t> getZeroPointTableByK(int32_t k);
    static std::vector<int32_t> getZeroPointInversePermutationTableByK(int32_t k);
    static std::vector<int32_t> getPointerTableByK(int32_t k);

    // encoding and decoding functions between the zero point initial index and its new index in the new format
    // these two functions use statically-constructed vectors in order to deduce the result
    static int32_t encodePositionInNewZeroPointOnlyTableLayout(int32_t zeroPointIndex, int32_t k);
    static int32_t decodePositionInNewZeroPointOnlyTableLayout(int32_t position, int32_t k);

    // store 2 zero-points in one byte:
    // lowerZeroPoint in the least significant 4 bits and upperZeroPoint in the most significant 4 bits
    // typename T should be one of uint8_t (for U4 zero-points) and int8_t (for I4 zero-points)
    template <typename T>
    static T storeTwoZPInZPPalletizedByte(T lowerZeroPoint, T upperZeroPoint) {
        constexpr uint8_t mask = 0x0F;
        constexpr uint8_t shiftUpperZeroPointToUpperPartInByte = 4;
        return (lowerZeroPoint & mask) | ((upperZeroPoint & mask) << shiftUpperZeroPointToUpperPartInByte);
    }

    // in a zero-point palletized byte there are stored 2 zero-points, each one on 4 bits
    // if lowerZP = true, the zero-point stored in the least significant 4 bits will be returned
    // if lowerZP = false, the zero-point stored in the most significant 4 bits will be returned
    static int8_t extractOneZPFromZPPalletizedByte(int8_t zeroPoint, bool lowerZP);

    // in a zero-point palletized byte there are stored 2 zero-points, each one on 4 bits
    // if lowerZP = true, the zero-point stored in the least significant 4 bits will be returned
    // if lowerZP = false, the zero-point stored in the most significant 4 bits will be returned
    static uint8_t extractOneZPFromZPPalletizedByte(uint8_t zeroPoint, bool lowerZP);

    // utility function used by constructNewZeroPointOnlyTable
    template <typename T>
    static void mapElementsToNewFormat(bool isZeroPoint4Bit, std::vector<T>& table, int32_t start, int32_t end,
                                       std::vector<T>& result) {
        int32_t range = end - start;

        VPUX_THROW_WHEN(start % 128 != 0, "The starting index of the range ({0}) is not a multiple of 128", start);
        VPUX_THROW_WHEN(range % 16 != 0, "Range length ({0}) is not a multiple of 16", range);
        VPUX_THROW_WHEN(range < 16 || range > 128, "Range ({0}) should be between 16 and 128", range);
        VPUX_THROW_WHEN(
                start / 128 != (end - 1) / 128,
                "All weight sets have to be from the same group of (at most) 128 weight sets: {0} / 128 != {1} / 128",
                start, end - 1);

        // if isZeroPoint4Bit = true, divide by 2, as 2 zero-points will be stored in each byte
        // otherwise, only 1 zero-point will be stored in each byte
        if (isZeroPoint4Bit) {
            start /= 2;
            end /= 2;
        }

        auto map = getZeroPointTableByK(range);
        for (auto index = start; index < end; index++) {
            if (isZeroPoint4Bit) {
                auto lowerZPPositionInOriginalTable = start + map[(2 * index) % 128];
                auto upperZPPositionInOriginalTable = start + map[(2 * index + 1) % 128];
                result[index] = storeTwoZPInZPPalletizedByte<T>(table[lowerZPPositionInOriginalTable],
                                                                table[upperZPPositionInOriginalTable]);
            } else {
                auto positionInOriginalTable = start + map[index % 128];
                result[index] = table[positionInOriginalTable];
            }
        }
    }

    // function that constructs a zero point only table (in the new format) starting from a vector that
    // contains the zero points in ascending order based on their indices
    // typename T should be one of uint8_t (for U4 and U8) and int8_t (for I4 and I8)
    // set isZeroPoint4Bit = true only if the zero points should be stored on 4 bits each
    template <typename T>
    static std::vector<T> constructNewZeroPointOnlyTable(bool isZeroPoint4Bit, std::vector<T> table) {
        static_assert(
                std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Typename should be one of uint8_t (for U4 and U8 zero-points) and int8_t (for I4 and I8 zero-points)");

        int32_t k = table.size();

        int32_t countGroupsOf128ZeroPoints = k / 128;
        int32_t remainingZeroPointsInLastGroup = k % 128;

        // if isZeroPoint4Bit = true, mappedTable will have k/2 elements (bytes), as 2 zero-points will be stored in
        // each byte
        // otherwise, mappedTable will have k elements (bytes), as only 1 zero-point will be stored in each byte
        std::vector<T> mappedTable(isZeroPoint4Bit ? k / 2 : k, -1);

        for (int index = 0; index < countGroupsOf128ZeroPoints; index++) {
            mapElementsToNewFormat<T>(isZeroPoint4Bit, table, index * 128, index * 128 + 128, mappedTable);
        }

        if (remainingZeroPointsInLastGroup) {
            mapElementsToNewFormat<T>(isZeroPoint4Bit, table, countGroupsOf128ZeroPoints * 128,
                                      countGroupsOf128ZeroPoints * 128 + remainingZeroPointsInLastGroup, mappedTable);
        }

        return mappedTable;
    }

    static void mapElementsToNewPointerTableFormat(const std::vector<int32_t>& ptrs, int32_t start, int32_t end,
                                                   int32_t ptrStartingIndex, std::vector<int32_t>& result);
    static void constructNewPointerTableForWorkload(std::vector<int32_t>& mappedTable, int32_t workloadStartingIndex,
                                                    int32_t workloadSize, int32_t ptrStartingIndex,
                                                    const std::vector<int32_t>& ptrs);
    static std::vector<int32_t> constructNewPointerTable(ArrayRef<int32_t> workloadsSizes, ArrayRef<int32_t> ptrs);
    static std::vector<int32_t> constructNewPointerTable(ArrayRef<VPUIP::DPUTaskOp> tasks, ArrayRef<int32_t> ptrs);

private:
    static std::vector<int32_t> zeroPointsK16;
    static std::vector<int32_t> zeroPointsK32;
    static std::vector<int32_t> zeroPointsK48;
    static std::vector<int32_t> zeroPointsK64;
    static std::vector<int32_t> zeroPointsK80;
    static std::vector<int32_t> zeroPointsK96;
    static std::vector<int32_t> zeroPointsK112;
    static std::vector<int32_t> zeroPointsK128;

    static std::vector<int32_t> zeroPointsK16InversePermutation;
    static std::vector<int32_t> zeroPointsK32InversePermutation;
    static std::vector<int32_t> zeroPointsK48InversePermutation;
    static std::vector<int32_t> zeroPointsK64InversePermutation;
    static std::vector<int32_t> zeroPointsK80InversePermutation;
    static std::vector<int32_t> zeroPointsK96InversePermutation;
    static std::vector<int32_t> zeroPointsK112InversePermutation;
    static std::vector<int32_t> zeroPointsK128InversePermutation;

    static int32_t paddingValue;
    static std::vector<int32_t> pointersK16;
    static std::vector<int32_t> pointersK32;
    static std::vector<int32_t> pointersK48;
    static std::vector<int32_t> pointersK64;

    static std::vector<std::vector<int32_t>> zeroPointTables;
    static std::vector<std::vector<int32_t>> zeroPointInversePermutationTables;
    static std::vector<std::vector<int32_t>> pointerTables;
};
/**
 * @brief generate a dense data-pointer table
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadsSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrs - the addresses at which the data-pointers will be stored
 * @param OC - number of output channels
 * @param zeroPoints - array containing per-channel zero-points (one zero-point for each channel is needed; if the array
 * is empty, each zero-point will be 0) - for each channel, the zero-point and the actual weights starting address will
 * be embedded in one entry of 32 bits, formally called data-pointer
 *
 * @return constructed and formatted data-pointer table.
 */
template <typename T>
std::vector<int32_t> getDataPointerTable(mlir::Type inElemType, mlir::Type outElemType,
                                         ArrayRef<int32_t> workloadsSizes, ArrayRef<int32_t> weightsPtrs, int64_t OC,
                                         ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "Invalid zero-point type, expected int8_t or uint8_t");

    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create data pointer table without operation input/output types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Data pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(
            static_cast<int64_t>(zeroPoints.size()) != OC && static_cast<int64_t>(zeroPoints.size()) != 0,
            "Zero-points size {0} different than output channels {1} (and different than 0 - the default value)",
            zeroPoints.size(), OC);

    std::vector<T> zeroPointsVector(zeroPoints.begin(), zeroPoints.end());
    if (zeroPointsVector.empty()) {
        zeroPointsVector.assign(OC, 0);
    }

    std::vector<std::int32_t> dataPointerTableVals(OC, 0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);

        dataPointerTableVals[oc] = getDataOrSparsityPointer(weightsPtrs[oc], zeroPointsVector[oc]);
    });

    return NewWeightsTableFormatMapper::constructNewPointerTable(workloadsSizes, dataPointerTableVals);
}

/**
 * @brief generate a dense data-pointer table
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadsSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrOffset - the address at which the first data-pointer is stored (defaults to 0)
 * @param weightsPtrStep - distance between two consecutive data-pointer addresses
 * @param OC - number of output channels
 * @param zeroPoints - array containing per-channel zero-points (can be empty if not needed)
 *
 * @return constructed and formatted data-pointer table.
 */
template <typename T>
std::vector<int32_t> getDataPointerTable(mlir::Type inElemType, mlir::Type outElemType,
                                         ArrayRef<int32_t> workloadsSizes, std::optional<int32_t> weightsPtrOffset,
                                         int32_t weightsPtrStep, int64_t OC, ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    auto weightsPtrOffsetValue = weightsPtrOffset.value_or(0);

    SmallVector<int32_t> weightsPtrs(OC, 0);
    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffsetValue;
        weightsPtrOffsetValue += weightsPtrStep;
    }

    return getDataPointerTable<T>(inElemType, outElemType, workloadsSizes, weightsPtrs, OC, zeroPoints);
}

/**
 * @brief generate sparse data-pointer tables
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadsSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrs - the addresses at which the data-pointers will be stored
 * @param sparsityPtrs - the addresses at which the sparsity-pointers will be stored
 * @param OC - number of output channels
 * @param zeroPoints - array containing per-channel zero-points (one zero-point for each channel is needed; if the array
 * is empty, each zero-point will be 0) - for each channel, the zero-point and the actual weights starting address will
 * be embedded in one entry of 32 bits, formally called data-pointer
 *
 * @return constructed and formatted data-pointer and sparsity-pointer tables.
 */
template <typename T>
std::pair<std::vector<int32_t>, std::vector<int32_t>> getSparseDataPointerTablePair(
        mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> workloadsSizes, ArrayRef<int32_t> weightsPtrs,
        ArrayRef<int32_t> sparsityPtrs, int64_t OC, ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    static_assert(std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
                  "Invalid zero-point type, expected int8_t or uint8_t");

    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create sparse data pointer tables without operation input/output/weights types");
    VPUX_THROW_WHEN(static_cast<int64_t>(weightsPtrs.size()) != OC,
                    "Data pointers size {0} different than output channels {1}", weightsPtrs.size(), OC);
    VPUX_THROW_WHEN(static_cast<int64_t>(sparsityPtrs.size()) != OC,
                    "Sparsity pointers size {0} different than output channels {1}", sparsityPtrs.size(), OC);
    VPUX_THROW_WHEN(
            static_cast<int64_t>(zeroPoints.size()) != OC && static_cast<int64_t>(zeroPoints.size()) != 0,
            "Zero-points size {0} different than output channels {1} (and different than 0 - the default value)",
            zeroPoints.size(), OC);

    std::vector<T> zeroPointsVector(zeroPoints.begin(), zeroPoints.end());
    if (zeroPointsVector.empty()) {
        zeroPointsVector.assign(OC, 0);
    }

    std::vector<std::int32_t> dataPointerTableVals(OC, 0);
    std::vector<std::int32_t> sparsityPointerTableVals(OC, 0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        VPUX_THROW_UNLESS(weightsPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "weightsPtrs[{0}] must be multiple of {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          weightsPtrs[oc]);
        VPUX_THROW_UNLESS(sparsityPtrs[oc] == SPARSITY_PTR_WHEN_NO_SPARSITY ||
                                  sparsityPtrs[oc] % ALIGNMENT_REQUIREMENT_IN_ELEMENTS == 0,
                          "sparsityPtrs[{0}] must be aligned to {1}, got {2}", oc, ALIGNMENT_REQUIREMENT_IN_ELEMENTS,
                          sparsityPtrs[oc]);

        dataPointerTableVals[oc] = getDataOrSparsityPointer(weightsPtrs[oc], zeroPointsVector[oc]);
        sparsityPointerTableVals[oc] = getDataOrSparsityPointer<uint8_t>(sparsityPtrs[oc]);
    });

    const auto dataPointerTableFormatted =
            NewWeightsTableFormatMapper::constructNewPointerTable(workloadsSizes, dataPointerTableVals);
    const auto sparsityPointerTableFormatted =
            NewWeightsTableFormatMapper::constructNewPointerTable(workloadsSizes, sparsityPointerTableVals);

    return {dataPointerTableFormatted, sparsityPointerTableFormatted};
}

/**
 * @brief generate sparse data-pointer tables
 *
 * @param inElemType - input tensor type
 * @param outElemType - output tensor type
 * @param workloadsSizes - specifies the size of each workload - this size is equivalent to the number of weight sets in
 * that particular workload
 * @param weightsPtrOffset - the address at which the first data-pointer is stored (defaults to 0)
 * @param weightsShape - weights tensor's shape
 * @param sparsityPtrOffset - the address at which the first sparsity-pointer is stored
 * @param sparsityMap - one sparsity value for each weight (sparsity == 0 means weight == 0, sparsity != 0 means
 * weight != 0)
 * @param OC - number of output channels
 * @param weightsElemType - weights tensor type
 * @param zeroPoints - array containing per-channel zero-points (can be empty if not needed)
 *
 * @return constructed and formatted data-pointer and sparsity-pointer tables.
 *
 * NOTE: even if sparsityMap can contain uint8 values, in the generated structure we will have only 0s (when weight ==
 * 0), and 1s (when weight != 0), each stored on 1 bit.
 */
template <typename T>
std::pair<std::vector<int32_t>, std::vector<int32_t>> getSparseDataPointerTablePair(
        mlir::Type inElemType, mlir::Type outElemType, ArrayRef<int32_t> workloadsSizes,
        std::optional<int32_t> weightsPtrOffset, ShapeRef weightsShape, int32_t sparsityPtrOffset,
        ArrayRef<uint8_t> sparsityMap, int64_t OC, mlir::Type weightsElemType, ArrayRef<T> zeroPoints = ArrayRef<T>()) {
    VPUX_THROW_WHEN(weightsElemType == nullptr,
                    "Can't create sparse data pointer tables without operation weights type");

    const int32_t weightSetsCounter =
            weightsShape[Dims4D::Filter::IC] * weightsShape[Dims4D::Filter::KY] * weightsShape[Dims4D::Filter::KX];

    VPUX_THROW_WHEN(static_cast<size_t>(weightSetsCounter * weightsShape[Dims4D::Filter::OC]) != sparsityMap.size(),
                    "There has to be one sparse value for each weight");

    auto weightsPtrOffsetValue = weightsPtrOffset.value_or(0);
    auto sparsityPtrStep = vpux::alignValUp(weightSetsCounter / 8, ALIGNMENT_REQUIREMENT_IN_ELEMENTS);

    SmallVector<int32_t> weightsPtrs(OC, 0);
    SmallVector<int32_t> sparsityPtrs(OC, 0);
    int32_t sparsityTableOffset = 0;

    for (auto oc : irange(OC)) {
        weightsPtrs[oc] = weightsPtrOffsetValue;

        int32_t nonZeroWeightsCounter =
                std::count_if(sparsityMap.begin() + sparsityTableOffset,
                              sparsityMap.begin() + sparsityTableOffset + weightSetsCounter, [](int elem) {
                                  return elem != 0;
                              });
        const Bit eltSize = getElemTypeSize(weightsElemType);

        auto weightsPtrStepSizeInBits =
                vpux::alignValUp(checked_cast<int32_t>(eltSize.count()) * nonZeroWeightsCounter, CHAR_BIT);
        auto weightsPtrStep = weightsPtrStepSizeInBits / CHAR_BIT;
        weightsPtrOffsetValue += vpux::alignValUp(weightsPtrStep, ALIGNMENT_REQUIREMENT_IN_ELEMENTS);

        sparsityPtrs[oc] = sparsityPtrOffset;
        sparsityPtrOffset += sparsityPtrStep;
        sparsityTableOffset += weightSetsCounter;
    }

    return getSparseDataPointerTablePair<T>(inElemType, outElemType, workloadsSizes, weightsPtrs, sparsityPtrs, OC,
                                            zeroPoints);
}

template <typename T>
std::vector<T> getScaleTable(mlir::Type inElemType, mlir::Type outElemType,
                             VPU::NCESparsity::PPEConverterCb ppeConverter, int64_t OC,
                             mlir::Type weightsElemType = nullptr, mlir::FloatAttr constScale = nullptr) {
    VPUX_THROW_WHEN(inElemType == nullptr || outElemType == nullptr,
                    "Can't create weights table without operation input/output types");

    mlir::Type scalesType = nullptr;
    if (std::is_same<T, float>::value) {
        scalesType = mlir::Float32Type::get(inElemType.getContext());
    } else if (std::is_same<T, vpux::type::float8_e5m2>::value) {
        scalesType = mlir::Float8E5M2Type::get(inElemType.getContext());
    } else if (std::is_same<T, vpux::type::float8_e4m3>::value) {
        scalesType = mlir::Float8E4M3FNType::get(inElemType.getContext());
    } else {
        VPUX_THROW("Only F32/F8E5M2/F8E4M3 scales are supported in the new weights table format");
    }

    auto getMultShift = getMultShiftFunc<T>(inElemType, outElemType, weightsElemType, ppeConverter,
                                            checked_cast<size_t>(OC), constScale, scalesType);

    std::vector<T> scaleTableVals(OC, 0.0);

    loop_1d(LoopExecPolicy::Parallel, inElemType.getContext(), checked_cast<size_t>(OC), [&](const size_t oc) {
        scaleTableVals[oc] = getMultShift(oc);
    });

    return scaleTableVals;
}

std::vector<float> getBiasTable(mlir::Type inElemType, mlir::Type outElemType,
                                VPU::NCESparsity::BiasConverterCb biasConverter, int64_t OC,
                                mlir::Type weightsElemType = nullptr, const Const::ContentAttr& bias = {});

// typename T should be one of uint8_t (for U4 and U8) and int8_t (for I4 and I8)
// set isZeroPoint4Bit = true only if the zero points should be stored on 4 bits each
// so, for U4 zero points set: T=uint8_t and isZeroPoint4Bit=true
// for U8 zero points set: T=uint8_t and isZeroPoint4Bit=false
// for I4 zero points set: T=int8_t and isZeroPoint4Bit=true
// for I8 zero points set: T=int8_t and isZeroPoint4Bit=false
template <typename T>
std::vector<T> getZeroPointOnlyTable(int64_t OC, mlir::Type weightsElemType, bool isZeroPoint4Bit,
                                     ArrayRef<T> zeroPoints) {
    VPUX_THROW_WHEN(weightsElemType == nullptr || !mlir::isa<mlir::quant::QuantizedType>(weightsElemType),
                    "weightsElemType has to be a quantized type, not {0}", weightsElemType);

    mlir::Type storageType = mlir::cast<mlir::quant::QuantizedType>(weightsElemType).getStorageType();

    if (std::is_same_v<T, uint8_t>) {
        VPUX_THROW_UNLESS((storageType.isUnsignedInteger(8) && !isZeroPoint4Bit) ||
                                  (storageType.isUnsignedInteger(4) && isZeroPoint4Bit),
                          "The storage type of the quantized weightsElemType {0} has to be the same as the type of the "
                          "zero points",
                          storageType);
    } else if (std::is_same_v<T, int8_t>) {
        VPUX_THROW_UNLESS(
                ((storageType.isSignedInteger(8) || storageType.isSignlessInteger(8)) && !isZeroPoint4Bit) ||
                        ((storageType.isSignedInteger(4) || storageType.isSignlessInteger(4)) && isZeroPoint4Bit),
                "The storage type of the quantized weightsElemType {0} has to be the same as the type of the "
                "zero points",
                storageType);
    } else {
        VPUX_THROW("Template argument type has to be uint8_t or int8_t");
    }

    VPUX_THROW_WHEN(static_cast<int64_t>(zeroPoints.size()) != OC,
                    "Zero-points size {0} different than output channels {1}", zeroPoints.size(), OC);

    return NewWeightsTableFormatMapper::constructNewZeroPointOnlyTable<T>(isZeroPoint4Bit, zeroPoints);
}

std::vector<int32_t> patchWeightsTableSparsityPtrs(const std::vector<std::int32_t>& weightsTableVals,
                                                   const int32_t sparsityPtrOffset, const int32_t sparsityPtrStep,
                                                   std::optional<int64_t> origOC = std::nullopt);

Shape inferWeightsTableShape(int64_t OC);
Shape inferWeightsTablesNewFormatShape(int64_t OC);
Shape inferWeightsSparsityMapShape(ShapeRef dataShape);

mlir::FailureOr<SmallVector<double>> getRescaledBias(const Const::ContentAttr& biasAttr, mlir::Type inElemType,
                                                     mlir::Type filterElemType, int64_t OC);

double getSparsityRatio(vpux::NDTypeInterface weightsType, ArrayRef<int64_t> numNonSparseElemsPerOC);

bool isSparsifiableWeightsOperand(mlir::Value operand);
bool isSuperdenseRequired(const DimsOrder outOrder, const ShapeRef outShape, const mlir::Type outElemType);
inline VPU::SparsitySupport bitwiseNot(const VPU::SparsitySupport bits) {
    static_assert(sizeof(bits) == sizeof(uint32_t), "VPU::SparsitySupport has unexpected size");
    return static_cast<VPU::SparsitySupport>(~static_cast<uint32_t>(bits));
}

// 5D weights.
int32_t get5DWeightPtrStep(mlir::Value weights);

std::vector<int32_t> create5DWeightsTableData(mlir::Value opInput, mlir::Type opOutputElemType, mlir::Value weights,
                                              const Const::ContentAttr& bias, int64_t outputChannels,
                                              VPU::NCESparsity::PPEConverterCb ppeConverter,
                                              VPU::NCESparsity::BiasConverterCb biasConverter, bool hasAutopad);
std::vector<int32_t> create5DWeightsTableData(mlir::Value opInput, mlir::Value opOutput, mlir::Value weights,
                                              const Const::ContentAttr& bias, int64_t outputChannels,
                                              VPU::NCESparsity::PPEConverterCb ppeConverter,
                                              VPU::NCESparsity::BiasConverterCb biasConverter, bool hasAutopad);

mlir::Value create5DWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<int32_t> weightsTable,
                                       int64_t outputChannels, int64_t groups);

//
// Convert real numbers to fixed point S16.16 format.
//

int32_t toFixedPoint(const double realVal);

//
// Convert real numbers to hex format.
//

int32_t toHex(double realVal);

//
// RuntimeSparsityStatsProvider
//

class RuntimeSparsityStatsProvider {
    const double MINIMAL_SPARSITY_THRESHOLD = 0.2;

public:
    RuntimeSparsityStatsProvider(mlir::func::FuncOp func, vpux::Logger log);

    bool containsStatistics() const;
    bool likelySparsityConsumer(mlir::Operation* op, int64_t requestedInputId) const;

private:
    vpux::Logger _logger;
    std::multimap<std::string, net::SparsityInfoOp> _lookup;
};

}  // namespace NCESparsity

}  // namespace VPU
}  // namespace vpux
