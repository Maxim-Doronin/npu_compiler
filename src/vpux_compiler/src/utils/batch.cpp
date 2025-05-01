//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/batch.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/common_string_utils.hpp"

#include <charconv>
#include <functional>
#include <iterator>

using namespace vpux;

bool DebatchedCallOpData::canBeDeserialized(const SmallVector<ValueType>& array) {
    constexpr uint32_t minElementsInArray = 2;
    return array.size() >= minElementsInArray;
}

DebatchedCallOpData DebatchedCallOpData::deserialize(const SmallVector<ValueType>& array) {
    VPUX_THROW_UNLESS(DebatchedCallOpData::canBeDeserialized(array),
                      "Cannot deserialzie DebatchedCallOpData from array. More elements expected, got {1}",
                      array.size());
    return DebatchedCallOpData{array[0], array[1]};
}

SmallVector<DebatchedCallOpData::ValueType> DebatchedCallOpData::serialize() const {
    return SmallVector<ValueType>({callOpIndex, totalBatchSize});
}

std::string DebatchedCallOpData::to_string() const {
    return "call Index: " + std::to_string(callOpIndex) + ", batch size: " + std::to_string(totalBatchSize);
}

const DebatchedCallOpData& DebatchedCallOpAttributeView::getCallData() const {
    return data;
}

std::optional<DebatchedCallOpAttributeView> DebatchedCallOpAttributeView::extract(mlir::func::CallOp callOp) {
    if (!callOp->hasAttr(DebatchedCallOpAttributeView::name())) {
        return {};
    }
    auto attr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(callOp->getAttr(DebatchedCallOpAttributeView::name()));
    VPUX_THROW_UNLESS(attr != nullptr, "Unexpected type for \"{0}\", only \"mlir::ArrayAttr\" supported",
                      DebatchedCallOpAttributeView::name());
    return DebatchedCallOpAttributeView(
            DebatchedCallOpData::deserialize(parseIntArrayAttr<DebatchedCallOpData::ValueType>(attr)));
}

void DebatchedCallOpAttributeView::injectImpl(mlir::func::CallOp callOp) const {
    auto serializedArray = data.serialize();

    auto debatchedAttr = getIntArrayAttr(callOp->getContext(), serializedArray);
    VPUX_THROW_UNLESS(debatchedAttr != nullptr, "Cannot create 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::name());
    callOp->setAttr(DebatchedCallOpAttributeView::name(), debatchedAttr);
}

bool DebatchedCallOpAttributeView::hasAvailableTilesAttr(mlir::func::CallOp callOp) {
    return callOp->hasAttr(DebatchedCallOpAttributeView::availableTilesAttrName());
}

void DebatchedCallOpAttributeView::setAvailableTilesAttr(mlir::func::CallOp callOp,
                                                         DebatchedCallOpData::ValueType val) {
    VPUX_THROW_UNLESS(!hasAvailableTilesAttr(callOp),
                      "Detected existing 'DebatchedCallOpAttributeView' attribute \"{0}\", cannot create new attribute",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    auto newAttr = getIntAttr(callOp->getContext(), val);

    VPUX_THROW_UNLESS(newAttr != nullptr, "Failed to create new 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    callOp->setAttr(DebatchedCallOpAttributeView::availableTilesAttrName(), newAttr);
}

void DebatchedCallOpAttributeView::removeAvailableTilesAttr(mlir::func::CallOp callOp) {
    VPUX_THROW_UNLESS(hasAvailableTilesAttr(callOp),
                      "'DebatchedCallOpAttributeView' attribute \"{0}\" not found, cannot remove",
                      DebatchedCallOpAttributeView::availableTilesAttrName());

    callOp->removeAttr(DebatchedCallOpAttributeView::availableTilesAttrName());
}

DebatchedCallOpData::ValueType DebatchedCallOpAttributeView::getAvailableTilesVal(mlir::func::CallOp callOp) {
    VPUX_THROW_UNLESS(hasAvailableTilesAttr(callOp), "'DebatchedCallOpAttributeView' attribute \"{0}\" not found",
                      DebatchedCallOpAttributeView::availableTilesAttrName());
    return static_cast<DebatchedCallOpData::ValueType>(
            mlir::cast<mlir::IntegerAttr>(callOp->getAttr(DebatchedCallOpAttributeView::availableTilesAttrName()))
                    .getValue()
                    .getSExtValue());
}

bool DebatchedCallOpAttributeView::hasReorderingAttr(mlir::func::CallOp callOp) {
    return callOp->hasAttr(DebatchedCallOpAttributeView::reorderingAttrName());
}

void DebatchedCallOpAttributeView::setReorderingAttr(mlir::func::CallOp callOp) {
    auto newAttr = getIntAttr(callOp->getContext(), 1);

    VPUX_THROW_UNLESS(newAttr != nullptr, "Failed to create new 'DebatchedCallOpAttributeView' attribute \"{0}\"",
                      DebatchedCallOpAttributeView::reorderingAttrName());

    callOp->setAttr(DebatchedCallOpAttributeView::reorderingAttrName(), newAttr);
}

DebatchCoeffDescription DebatchCoeffDescription::createFromString(std::string_view descr) {
    VPUX_THROW_WHEN(descr.empty(), "DebatchCoeffDescription must be created from a non-empty description");
    VPUX_THROW_WHEN(descr.size() < 2,
                    "DebatchCoeffDescription must be started with \"[\" and finished with \"]\", got: {0}", descr);
    VPUX_THROW_UNLESS(descr[0] == '[' && descr.back() == ']',
                      "DebatchCoeffDescription must be started with \"[\" and finished with \"]\", got: {0}", descr);
    auto cend = descr.cbegin();
    std::advance(cend, descr.size() - 1);
    std::vector<size_t> parsedValues;
    parsedValues.reserve(2);
    auto cbegin = descr.cbegin();
    cbegin++;
    vpux::splitRangeAndApply(cbegin, cend, '-', [&parsedValues](std::string_view item) {
        size_t result = 0;
        auto [ptr, ec] = std::from_chars(item.data(), item.data() + item.size(), result);
        VPUX_THROW_UNLESS(ec == std::errc(), "Cannot convert string: {0} to a number", item);
        parsedValues.push_back(result);
    });
    VPUX_THROW_UNLESS(parsedValues.size() == 2,
                      "DebatchCoeffDescription expects the format \"[BatchPositionInShape-DesiredBatchValue]\"");
    return DebatchCoeffDescription{Dim{parsedValues[0]}, parsedValues[1]};
}

Shape DebatchCoeffDescription::apply(ShapeRef shape) const {
    // class Shape should have the method "at()" which check a dimension value on correctness by itself
    VPUX_THROW_UNLESS(batchPositionIndex.ind() >= 0 && static_cast<size_t>(batchPositionIndex.ind()) < shape.size(),
                      "Dimension value: {0} is not apt for a shape: {1}", batchPositionIndex, shape);
    VPUX_THROW_WHEN(
            shape[batchPositionIndex] % desiredBatchValue,
            "Cannot get the desired batch value: {0} from a shape: {1}, where a batch is expected on the position: "
            "{2}. A division operation on those number produces the remnant: {3} which otherwise would be adandoned",
            desiredBatchValue, shape, batchPositionIndex, shape[batchPositionIndex] % desiredBatchValue);
    Shape retShape{shape};
    retShape[batchPositionIndex] = desiredBatchValue;
    return retShape;
}

std::string DebatchCoeffDescription::to_string() const {
    return llvm::formatv("[{0}-{1}]", batchPositionIndex.ind(), desiredBatchValue).str();
}

std::optional<DebatchCoefficients> DebatchCoefficients::create(std::string_view tensorsCoeffFormatted) {
    if (tensorsCoeffFormatted.empty()) {
        return {};
    }
    DebatchCoefficients ret;
    vpux::splitRangeAndApply(tensorsCoeffFormatted.cbegin(), tensorsCoeffFormatted.cend(), ',',
                             [&ret](std::string_view item) {
                                 auto nameCoeffDivPos = item.find(":");
                                 std::string nodeName;
                                 std::string_view::iterator itBegin = item.begin();
                                 std::string_view::iterator itEnd = item.end();
                                 if (nameCoeffDivPos != std::string::npos) {
                                     nodeName = std::string(item.begin(), item.begin() + nameCoeffDivPos);
                                     itBegin += (nameCoeffDivPos + 1);
                                 }
                                 std::string_view coeffPairStr{&(*itBegin), static_cast<size_t>(itEnd - itBegin)};
                                 ret.orderedInputCoefficients.emplace(
                                         std::move(nodeName), DebatchCoeffDescription::createFromString(coeffPairStr));
                             });

    VPUX_THROW_UNLESS(ret.orderedInputCoefficients.size(),
                      "DebatchCoefficients must contain something once coefficient string: \"{0}\" has been parsed",
                      tensorsCoeffFormatted);
    auto isEmptyNodeName = [](const auto& node) {
        return node.first.empty();
    };
    size_t nonNamedNodeCount =
            std::count_if(ret.orderedInputCoefficients.begin(), ret.orderedInputCoefficients.end(), isEmptyNodeName);
    size_t namedNodeCount = std::count_if(ret.orderedInputCoefficients.begin(), ret.orderedInputCoefficients.end(),
                                          std::not_fn(isEmptyNodeName));
    VPUX_THROW_WHEN(namedNodeCount != 0 && nonNamedNodeCount != 0,
                    "DebatchCoefficients encloses both named: {0} and unnamed: {1} node types, "
                    "which is forbidden. Please specify absent names in nodes, or don't give names to all nodes at all",
                    namedNodeCount, nonNamedNodeCount);
    return ret;
}
Shape DebatchCoefficients::applyDefault(ShapeRef shape) {
    return DebatchCoeffDescription{}.apply(shape);
}

Shape DebatchCoefficients::apply(ShapeRef shape, size_t index) const {
    VPUX_THROW_UNLESS(index < orderedInputCoefficients.size(),
                      "DebatchCoefficients requested index: {0} must be lesser than tesnor description count: {1}",
                      index, orderedInputCoefficients.size());
    auto [nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt] = orderedInputCoefficients.equal_range("");
    size_t nodesCount = std::distance(nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt);
    VPUX_THROW_WHEN(nodesCount != orderedInputCoefficients.size(),
                    "DebatchCoefficients must encompass only named nodes only or unnamed");
    std::advance(nodeCoeffDescriptionBeginIt, index);
    return nodeCoeffDescriptionBeginIt->second.apply(shape);
}

Shape DebatchCoefficients::apply(ShapeRef shape, const std::string& nodeName) const {
    VPUX_THROW_WHEN(nodeName.empty(),
                    "DebatchCoefficients cannot apply shape transformation on empty node name. Use index instead");
    auto [nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt] = orderedInputCoefficients.equal_range(nodeName);
    auto nodesCount = std::distance(nodeCoeffDescriptionBeginIt, nodeCoeffDescriptionEndIt);
    VPUX_THROW_WHEN(nodesCount == 0, "DebatchCoefficients must contain node name: {0}", nodeName);
    VPUX_THROW_WHEN(nodesCount != 1,
                    "DebatchCoefficients has multiple nodes with the name: {0}, please use index instead", nodeName);
    return nodeCoeffDescriptionBeginIt->second.apply(shape);
}

size_t DebatchCoefficients::size() const {
    return orderedInputCoefficients.size();
}
