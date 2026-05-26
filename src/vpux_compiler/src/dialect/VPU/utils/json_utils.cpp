//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <fstream>
#include <iomanip>

namespace vpux::VPU {

llvm::Expected<llvm::json::Value> readManualStrategyJSON(StringRef fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for input strategy json was not provided");

    std::ifstream i(fileName.str());
    VPUX_THROW_UNLESS(i.good(), "File with manual strategy not opened correctly");
    std::stringstream input{};
    input << i.rdbuf();
    return llvm::json::parse(input.str());
}

void writeManualStrategyJSON(StringRef fileName, const llvm::json::Value& json) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for output strategy json was not provided");

    std::ofstream os(fileName.str());
    VPUX_THROW_UNLESS(os.good(), "File with manual strategy not created correctly");
    os << llvm::formatv("{0:2}", json).str() << std::endl;

    return;
}

llvm::json::Value convertAttrToJSON(mlir::Attribute attr) {
    if (mlir::isa_and_nonnull<vpux::VPU::MultiClusterStrategyAttr>(attr)) {
        return stringifyMultiClusterStrategy(mlir::cast<vpux::VPU::MultiClusterStrategyAttr>(attr).getValue());
    } else if (mlir::isa_and_nonnull<mlir::ArrayAttr>(attr)) {
        auto values = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(attr)));
        VPUX_THROW_UNLESS(values.size() <= 5 && values.size() >= 1,
                          "Shape has invalid dimensions than expected ([1-5]), got '{0}'", values.size());
        llvm::json::Object tilingStrategy{};
        if (values.size() == 5) {
            tilingStrategy["G"] = values[DimsGroups5D::Act::G];
            tilingStrategy["N"] = values[DimsGroups5D::Act::N];
            tilingStrategy["C"] = values[DimsGroups5D::Act::C];
            tilingStrategy["H"] = values[DimsGroups5D::Act::H];
            tilingStrategy["W"] = values[DimsGroups5D::Act::W];
        } else if (values.size() == 4) {
            tilingStrategy["N"] = values[Dims4D::Act::N];
            tilingStrategy["C"] = values[Dims4D::Act::C];
            tilingStrategy["H"] = values[Dims4D::Act::H];
            tilingStrategy["W"] = values[Dims4D::Act::W];
        } else if (values.size() == 3) {
            tilingStrategy["B"] = values[Dims3D::Output::B];
            tilingStrategy["H"] = values[Dims3D::Output::H];
            tilingStrategy["OC"] = values[Dims3D::Output::OC];
        } else if (values.size() == 2) {
            tilingStrategy["H"] = values[Dim(0)];
            tilingStrategy["W"] = values[Dim(1)];
        } else {
            tilingStrategy["N"] = values[Dim(0)];
        }

        return tilingStrategy;
    } else if (mlir::isa_and_nonnull<VPU::VFScenarioAttr>(attr)) {
        return stringifyVFScenario(mlir::cast<VPU::VFScenarioAttr>(attr).getValue());
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", attr);
}

mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, const llvm::json::Value& newAttrVal) {
    if (mlir::isa<vpux::VPU::MultiClusterStrategyAttr>(oldAttr)) {
        return VPU::MultiClusterStrategyAttr::get(
                oldAttr.getContext(), symbolizeMultiClusterStrategy(newAttrVal.getAsString().value()).value());
    } else if (mlir::isa<mlir::ArrayAttr>(oldAttr)) {
        Shape newShape(4, 1);
        VPUX_THROW_WHEN(newAttrVal.getAsObject() == nullptr, "Invalid JSON representation of array attribute");
        llvm::json::Object dimenstions = *newAttrVal.getAsObject();
        newShape[Dims4D::Act::N] = static_cast<int64_t>(dimenstions["N"].getAsUINT64().value());
        newShape[Dims4D::Act::C] = static_cast<int64_t>(dimenstions["C"].getAsUINT64().value());
        newShape[Dims4D::Act::H] = static_cast<int64_t>(dimenstions["H"].getAsUINT64().value());
        newShape[Dims4D::Act::W] = static_cast<int64_t>(dimenstions["W"].getAsUINT64().value());
        return getIntArrayAttr(oldAttr.getContext(), newShape);
    } else if (mlir::isa<VPU::VFScenarioAttr>(oldAttr)) {
        auto defaultScenario = VPU::VFScenarioAttr::get(oldAttr.getContext(), VPU::VFScenario::MINIMAL);
        if (!newAttrVal.getAsString().has_value()) {
            // VF scenario is not configured, return default value
            return defaultScenario;
        }
        auto scenario = symbolizeVFScenario(newAttrVal.getAsString().value());
        return scenario.has_value() ? VPU::VFScenarioAttr::get(oldAttr.getContext(), scenario.value())
                                    : defaultScenario;
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", oldAttr);
}

std::optional<llvm::json::Value> getPreviousAttributeValue(const llvm::json::Value& json, const std::string& opName,
                                                           StringRef attribute) {
    auto jsonAsObject = json.getAsObject();
    if (jsonAsObject == nullptr) {
        return std::nullopt;
    }

    if (jsonAsObject->find(opName) == jsonAsObject->end()) {
        return std::nullopt;
    }

    auto jsonOpsToAttributes = *jsonAsObject;
    if (jsonOpsToAttributes[opName].getAsObject() == nullptr) {
        return std::nullopt;
    }

    auto jsonAttrsToLayerAttribute = *jsonOpsToAttributes[opName].getAsObject();
    if (jsonAttrsToLayerAttribute.find(attribute.str()) != jsonAttrsToLayerAttribute.end()) {
        return jsonAttrsToLayerAttribute[attribute.str()];
    }

    return std::nullopt;
}

std::string getOpHash(mlir::Operation* op) {
    if (op == nullptr) {
        return "Null";
    }
    std::string opLocation;
    std::hash<std::string> hasher;
    llvm::raw_string_ostream oLocation(opLocation);

    op->getLoc().print(oLocation);

    std::stringstream hexHash;
    hexHash << "0x" << std::setw(4) << std::setfill('0') << std::hex << hasher(opLocation);
    return hexHash.str();
}

void createStrategyJSONFromOperations(llvm::json::Value& json,
                                      llvm::MapVector<mlir::Location, mlir::Operation*>& operations,
                                      DenseMap<StringRef, StringRef>& strategyAttributes) {
    llvm::json::Object opsToStrategies{};
    for (auto& op : operations) {
        auto opName = vpux::stringifyPrimaryLocation(op.first);
        auto parentVFOp = op.second->getParentOfType<VPU::VerticalFusionOp>();

        // retrieve related attributes and save in JSON
        llvm::json::Object layerAttributes{};
        bool updatedVFTiling = false;
        for (const auto& attribute : strategyAttributes) {
            llvm::json::Value attributeValue(attribute.second);
            if (op.second->hasAttr(attribute.first)) {
                // Get value present in IR
                attributeValue = convertAttrToJSON(op.second->getAttr(attribute.first));
            } else {
                // If opName is found, assign the value read from previous runs
                attributeValue = getPreviousAttributeValue(json, opName, attribute.first).value_or(attributeValue);
            }

            if (attribute.first == vpux::layerTypeName) {
                std::string layerTypeName;
                llvm::raw_string_ostream oLayerTypeName(layerTypeName);
                op.second->getName().print(oLayerTypeName);
                attributeValue = std::move(layerTypeName);
            }
            if (parentVFOp != nullptr) {
                // If such layer is encountered, we can no longer find the tilingStrategy or VF scenario in NCEOp
                if (attribute.first == vpux::tilingStrategy) {
                    auto prevAttrValue = getPreviousAttributeValue(json, opName, attribute.first).value_or("");
                    auto currentAttrValue = convertAttrToJSON(parentVFOp->getAttr(attribute.first));
                    attributeValue = currentAttrValue;
                    updatedVFTiling = prevAttrValue != currentAttrValue;
                } else if (attribute.first == vpux::verticalFusionScenario && parentVFOp.getScenarioAttr() != nullptr) {
                    auto currentAttrValue = convertAttrToJSON(parentVFOp.getScenarioAttr());
                    attributeValue = currentAttrValue;
                }
            }

            layerAttributes[attribute.first.str()] = std::move(attributeValue);
        }
        layerAttributes[vpux::verticalFusion] = parentVFOp != nullptr ? "True" : "False";
        layerAttributes[vpux::verticalFusionHash] = getOpHash(parentVFOp);
        layerAttributes[vpux::updatedVFTiling] = updatedVFTiling ? "True" : "False";
        opsToStrategies[std::move(opName)] = llvm::json::Value(std::move(layerAttributes));
    }
    json = llvm::json::Value(std::move(opsToStrategies));
}

void overwriteManualVFStrategy(llvm::json::Value& manualStrategyValue,
                               llvm::MapVector<mlir::Location, mlir::Operation*>& operations) {
    llvm::json::Object manualStrategyObject = *manualStrategyValue.getAsObject();
    std::unordered_map<std::string, llvm::SmallVector<mlir::Operation*>> vfHashToOps;
    std::unordered_map<std::string, mlir::Attribute> vfHashToTilingAttr;
    std::unordered_map<std::string, mlir::Attribute> vfHashToVFScenarioAttr;
    for (auto& item : manualStrategyObject) {
        // skip if it's not found in the map
        auto opLoc = item.first.str();
        auto opIter = llvm::find_if(operations, [&](auto& op) {
            return vpux::stringifyPrimaryLocation(op.first) == opLoc;
        });
        if (opIter == operations.end()) {
            continue;
        }
        auto op = opIter->second;
        const auto getOpPointer = [](auto& op) -> mlir::Operation* {
            return &op;
        };

        // skip if vertical fusion is disabled or not configured
        auto currOpStrategyObject = item.second.getAsObject();
        auto iter = currOpStrategyObject->find(vpux::verticalFusion);
        auto isVFDisabled =
                iter != currOpStrategyObject->end() && iter->second.getAsString().value_or("True") == "False";
        if (isVFDisabled) {
            continue;
        }

        // skip if no tiling strategy found
        iter = currOpStrategyObject->find(vpux::tilingStrategy);
        if (iter == currOpStrategyObject->end()) {
            continue;
        }
        auto dummyAttr = getIntArrayAttr(op->getContext(), Shape(4));
        auto manualTilingAttribute = convertJSONToAttr(dummyAttr, iter->second);

        // skip if no vertical fusion hash found
        iter = currOpStrategyObject->find(vpux::verticalFusionHash);
        if (iter == currOpStrategyObject->end()) {
            continue;
        }
        auto opHash = iter->second.getAsString().value_or("").str();

        // skip if no vertical fusion scenario found
        iter = currOpStrategyObject->find(vpux::verticalFusionScenario);
        if (iter == currOpStrategyObject->end()) {
            continue;
        }
        auto dummyVFScheduleTypeAttr = VPU::VFScenarioAttr::get(op->getContext(), VPU::VFScenario::MINIMAL);
        auto manualVFScheduleTypeAttr = convertJSONToAttr(dummyVFScheduleTypeAttr, iter->second);

        // Only try to manually overwrite VF ops before MergeVF pass
        auto parentVerticalFusionOp = op->getParentOfType<VPU::VerticalFusionOp>();
        if (parentVerticalFusionOp == nullptr) {
            continue;
        }
        auto operations = to_small_vector(parentVerticalFusionOp.getBody()->without_terminator() |
                                          transformed(getOpPointer) | filtered([](mlir::Operation* op) {
                                              return mlir::isa_and_nonnull<VPU::VerticalFusionOpInterface>(op);
                                          }));
        if (operations.size() != 1) {
            continue;
        }
        auto existedTilingIter = vfHashToTilingAttr.find(opHash);
        if (existedTilingIter == vfHashToTilingAttr.end()) {
            vfHashToTilingAttr.emplace(opHash, manualTilingAttribute);
        } else if (existedTilingIter->second != manualTilingAttribute) {
            Logger::global().warning("Got mismatched tiling strategies for VFRegion: {0}", opHash);
        }

        auto existedScenarioIter = vfHashToVFScenarioAttr.find(opHash);
        if (existedScenarioIter == vfHashToVFScenarioAttr.end()) {
            vfHashToVFScenarioAttr.emplace(opHash, manualVFScheduleTypeAttr);
        } else if (existedScenarioIter->second != manualVFScheduleTypeAttr) {
            Logger::global().warning("Got mismatched VF scheduling types for VFRegion: {0}", opHash);
        }

        vfHashToOps[opHash].push_back(parentVerticalFusionOp);
    }

    auto isViewOpFusable = [](mlir::Operation* op) {
        if (auto tilingViewOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(op)) {
            return tilingViewOp.isVFSupported();
        }
        return false;
    };

    for (auto& item : vfHashToOps) {
        auto& ops = item.second;
        llvm::sort(ops, [](mlir::Operation* a, mlir::Operation* b) {
            return a->isBeforeInBlock(b);
        });

        while (ops.size() > 1) {
            auto vfOp = ops.back();
            ops.pop_back();
            // get next VF to be merged
            auto operands = to_small_vector(vfOp->getOperands() | filtered([&](auto operand) {
                                                auto parentOp = findParent(operand);
                                                return parentOp != nullptr && llvm::find(ops, parentOp) != ops.end();
                                            }));
            if (operands.empty()) {
                Logger::global().warning("No parent VF found for merging with current VF {0}",
                                         vpux::stringifyPrimaryLocation(vfOp->getLoc()));
                break;
            }
            llvm::sort(operands, [](auto& operandA, auto& operandB) {
                return findParent(operandA)->isBeforeInBlock(findParent(operandB));
            });

            auto operand = operands.back();
            auto nearestParent = findParent(operand);
            ops.erase(llvm::find(ops, nearestParent));

            auto parent = operand.getDefiningOp();
            auto fusedOp = vfOp;
            while (parent != nullptr && (isViewOpFusable(parent) || parent == nearestParent)) {
                mlir::OpBuilder builder(fusedOp);
                builder.setInsertionPointAfter(fusedOp);
                auto newFusedOp =
                        VPU::fuseOpsInBlock(builder, mlir::cast<VPU::VerticalFusionOp>(fusedOp), parent,
                                            mlir::cast<mlir::ArrayAttr>(vfHashToTilingAttr[item.first]), true);
                newFusedOp.setScenarioAttr(mlir::cast<VPU::VFScenarioAttr>(vfHashToVFScenarioAttr[item.first]));
                parent->replaceUsesWithIf(newFusedOp, [&](mlir::OpOperand& operand) {
                    return operand.getOwner() == fusedOp;
                });

                fusedOp->replaceAllUsesWith(newFusedOp);
                fusedOp->erase();
                fusedOp = newFusedOp.getOperation();
                auto nextParent = parent == nearestParent ? nullptr : parent->getOperand(0).getDefiningOp();
                if (parent->use_empty()) {
                    parent->erase();
                }
                parent = nextParent;
            }
            ops.push_back(fusedOp);
        }
    }
}

void overwriteManualStrategy(llvm::json::Value& manualStrategyValue,
                             llvm::MapVector<mlir::Location, mlir::Operation*>& operations) {
    DenseMap<mlir::Operation*, std::pair<llvm::json::Value, bool>> vfOpVisited;
    SmallVector<StringRef> allowedValues = {layerTypeName, verticalFusionHash, verticalFusion, updatedVFTiling,
                                            verticalFusionScenario};
    auto isAllowedAttr = [&allowedValues](std::string& currentType) {
        return std::find(allowedValues.begin(), allowedValues.end(), currentType) != allowedValues.end();
    };

    for (auto& op : operations) {
        const auto opName = vpux::stringifyPrimaryLocation(op.first);

        VPUX_THROW_WHEN(manualStrategyValue.getAsObject() == nullptr,
                        "Manual strategy JSON should represent JSON object");
        llvm::json::Object manualStrategyObject = *manualStrategyValue.getAsObject();
        // check if manual strategy for layer exists
        if (manualStrategyObject.find(opName) == manualStrategyObject.end()) {
            continue;
        }

        auto parentVerticalFusionOp = op.second->getParentOfType<VPU::VerticalFusionOp>();
        VPUX_THROW_WHEN(manualStrategyObject[opName].getAsObject() == nullptr,
                        "JSON value for operation should represent JSON object");
        auto currOpStrategyObject = *manualStrategyObject[opName].getAsObject();
        for (auto it = currOpStrategyObject.begin(); it != currOpStrategyObject.end(); ++it) {
            // replace attributes of the operation (skip NONE) using it->second
            if (!(it->second.kind() == llvm::json::Value::Kind::String) ||
                (it->second.kind() == llvm::json::Value::Kind::String &&
                 it->second.getAsString().value() != defaultNoValue.str())) {
                if (it->first.str() == multiClusterStrategy) {
                    // Clustering is set as placeholder to be replaced with provided strategy
                    auto dummyAttr = VPU::MultiClusterStrategyAttr::get(op.second->getContext(),
                                                                        VPU::MultiClusterStrategy::Clustering);
                    auto manualAttribute = convertJSONToAttr(dummyAttr, it->second);

                    if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op.second)) {
                        auto manualStratAttr = mlir::cast<vpux::VPU::MultiClusterStrategyAttr>(manualAttribute);
                        clusteredOp.setMultiClusterStrategy(manualStratAttr.getValue());
                    }

                } else if (it->first.str() == tilingStrategy) {
                    // tiling case, where strategy selection and IR modification occurs in a single pass
                    // TODO: remove "else" when tiling strategy will be abstracted into strategy pass
                    auto dummyAttr = getIntArrayAttr(op.second->getContext(), Shape(4));
                    auto manualAttribute = convertJSONToAttr(dummyAttr, it->second);
                    if (parentVerticalFusionOp != nullptr) {
                        auto [itr, inserted] = vfOpVisited.try_emplace(parentVerticalFusionOp, it->second, true);
                        if (!inserted) {
                            // We visited this VFSubgraph before, in such case check if op visited before had same
                            // tiling strategy as current one
                            VPUX_THROW_WHEN(itr->second.first != it->second,
                                            "Got mismatched tiling strategies for VFRegion: {0}",
                                            getOpHash(parentVerticalFusionOp));
                            continue;
                        }
                        parentVerticalFusionOp.getOperation()->setAttr(tilingStrategy, manualAttribute);
                    } else {
                        Logger::global().warning("Overwrite manual strategy {0} for op. opName {1}, opLoc {2}",
                                                 manualAttribute, opName, op.second->getLoc());
                        op.second->setAttr(tilingStrategy, manualAttribute);
                    }
                } else if (it->first.str() == verticalFusion) {
                    // Disable vertical fusion if related attribute is set to False
                    if (parentVerticalFusionOp == nullptr) {
                        continue;
                    }
                    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
                        return &op;
                    };
                    auto operations =
                            to_small_vector(parentVerticalFusionOp.getBody()->without_terminator() |
                                            transformed(getOpPointer) | filtered([](mlir::Operation* op) {
                                                return mlir::isa_and_nonnull<VPU::VerticalFusionOpInterface>(op);
                                            }));
                    if (operations.size() > 1) {
                        continue;
                    }

                    auto iter = currOpStrategyObject.find(vpux::verticalFusion);
                    auto isVerticalFusionDisabled = iter != currOpStrategyObject.end() &&
                                                    iter->second.getAsString().value_or("True") == "False";
                    if (isVerticalFusionDisabled) {
                        parentVerticalFusionOp.setIsManualConfigured(true);
                        continue;
                    }
                } else {
                    auto attrType = it->first.str();
                    VPUX_THROW_WHEN(!isAllowedAttr(attrType), "Unsupported Attribute '{0}'", it->first.str());
                }
            } else {
                auto attrType = it->first.str();
                if (op.second->hasAttr(attrType) && !isAllowedAttr(attrType)) {
                    // currently no default value, to disable multiclustering remove the attribute
                    op.second->removeAttr(attrType);
                }
            }
        }
    }
    overwriteManualVFStrategy(manualStrategyValue, operations);
}

void updateAttributeValue(llvm::json::Value& json, const std::string& opName, StringRef attribute,
                          llvm::json::Value&& newValue) {
    auto jsonAsObject = json.getAsObject();
    if (jsonAsObject == nullptr) {
        return;
    }

    if (jsonAsObject->find(opName) == jsonAsObject->end()) {
        return;
    }

    auto jsonOpsToAttributes = jsonAsObject->operator[](opName).getAsObject();
    if (jsonOpsToAttributes == nullptr) {
        return;
    }

    auto jsonAttrsToLayerAttribute = jsonOpsToAttributes;
    if (jsonAttrsToLayerAttribute->find(attribute.str()) == jsonAttrsToLayerAttribute->end()) {
        return;
    }

    (*jsonAttrsToLayerAttribute)[attribute.str()] = std::move(newValue);
}

// find tilingStrategy JSON value in JSON file via key 'opName' and update this JSON value according to operation's
// tilingStrategy attribute
void updateTilingStrategyInJSONForOperations(llvm::json::Value& json,
                                             llvm::MapVector<mlir::Location, mlir::Operation*>& operations) {
    for (auto& op : operations) {
        auto opName = vpux::stringifyPrimaryLocation(op.first);

        // If opName is found, retrieve previous attribute from json
        auto prevAttributeValue = getPreviousAttributeValue(json, opName, tilingStrategy);
        if (!prevAttributeValue.has_value()) {
            continue;
        }

        llvm::json::Value currAttributeValue(defaultNoValue);
        if (op.second->hasAttr(tilingStrategy)) {
            // Get value present in IR
            currAttributeValue = convertAttrToJSON(op.second->getAttr(tilingStrategy));
        }

        // Update tiling strategy attribute in json
        if (prevAttributeValue.value() != currAttributeValue) {
            Logger::global().warning("Update tiling strategy in JSON. opName {0}, opLoc {1}, from {2} to {3}", opName,
                                     op.second->getLoc(), prevAttributeValue.value(), currAttributeValue);
            updateAttributeValue(json, opName, tilingStrategy, std::move(currAttributeValue));
        }
    }
}

}  // namespace vpux::VPU
