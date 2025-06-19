#include "vpux/compiler/dialect/const/attributes/attributes.hpp"

using namespace vpux;

bool Const::canChangeShape(Const::TransformAttrInterface attr) {
    return llvm::isa<Const::PadWithZeroAttr, Const::BroadcastAttr, Const::ReshapeAttr, Const::SubViewAttr,
                     Const::AffineReshapeAttr>(attr);
}
