//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/compiler.hpp>
#include "openvino/core/visibility.hpp"

OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS void CreateNPUCompiler(std::shared_ptr<vpux::ICompiler>& obj) {
    obj = std::make_shared<vpux::CompilerImpl>();
}
