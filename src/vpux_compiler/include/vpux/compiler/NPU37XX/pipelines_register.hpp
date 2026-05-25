//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/pipelines_register.hpp"

namespace vpux {

//
// PipelineRegistry37XX
//

class PipelineRegistry37XX final : public IPipelineRegistry {
public:
    void registerPipelines() override;
};

}  // namespace vpux
