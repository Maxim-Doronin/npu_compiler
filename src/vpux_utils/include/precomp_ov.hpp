//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/dimension.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/runtime/icompiled_model.hpp>
#include <openvino/runtime/iinfer_request.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/iplugin.hpp>
#include <openvino/runtime/isync_infer_request.hpp>
#include <openvino/runtime/ivariable_state.hpp>
#include <openvino/runtime/profiling_info.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/shared_buffer.hpp>
