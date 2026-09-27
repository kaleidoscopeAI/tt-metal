// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Uniform CB IDs for LLK calls accepting either CircularBuffer or DataflowBuffer.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"

namespace compute_kernel_lib {

ALWI uint32_t buf_id(const ::CircularBuffer& cb) { return cb.get_cb_id(); }

ALWI uint32_t buf_id(const ::DataflowBuffer& dfb) { return dfb.get_id(); }

}  // namespace compute_kernel_lib
