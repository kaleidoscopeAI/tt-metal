// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // OutputCBLayout
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

namespace compute_kernel_lib {

/**
 * RowBroadcast adds one bias tile per output column across all M rows.
 * Elementwise adds bias tiles matching each output subblock.
 */
enum class BiasBroadcast { RowBroadcast, Elementwise };

/**
 * Subblock grid for bias addition. For TileRowMajor, out_row_width is the
 * output row stride; 0 derives it from out_subblock_w * in1_num_subblocks.
 */
struct BiasAddShape {
    uint32_t in0_num_subblocks;  // Subblock count along M.
    uint32_t in1_num_subblocks;  // Subblock count along N.
    uint32_t out_subblock_h;     // Subblock height in tiles.
    uint32_t out_subblock_w;     // Subblock width in tiles.
    uint32_t out_row_width = 0;  // Row stride in tiles for TileRowMajor pack; ignored for SubblockMajor.

    static constexpr BiasAddShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t out_row_width = 0) {
        return {in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, out_row_width};
    }
};

namespace bias_add_config {

// Called on the math thread after bias addition, before commit; receives the DST tile count.
struct NoPostBias {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

}  // namespace bias_add_config

/**
 * Add bias to matmul partials and pack the result, optionally applying Activation.
 * tile_order must match the upstream matmul. bias_offset shifts the bias read
 * base in tiles; the caller owns bias_buf wait/pop across invocations.
 *
 * The helper consumes partials and publishes out per subblock (SubblockMajor)
 * or per M-row-group (TileRowMajor). The caller configures the output pack format,
 * packer ReLU, and disables L1 accumulation before calling. Initialize SFPU
 * activation at kernel startup. PostBiasFn runs before commit on the math thread;
 * Activation runs on the packer thread.
 *
 * in_place requires TileRowMajor and partials_buf == out_buf. The buffer must hold
 * exactly one output block, already published, with its write pointer wrapped to
 * the block base. The helper rewrites tiles without per-row reservations, then
 * pops and republishes the whole block to synchronize downstream consumers.
 *
 * @example
 *   bias_buf.wait_front(Nt);
 *   add_bias_bcast_rows<>(partials_buf, bias_buf, out_buf, BiasAddShape::of(Mt, Nt, 1, 1));
 *   bias_buf.pop_front(Nt);
 */
template <
    BiasBroadcast broadcast = BiasBroadcast::RowBroadcast,
    OutputCBLayout tile_order = OutputCBLayout::SubblockMajor,
    typename PostBiasFn = bias_add_config::NoPostBias,
    typename Activation = NoneActivation,
    bool in_place = false,
    typename Buf = ::CircularBuffer>
ALWI void add_bias_bcast_rows(
    Buf& partials_buf,
    Buf& bias_buf,
    Buf& out_buf,
    BiasAddShape shape,
    PostBiasFn post_bias = {},
    uint32_t bias_offset = 0);

}  // namespace compute_kernel_lib

#include "bias_add_helpers.inl"
