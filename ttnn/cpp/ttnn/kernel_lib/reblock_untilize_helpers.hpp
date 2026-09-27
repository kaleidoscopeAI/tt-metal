// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "internal/mod_div_lib.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // OutputCBLayout

namespace compute_kernel_lib {

namespace reblock_untilize_config {

/**
 * InitAndUninit brackets each call. For a chain, use InitOnly for the first,
 * Neither for intermediate calls, and UninitOnly for the last.
 */
enum class InitUninitMode : uint8_t { InitAndUninit, InitOnly, UninitOnly, Neither };

/**
 * Reconfigure unpack srcA for interm, the packer for out, both, or neither.
 * Independent of InitUninitMode; skipped reconfiguration is the caller's responsibility.
 */
enum class ReconfigureRegisterDatatypeMode : uint8_t {
    NoReconfigure,
    UnpackReconfigure,
    PackReconfigure,
    UnpackAndPackReconfigure
};

}  // namespace reblock_untilize_config

/**
 * Convert a whole SubblockMajor tiled block to untilized row-major output.
 * TileRowMajor input should use the standard untilize helper instead.
 *
 * Consumes interm and publishes out one subblock row-group at a time.
 * out_block_w must equal out_subblock_w * num_subblocks_w;
 * out_subblock_num_tiles must equal out_subblock_h * out_subblock_w.
 * in0_num_subblocks counts row-groups along M. All dimensions are in tiles.
 *
 * The caller performs compute_kernel_hw_startup once. By default this helper
 * reconfigures formats and initializes/uninitializes the entire call; partial
 * lifecycle modes allow consecutive calls to share initialization.
 */
template <
    uint32_t out_subblock_w,
    uint32_t out_block_w,
    reblock_untilize_config::InitUninitMode init_uninit_mode = reblock_untilize_config::InitUninitMode::InitAndUninit,
    reblock_untilize_config::ReconfigureRegisterDatatypeMode reconfig_mode =
        reblock_untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
    OutputCBLayout layout = OutputCBLayout::SubblockMajor,
    typename Buf = ::CircularBuffer>
inline void reblock_and_untilize(
    uint32_t in0_num_subblocks,
    uint32_t num_subblocks_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_h,
    Buf& interm_buf,
    Buf& out_buf);

}  // namespace compute_kernel_lib

#include "reblock_untilize_helpers.inl"
