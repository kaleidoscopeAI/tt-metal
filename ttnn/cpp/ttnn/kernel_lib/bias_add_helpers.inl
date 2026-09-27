// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"

/**
 * Implementation of add_bias_bcast_rows; include via bias_add_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    BiasBroadcast broadcast,
    OutputCBLayout tile_order,
    typename PostBiasFn,
    typename Activation,
    bool in_place,
    typename Buf>
ALWI void add_bias_bcast_rows(
    Buf& partials_buf,
    Buf& bias_buf,
    Buf& out_buf,
    BiasAddShape shape,
    PostBiasFn post_bias,
    uint32_t bias_offset) {

    static_assert(
        !in_place || tile_order == OutputCBLayout::TileRowMajor,
        "add_bias_bcast_rows in_place is supported only for TileRowMajor layout.");

    const uint32_t partials_cb_id = buf_id(partials_buf);
    const uint32_t bias_cb_id = buf_id(bias_buf);
    const uint32_t out_cb_id = buf_id(out_buf);

    const uint32_t in0_num_subblocks = shape.in0_num_subblocks;
    const uint32_t in1_num_subblocks = shape.in1_num_subblocks;
    const uint32_t out_subblock_h = shape.out_subblock_h;
    const uint32_t out_subblock_w = shape.out_subblock_w;
    uint32_t out_row_width = shape.out_row_width;

    const uint32_t out_num_tiles = out_subblock_h * out_subblock_w;

    ASSERT(in0_num_subblocks > 0);
    ASSERT(in1_num_subblocks > 0);
    ASSERT(out_subblock_h > 0);
    ASSERT(out_subblock_w > 0);
    ASSERT(out_num_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT);

    reconfig_data_format_srca(partials_cb_id);
    reconfig_data_format_srcb(bias_cb_id);
    pack_reconfig_data_format(out_cb_id);
    if constexpr (broadcast == BiasBroadcast::RowBroadcast) {
        add_bcast_rows_init(partials_cb_id, bias_cb_id);
    } else {
        add_init(partials_cb_id, bias_cb_id);
    }

    if constexpr (tile_order == OutputCBLayout::TileRowMajor) {
        // Read and write each subblock at its tile-row-major offsets.
        if (out_row_width == 0) {
            out_row_width = out_subblock_w * in1_num_subblocks;
        }
        const uint32_t row_group_tiles = out_subblock_h * out_row_width;

        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            // The one-block buffer is already fronted. Wait once and rewrite in place;
            // publication occurs after all row groups are complete.
            if constexpr (in_place) {
                if (in0_subblock == 0) {
                    partials_buf.wait_front(in0_num_subblocks * row_group_tiles);
                }
            } else {
                partials_buf.wait_front(row_group_tiles);
                out_buf.reserve_back(row_group_tiles);
            }

            // In-place addressing includes the row-group base; FIFO addressing advances it via pop/push.
            const uint32_t group_base = in_place ? in0_subblock * row_group_tiles : 0;

            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                const uint32_t col_base = group_base + in1_subblock * out_subblock_w;
                const uint32_t bias_col_base = bias_offset + in1_subblock * out_subblock_w;

                tile_regs_acquire();
                {
                    uint32_t dst_idx = 0;
                    for (uint32_t r = 0; r < out_subblock_h; r++) {
                        const uint32_t partial_row_pos = r * out_row_width + col_base;
                        for (uint32_t c = 0; c < out_subblock_w; c++) {
                            if constexpr (broadcast == BiasBroadcast::RowBroadcast) {
                                add_tiles_bcast_rows(
                                    partials_cb_id,
                                    bias_cb_id,
                                    partial_row_pos + c,
                                    bias_col_base + c,
                                    dst_idx);
                            } else {
                                add_tiles(
                                    partials_cb_id,
                                    bias_cb_id,
                                    partial_row_pos + c,
                                    bias_col_base + c,
                                    dst_idx);
                            }
                            dst_idx++;
                        }
                    }
                }

                post_bias(out_num_tiles);
                tile_regs_commit();
                if constexpr (Activation::activation != KernelActivation::NONE) {
                    apply_activation_from_pack<
                        Activation::activation,
                        Activation::param0,
                        Activation::param1,
                        Activation::param2>(out_num_tiles);
                } else {
                    tile_regs_wait();
                }

                pack_subblock_row_strided(
                    0, out_cb_id, col_base, out_row_width, out_subblock_h, out_subblock_w);
                tile_regs_release();
            }

            if constexpr (!in_place) {
                partials_buf.pop_front(row_group_tiles);
                out_buf.push_back(row_group_tiles);
            }
        }

        // Republish after the in-place write to synchronize PACK with downstream UNPACK.
        // The original matmul credit alone would allow the consumer to read before bias
        // packing finishes. A full CB cycle preserves the physical buffer addresses.
        if constexpr (in_place) {
            partials_buf.pop_front(in0_num_subblocks * row_group_tiles);
            partials_buf.push_back(in0_num_subblocks * row_group_tiles);
        }
    } else {
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                partials_buf.wait_front(out_num_tiles);
                tile_regs_acquire();

                for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
                    uint32_t bias_tile_idx = bias_offset + in1_index_subblock_offset;
                    for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
                        if constexpr (broadcast == BiasBroadcast::RowBroadcast) {
                            add_tiles_bcast_rows(partials_cb_id, bias_cb_id, i, bias_tile_idx, i);
                        } else {
                            add_tiles(partials_cb_id, bias_cb_id, i, bias_tile_idx, i);
                        }
                        bias_tile_idx++;
                    }
                }

                post_bias(out_num_tiles);

                tile_regs_commit();
                partials_buf.pop_front(out_num_tiles);

                out_buf.reserve_back(out_num_tiles);
                if constexpr (Activation::activation != KernelActivation::NONE) {
                    apply_activation_from_pack<
                        Activation::activation,
                        Activation::param0,
                        Activation::param1,
                        Activation::param2>(out_num_tiles);
                } else {
                    tile_regs_wait();
                }
                for (uint32_t i = 0; i < out_num_tiles; i++) {
                    pack_tile(i, out_cb_id);
                }
                tile_regs_release();
                out_buf.push_back(out_num_tiles);

                in1_index_subblock_offset += out_subblock_w;
            }
        }
    }
}

}  // namespace compute_kernel_lib
