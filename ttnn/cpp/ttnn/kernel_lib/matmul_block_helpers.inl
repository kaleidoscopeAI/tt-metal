// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp"

/**
 * Implementation of matmul_block; include via matmul_block_helpers.hpp.
 */

namespace compute_kernel_lib {

/**
 * Pack a DST subblock at row-strided tile offsets. The caller reserves the
 * row group and owns push_back; indices are relative to the current write pointer.
 */
ALWI void pack_subblock_row_strided(
    uint32_t dst_start_idx,
    uint32_t pack_target_id,
    uint32_t col_base,
    uint32_t row_stride,
    uint32_t subblock_height,
    uint32_t subblock_width) {
    for (uint32_t r = 0; r < subblock_height; r++) {
        const uint32_t row_base = r * row_stride + col_base;
        for (uint32_t c = 0; c < subblock_width; c++) {
            pack_tile<true>(dst_start_idx + r * subblock_width + c, pack_target_id, row_base + c);
        }
    }
}

/**
 * Pack a contiguous DST subblock at an offset from the current write pointer.
 * The caller owns reservation and publication.
 */
ALWI void pack_subblock_at_offset(uint32_t dst_start_idx, uint32_t pack_target_id, uint32_t abs_off, uint32_t ntiles) {
    for (uint32_t t = 0; t < ntiles; t++) {
        pack_tile<true>(dst_start_idx + t, pack_target_id, abs_off + t);
    }
}

/**
 * Gather a row-strided subblock into contiguous DST. The caller initializes copy,
 * waits for the full source row group, and pops it after use.
 */
ALWI void copy_subblock_row_strided(
    uint32_t src_cb_id, uint32_t col_base, uint32_t row_stride, uint32_t subblock_height, uint32_t subblock_width) {
    for (uint32_t r = 0; r < subblock_height; r++) {
        copy_block(src_cb_id, r * row_stride + col_base, r * subblock_width, subblock_width);
    }
}

/**
 * Copy the accumulated block to out in DST-sized chunks, preserving tile order.
 * Apply output conversion, ReLU, and SFPU activation here; consume interm and publish out.
 */
template <bool packer_l1_acc, bool apply_relu_on_last_block, typename Activation, typename Buf>
ALWI void matmul_block_finalize_copy(
    Buf& interm_buf,
    Buf& out_buf,
    uint32_t out_block_num_tiles,
    uint32_t out_subblock_num_tiles,
    uint32_t prev_srca_cb_id) {
    const uint32_t interm_cb_id = buf_id(interm_buf);
    const uint32_t out_cb_id = buf_id(out_buf);

    // srcA previously held in1. Reconfigure from that operand to avoid stale formats
    // when partials use a different data type.
    reconfig_data_format_srca(prev_srca_cb_id, interm_cb_id);
    copy_init(interm_cb_id);
    PACK((pack_reconfig_data_format(out_cb_id)));
    // Disable L1 accumulation before copying the completed sum.
    if constexpr (packer_l1_acc) {
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
    if constexpr (apply_relu_on_last_block) {
        PACK((llk_pack_relu_config(ReluConfig::zero())));
    }

    interm_buf.wait_front(out_block_num_tiles);
    out_buf.reserve_back(out_block_num_tiles);
    for (uint32_t off = 0; off < out_block_num_tiles; off += out_subblock_num_tiles) {
        tile_regs_acquire();
        copy_block(interm_cb_id, off, 0, out_subblock_num_tiles);
        tile_regs_commit();
        // SFPU activation replaces tile_regs_wait on the packer thread.
        if constexpr (Activation::activation != KernelActivation::NONE) {
            apply_activation_from_pack<
                Activation::activation,
                Activation::param0,
                Activation::param1,
                Activation::param2>(out_subblock_num_tiles);
        } else {
            tile_regs_wait();
        }
        pack_block(0, out_cb_id, out_subblock_num_tiles);
        tile_regs_release();
    }
    out_buf.push_back(out_block_num_tiles);
    interm_buf.pop_front(out_block_num_tiles);

    if constexpr (apply_relu_on_last_block) {
        PACK((llk_pack_relu_config(ReluConfig::none())));
    }
}

namespace detail {

template <
    bool transpose,
    bool packer_l1_acc,
    LastBlockTarget last_block_target,
    OutputCBLayout tile_order,
    matmul_config::InitMode init_mode,
    InputPolicy in0_policy,
    InputPolicy in1_policy,
    typename PostComputeFn,
    typename PreKBlockFn,
    typename PostKBlockFn,
    typename KBlockInnerDimFn,
    typename In0SourceFn,
    typename In1BaseOffsetFn,
    typename Activation,
    matmul_config::DataFormatReconfig reconfig,
    typename Shape,
    typename Buf>
ALWI void matmul_block_impl(
    Buf& in0_buf,
    Buf& in1_buf,
    Buf& out_buf,
    Buf& interm_buf,
    const Shape& shape,
    PostComputeFn post_compute = {},
    PreKBlockFn pre_k_block = {},
    PostKBlockFn post_k_block = {},
    KBlockInnerDimFn k_block_inner_dim = {},
    In0SourceFn in0_source_fn = {},
    In1BaseOffsetFn in1_base_offset_fn = {}) {
    static_assert(
        in0_policy != InputPolicy::NoWaitNoPop,
        "InputPolicy::NoWaitNoPop is only valid for in1 — use WaitAndPopPerKBlock or "
        "WaitAndRetainOnLastBlock for in0.");

    // Publish completed subblocks or row groups when possible. L1 accumulation
    // for Out targets requires a finalize unless it can write directly to out.
    constexpr bool target_is_interm = (last_block_target == LastBlockTarget::Interm);
    constexpr bool apply_relu_on_last_block = (last_block_target == LastBlockTarget::OutWithRelu);

    // L1 accumulation retains partials in place. Software reload packs the last
    // K block directly to out unless a downstream operation needs Interm. Re-reserving
    // the same full Interm buffer before popping its input would deadlock.
    constexpr bool pack_last_to_interm = target_is_interm || packer_l1_acc;

    // Apply activation only after the full K reduction, never to partial sums.
    constexpr bool activate_on_last_pack = !pack_last_to_interm;

    const uint32_t in0_cb_id = buf_id(in0_buf);
    const uint32_t in1_cb_id = buf_id(in1_buf);
    const uint32_t out_cb_id = buf_id(out_buf);
    const uint32_t interm_cb_id = buf_id(interm_buf);
    const uint32_t reload_cb_id =
        shape.partials_reload_cb_id == UINT32_MAX ? interm_cb_id : shape.partials_reload_cb_id;

    // Accumulate directly into out only when finalize would be a same-format copy.
    // Keep conversion and activation paths in interm. Check formats at runtime
    // because a BF16 accumulator may feed a narrower output even without FP32 DST.
    constexpr bool alias_out_eligible =
        packer_l1_acc && (tile_order == OutputCBLayout::SubblockMajor) &&
        (last_block_target != LastBlockTarget::Interm) && (last_block_target != LastBlockTarget::OutWithRelu) &&
        (Activation::activation == KernelActivation::NONE) && !get_fp32_dest_acc_enabled();
    const bool alias_out = alias_out_eligible && (unpack_src_format[out_cb_id] == unpack_src_format[interm_cb_id]);
    Buf& accum_buf = alias_out ? out_buf : interm_buf;
    const uint32_t accum_cb_id = alias_out ? out_cb_id : interm_cb_id;

    ASSERT(shape.in0_block_k > 0);
    ASSERT(shape.in0_num_subblocks > 0);
    ASSERT(shape.in1_num_subblocks > 0);
    ASSERT(shape.num_k_blocks > 0);
    ASSERT(shape.out_subblock_h > 0);
    ASSERT(shape.out_subblock_w > 0);
    ASSERT(shape.batch > 0);
    ASSERT(in0_cb_id != out_cb_id);
    ASSERT(in1_cb_id != out_cb_id);
    ASSERT(shape.out_subblock_h * shape.out_subblock_w <= compute_kernel_lib::DEST_AUTO_LIMIT);
    // ShortAfterPreKBlock defers initialization and reconfiguration until after preprocessing.
    if constexpr (init_mode != matmul_config::InitMode::ShortAfterPreKBlock) {
        if constexpr (
            reconfig == matmul_config::DataFormatReconfig::Input ||
            reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
            // Matmul convention: srca takes in1, srcb takes in0.
            reconfig_data_format(in1_cb_id, in0_cb_id);
        }
        // Even a single K block requires interm to have the output format.
        if constexpr (
            reconfig == matmul_config::DataFormatReconfig::Output ||
            reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
            PACK((pack_reconfig_data_format(accum_cb_id)));
        }
        if constexpr (init_mode == matmul_config::InitMode::Short) {
            matmul_block_init(
                in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
        }
    }

    const uint32_t out_subblock_num_tiles = shape.out_subblock_h * shape.out_subblock_w;
    const uint32_t in0_subblock_num_tiles = shape.out_subblock_h * shape.in0_block_k;
    const uint32_t in0_block_num_tiles = in0_subblock_num_tiles * shape.in0_num_subblocks;
    uint32_t in1_per_core_w = shape.in1_per_core_w;
    if (in1_per_core_w == 0) {
        in1_per_core_w = shape.out_subblock_w * shape.in1_num_subblocks;
    }
    uint32_t out_row_width = shape.out_row_width;
    if (out_row_width == 0) {
        out_row_width = in1_per_core_w;
    }
    const uint32_t in1_block_num_tiles = in1_per_core_w * shape.in0_block_k;
    const uint32_t out_block_num_tiles = out_subblock_num_tiles * shape.in0_num_subblocks * shape.in1_num_subblocks;
    const uint32_t row_group_tiles = shape.out_subblock_h * out_row_width;

    for (uint32_t b = 0; b < shape.batch; b++) {
        // L1 accumulation holds one reservation across the entire K loop.
        if constexpr (packer_l1_acc) {
            accum_buf.reserve_back(out_block_num_tiles);
        }

        for (uint32_t block = 0; block < shape.num_k_blocks; block++) {
            const bool last_out = block == (shape.num_k_blocks - 1);
            const bool enable_reload = !packer_l1_acc && block != 0;

            pre_k_block(block, shape.num_k_blocks, last_out);

            const uint32_t inner_steps = k_block_inner_dim(block, shape.in0_block_k);

            const uint32_t active_in0_cb_id = in0_source_fn(block, in0_cb_id);
            Buf active_in0_buf(active_in0_cb_id);

            // Preprocessing may change compute state; restore it for the selected input CB.
            if constexpr (init_mode == matmul_config::InitMode::ShortAfterPreKBlock) {
                if constexpr (
                    reconfig == matmul_config::DataFormatReconfig::Input ||
                    reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
                    reconfig_data_format(in1_cb_id, active_in0_cb_id);
                }
                if constexpr (
                    reconfig == matmul_config::DataFormatReconfig::Output ||
                    reconfig == matmul_config::DataFormatReconfig::InputAndOutput) {
                    PACK((pack_reconfig_data_format(accum_cb_id)));
                }
                matmul_block_init(
                    active_in0_cb_id,
                    in1_cb_id,
                    transpose,
                    shape.out_subblock_w,
                    shape.out_subblock_h,
                    shape.in0_block_k);
            }

            const uint32_t in1_base_offset = in1_base_offset_fn(block);

            // Wait for the full input block before computing its subblocks.
            active_in0_buf.wait_front(in0_block_num_tiles);
            if constexpr (in1_policy != InputPolicy::NoWaitNoPop) {
                in1_buf.wait_front(in1_block_num_tiles);
            }

            Buf& pack_target_buf = pack_last_to_interm ? accum_buf : out_buf;
            const uint32_t pack_target_id = pack_last_to_interm ? accum_cb_id : out_cb_id;

            // Reserve the full block before spilling when interm shares out's L1 storage.
            // L1 accumulation already reserved it before the K loop.
            if constexpr (tile_order == OutputCBLayout::SubblockMajor && !packer_l1_acc) {
                if (block == 0 && !last_out) {
                    out_buf.reserve_back(out_block_num_tiles);
                }
            }

            // Use the same row-strided placement for L1 accumulation and software reload.
            constexpr bool trm_row_grouped = tile_order == OutputCBLayout::TileRowMajor;

            // Packer format and accumulation mode are constant within each K block.
            if constexpr (packer_l1_acc) {
                PACK((pack_reconfig_data_format(accum_cb_id)));
                PACK((llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)));
            }

            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < shape.in0_num_subblocks; in0_subblock++) {
                if constexpr (tile_order == OutputCBLayout::TileRowMajor && !packer_l1_acc) {
                    // Reuse fronted Interm storage for reload; reserve only a distinct destination.
                    if (enable_reload) {
                        interm_buf.wait_front(row_group_tiles);
                    }
                    if (last_out && !pack_last_to_interm) {
                        pack_target_buf.reserve_back(row_group_tiles);
                    } else if (!enable_reload) {
                        interm_buf.reserve_back(row_group_tiles);
                    }
                }

                int in1_index_subblock_offset = in1_base_offset;
                for (uint32_t in1_subblock = 0; in1_subblock < shape.in1_num_subblocks; in1_subblock++) {
                    tile_regs_acquire();

                    // Narrow the final FMA width without changing the packed subblock size.
                    const uint32_t effective_subblock_w =
                        (shape.last_in1_subblock_w_valid != 0 && in1_subblock == shape.in1_num_subblocks - 1)
                            ? shape.last_in1_subblock_w_valid
                            : shape.out_subblock_w;

                    if (enable_reload) {
                        reconfig_data_format_srca(in1_cb_id, reload_cb_id);
                        copy_init(reload_cb_id);
                        if (reload_cb_id != interm_cb_id) {
                            DataflowBuffer source(interm_cb_id), reload(reload_cb_id);
                            UNPACK((reload.evil_set_read_ptr(source.get_read_ptr())));
                        }
                        if constexpr (trm_row_grouped) {
                            // Gather the fronted row-strided partials into contiguous DST.
                            copy_subblock_row_strided(
                                reload_cb_id,
                                in1_subblock * shape.out_subblock_w,
                                out_row_width,
                                shape.out_subblock_h,
                                shape.out_subblock_w);
                        } else {
                            interm_buf.wait_front(out_subblock_num_tiles);
                            copy_block(reload_cb_id, 0, 0, out_subblock_num_tiles);
                            interm_buf.pop_front(out_subblock_num_tiles);
                        }
#ifndef ARCH_QUASAR
                        reconfig_data_format_srca(reload_cb_id, in1_cb_id);
                        matmul_block_init(
                            in0_cb_id,
                            in1_cb_id,
                            transpose,
                            shape.out_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#endif
                    }

                    uint32_t dst_index = 0;
                    uint32_t in0_index = in0_index_subblock_offset;
                    uint32_t in1_index = in1_index_subblock_offset;
                    for (uint32_t inner_dim = 0; inner_dim < inner_steps; inner_dim++) {
#ifndef SKIP_COMPUTE
                        ckernel::matmul_block(
                            active_in0_cb_id,
                            in1_cb_id,
                            in0_index,
                            in1_index,
                            dst_index,
                            transpose,
                            effective_subblock_w,
                            shape.out_subblock_h,
                            shape.in0_block_k);
#else
                        (void)in0_index;
                        (void)in1_index;
                        (void)dst_index;
#endif
                        in0_index++;
                        in1_index += in1_per_core_w;
                    }

                    if (last_out) {
                        post_compute(out_subblock_num_tiles);
                    }
                    Buf& dst_buf = last_out ? pack_target_buf : accum_buf;
                    const uint32_t dst_cb_id = last_out ? pack_target_id : accum_cb_id;
                    tile_regs_commit();
                    if constexpr (!packer_l1_acc && get_fp32_dest_acc_enabled()) {
                        PACK((pack_reconfig_data_format(dst_cb_id)));
                    }
                    if constexpr (tile_order == OutputCBLayout::SubblockMajor && !packer_l1_acc) {
                        dst_buf.reserve_back(out_subblock_num_tiles);
                    }
                    if constexpr (activate_on_last_pack && Activation::activation != KernelActivation::NONE) {
                        if (last_out) {
                            apply_activation_from_pack<
                                Activation::activation,
                                Activation::param0,
                                Activation::param1,
                                Activation::param2>(out_subblock_num_tiles);
                        } else {
                            tile_regs_wait();
                        }
                    } else {
                        tile_regs_wait();
                    }
                    if constexpr (activate_on_last_pack && apply_relu_on_last_block) {
                        if (last_out) {
                            PACK((llk_pack_relu_config(ReluConfig::zero())));
                        }
                    }

                    if constexpr (packer_l1_acc && tile_order == OutputCBLayout::SubblockMajor) {
                        // Aliased output advances its write pointer as final subblocks are published.
                        const uint32_t offset =
                            alias_out && last_out
                                ? 0u
                                : (in0_subblock * shape.in1_num_subblocks + in1_subblock) * out_subblock_num_tiles;
                        pack_subblock_at_offset(0, dst_cb_id, offset, out_subblock_num_tiles);
                    } else if constexpr (tile_order == OutputCBLayout::TileRowMajor) {
                        const uint32_t row_base = packer_l1_acc ? in0_subblock * row_group_tiles : 0;
                        pack_subblock_row_strided(
                            0,
                            dst_cb_id,
                            row_base + in1_subblock * shape.out_subblock_w,
                            out_row_width,
                            shape.out_subblock_h,
                            shape.out_subblock_w);
                    } else {
                        pack_block(0, dst_cb_id, out_subblock_num_tiles);
                    }
                    tile_regs_release();
                    if constexpr (activate_on_last_pack && apply_relu_on_last_block) {
                        if (last_out) {
                            PACK((llk_pack_relu_config(ReluConfig::none())));
                        }
                    }
                    if constexpr (tile_order == OutputCBLayout::SubblockMajor) {
                        if constexpr (!packer_l1_acc) {
                            dst_buf.push_back(out_subblock_num_tiles);
                        } else if (alias_out && last_out) {
                            dst_buf.push_back(out_subblock_num_tiles);
                        }
                    }

                    in1_index_subblock_offset += shape.out_subblock_w;
                }

                if constexpr (tile_order == OutputCBLayout::TileRowMajor && !packer_l1_acc) {
                    if (enable_reload) {
                        interm_buf.pop_front(row_group_tiles);
                    }
                    // After an in-place rewrite, release the old row before restoring producer credits.
                    if (enable_reload && (!last_out || pack_last_to_interm)) {
                        interm_buf.reserve_back(row_group_tiles);
                    }
                    if (last_out) {
                        pack_target_buf.push_back(row_group_tiles);
                    } else {
                        interm_buf.push_back(row_group_tiles);
                    }
                }

                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            if constexpr (in0_policy == InputPolicy::WaitAndPopPerKBlock) {
                active_in0_buf.pop_front(in0_block_num_tiles);
            } else {
                if (!last_out) {
                    active_in0_buf.pop_front(in0_block_num_tiles);
                }
            }
            if constexpr (in1_policy == InputPolicy::WaitAndPopPerKBlock) {
                in1_buf.pop_front(in1_block_num_tiles);
            } else if constexpr (in1_policy == InputPolicy::WaitAndRetainOnLastBlock) {
                if (!last_out) {
                    in1_buf.pop_front(in1_block_num_tiles);
                }
            }

            post_k_block(block, shape.num_k_blocks, last_out);
        }

        if constexpr (packer_l1_acc) {
            // Aliased out was already published per subblock.
            if (!alias_out) {
                accum_buf.push_back(out_block_num_tiles);
            }
            PACK((llk_pack_reconfig_l1_acc(0)));
        }

        // Finalize only when the accumulated result still needs to move from interm to out.
        if constexpr (pack_last_to_interm && !target_is_interm) {
            const bool needs_finalize = (out_cb_id != accum_cb_id);
            if (needs_finalize) {
                matmul_block_finalize_copy<packer_l1_acc, apply_relu_on_last_block, Activation>(
                    interm_buf,
                    out_buf,
                    out_block_num_tiles,
                    out_subblock_num_tiles,
                    /*prev_srca_cb_id=*/in1_cb_id);
                // Restore matmul state after the copy if another batch follows.
                if (b + 1 < shape.batch) {
                    reconfig_data_format(in1_cb_id, in0_cb_id);
                    PACK((pack_reconfig_data_format(interm_cb_id)));
                    matmul_block_init(
                        in0_cb_id, in1_cb_id, transpose, shape.out_subblock_w, shape.out_subblock_h, shape.in0_block_k);
                }
            }
        }
    }
}

}  // namespace detail

template <
    bool transpose,
    bool packer_l1_acc,
    LastBlockTarget last_block_target,
    OutputCBLayout tile_order,
    matmul_config::InitMode init_mode,
    InputPolicy in0_policy,
    InputPolicy in1_policy,
    matmul_config::DataFormatReconfig reconfig,
    typename Activation,
    typename Shape,
    typename Buf>
ALWI void matmul_block(Buf& in0_buf, Buf& in1_buf, Buf& out_buf, Buf& interm_buf, const Shape& shape) {
    detail::matmul_block_impl<
        transpose,
        packer_l1_acc,
        last_block_target,
        tile_order,
        init_mode,
        in0_policy,
        in1_policy,
        NoPostCompute,
        NoPreKBlock,
        NoPostKBlock,
        NoKBlockInnerDimFn,
        NoIn0Source,
        NoIn1BaseOffset,
        Activation,
        reconfig,
        Shape,
        Buf>(in0_buf, in1_buf, out_buf, interm_buf, shape);
}

}  // namespace compute_kernel_lib
