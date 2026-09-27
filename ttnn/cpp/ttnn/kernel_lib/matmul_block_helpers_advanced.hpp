// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"

namespace compute_kernel_lib {

/**
 * Matmul with per-K-block and per-subblock callbacks:
 * - PostComputeFn: math-thread hook before the last K block is packed.
 * - PreKBlockFn / PostKBlockFn: preprocessing and input-lifecycle hooks.
 * - KBlockInnerDimFn: valid FMA step count for padded K blocks.
 * - In0SourceFn: alternate in0 CB with the same data format.
 * - In1BaseOffsetFn: tile offset within the fronted in1 buffer.
 *
 * See the No* functor contracts and matmul_block documentation for initialization,
 * buffer ownership, and shared parameters. Use matmul_block when no callbacks are needed.
 */
template <
    bool transpose = false,
    bool packer_l1_acc = false,
    LastBlockTarget last_block_target = LastBlockTarget::Out,
    OutputCBLayout tile_order = OutputCBLayout::SubblockMajor,
    matmul_config::InitMode init_mode = matmul_config::InitMode::Short,
    InputPolicy in0_policy = InputPolicy::WaitAndPopPerKBlock,
    InputPolicy in1_policy = InputPolicy::WaitAndPopPerKBlock,
    matmul_config::DataFormatReconfig reconfig = matmul_config::DataFormatReconfig::InputAndOutput,
    typename Activation = NoneActivation,
    typename PostComputeFn = NoPostCompute,
    typename PreKBlockFn = NoPreKBlock,
    typename PostKBlockFn = NoPostKBlock,
    typename KBlockInnerDimFn = NoKBlockInnerDimFn,
    typename In0SourceFn = NoIn0Source,
    typename In1BaseOffsetFn = NoIn1BaseOffset,
    typename Shape,
    typename Buf = ::CircularBuffer>
ALWI void matmul_block_gathered(
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
    detail::matmul_block_impl<
        transpose,
        packer_l1_acc,
        last_block_target,
        tile_order,
        init_mode,
        in0_policy,
        in1_policy,
        PostComputeFn,
        PreKBlockFn,
        PostKBlockFn,
        KBlockInnerDimFn,
        In0SourceFn,
        In1BaseOffsetFn,
        Activation,
        reconfig,
        Shape,
        Buf>(
        in0_buf,
        in1_buf,
        out_buf,
        interm_buf,
        shape,
        post_compute,
        pre_k_block,
        post_k_block,
        k_block_inner_dim,
        in0_source_fn,
        in1_base_offset_fn);
}

}  // namespace compute_kernel_lib
