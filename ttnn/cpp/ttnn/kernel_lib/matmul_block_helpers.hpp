// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

namespace compute_kernel_lib {

/**
 * Tiled C = A x B, with dimensions measured in tiles. Each call produces M x N
 * tiles per batch, reducing K in num_k_blocks chunks of in0_block_k tiles.
 * Each output subblock must fit in DST. See matmul_block for buffer contracts.
 */

/**
 * Output tile order; computation always proceeds by subblock.
 * SubblockMajor groups tiles by subblock; TileRowMajor groups them by tile row.
 * Both are tiled storage, not untilized element layout. Writers and downstream
 * helpers must use the same order.
 */
enum class OutputCBLayout { SubblockMajor, TileRowMajor };

/**
 * Out materializes the accumulated block in out_buf. OutWithRelu also applies
 * packer ReLU after the full K reduction. Interm leaves the result in interm_buf
 * for a downstream operation, such as bias addition or untilize.
 */
enum class LastBlockTarget : uint8_t { Out, OutWithRelu, Interm };

/**
 * WaitAndPopPerKBlock waits and pops each K block.
 * WaitAndRetainOnLastBlock waits each block but retains the last for caller reuse.
 * NoWaitNoPop leaves lifecycle management to the caller; supported only for in1.
 */
enum class InputPolicy : uint8_t { WaitAndPopPerKBlock, WaitAndRetainOnLastBlock, NoWaitNoPop };

namespace matmul_config {

/**
 * Matmul initialization, independent of data-format reconfiguration.
 * Short initializes before the K loop; None assumes the caller has initialized.
 * ShortAfterPreKBlock restores matmul state after each preprocessing callback.
 * The caller owns boot-time hardware startup; do not repeat hardware startup mid-kernel.
 */
enum class InitMode : uint8_t { Short, None, ShortAfterPreKBlock };

/**
 * Input reconfigures unpack formats for (in1, in0); Output configures the packer
 * for interm. InputAndOutput does both. None requires compatible existing state.
 * These gates are independent of InitMode.
 */
enum class DataFormatReconfig : uint8_t { InputAndOutput, Input, Output, None };

}  // namespace matmul_config

/**
 * Block dimensions in tiles: M = in0_num_subblocks * out_subblock_h,
 * N = in1_num_subblocks * out_subblock_w, K = num_k_blocks * in0_block_k.
 * Optional stride overrides describe padded input and output buffers.
 */
struct MatmulBlockShape {
    uint32_t in0_num_subblocks;  // Output subblock count along M.
    uint32_t in1_num_subblocks;  // Output subblock count along N.
    uint32_t out_subblock_h;     // Output subblock height in tiles.
    uint32_t out_subblock_w;     // Output subblock width in tiles.
    uint32_t in0_block_k;        // K per K-block in tiles (= legacy "in0_block_w").
    uint32_t num_k_blocks;       // Number of K-blocks along K.
    uint32_t batch = 1;          // Independent batch slices; use an outer loop for work between matmuls.

    // Valid columns in the last in1 subblock; 0 uses out_subblock_w.
    // Only the FMA width narrows; packing remains full-width for the writer to trim.
    uint32_t last_in1_subblock_w_valid = 0;

    // Actual N tiles per input K block; 0 derives the width from the subblock grid.
    // Must match the producer, including when output subblocks are padded.
    uint32_t in1_per_core_w = 0;

    // TileRowMajor output row stride; 0 uses in1_per_core_w.
    uint32_t out_row_width = 0;

    // Optional UnpackToDestFp32 view for lossless K-loop reload with an Interm target.
    uint32_t partials_reload_cb_id = UINT32_MAX;

    static constexpr MatmulBlockShape of(
        uint32_t in0_num_subblocks,
        uint32_t in1_num_subblocks,
        uint32_t out_subblock_h,
        uint32_t out_subblock_w,
        uint32_t in0_block_k,
        uint32_t num_k_blocks,
        uint32_t batch = 1,
        uint32_t in1_per_core_w = 0,
        uint32_t out_row_width = 0) {
        return {
            in0_num_subblocks,
            in1_num_subblocks,
            out_subblock_h,
            out_subblock_w,
            in0_block_k,
            num_k_blocks,
            batch,
            /*last_in1_subblock_w_valid=*/0,
            in1_per_core_w,
            out_row_width};
    }
};

// Compile-time shape for kernels whose block dimensions are fixed at compilation.
template <
    uint32_t IN0_NUM_SUBBLOCKS,
    uint32_t IN1_NUM_SUBBLOCKS,
    uint32_t OUT_SUBBLOCK_H,
    uint32_t OUT_SUBBLOCK_W,
    uint32_t IN0_BLOCK_K,
    uint32_t NUM_K_BLOCKS,
    uint32_t BATCH = 1,
    uint32_t LAST_IN1_SUBBLOCK_W_VALID = 0,
    uint32_t IN1_PER_CORE_W = 0,
    uint32_t OUT_ROW_WIDTH = 0>
struct StaticMatmulBlockShape {
    static constexpr uint32_t in0_num_subblocks = IN0_NUM_SUBBLOCKS;
    static constexpr uint32_t in1_num_subblocks = IN1_NUM_SUBBLOCKS;
    static constexpr uint32_t out_subblock_h = OUT_SUBBLOCK_H;
    static constexpr uint32_t out_subblock_w = OUT_SUBBLOCK_W;
    static constexpr uint32_t in0_block_k = IN0_BLOCK_K;
    static constexpr uint32_t num_k_blocks = NUM_K_BLOCKS;
    static constexpr uint32_t batch = BATCH;
    static constexpr uint32_t last_in1_subblock_w_valid = LAST_IN1_SUBBLOCK_W_VALID;
    static constexpr uint32_t in1_per_core_w = IN1_PER_CORE_W;
    static constexpr uint32_t out_row_width = OUT_ROW_WIDTH;
    uint32_t partials_reload_cb_id = UINT32_MAX;
};

// Called per output subblock on the last K block, before packing; receives the DST tile count.
struct NoPostCompute {
    ALWI void operator()(uint32_t /* out_subblock_num_tiles */) const {}
};

// Called before each K-block input wait with (block, num_k_blocks, is_last).
// If preprocessing changes compute state, use ShortAfterPreKBlock to restore it,
// or None and restore unpack formats and matmul initialization in the callback.
struct NoPreKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Called after input pops and the L1 accumulation drain with (block, num_k_blocks, is_last).
struct NoPostKBlock {
    ALWI void operator()(uint32_t, uint32_t, bool) const {}
};

// Returns the FMA step count for a K block. Reducing it does not change the
// LLK kt_dim stride, which remains in0_block_k.
struct NoKBlockInnerDimFn {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t block_k) const { return block_k; }
};

// Selects the in0 CB for each K block. Alternate CBs must share the bound CB's data format.
struct NoIn0Source {
    ALWI uint32_t operator()(uint32_t /*block*/, uint32_t in0_cb_id) const { return in0_cb_id; }
};

// Offsets in1 reads within the fronted buffer; wait/pop still use the bound CB.
// Use NoWaitNoPop when the caller also advances the buffer externally.
struct NoIn1BaseOffset {
    ALWI uint32_t operator()(uint32_t /*block*/) const { return 0; }
};

/**
 * Compute a subblocked matmul, with software spill/reload or packer L1 accumulation.
 * Pass MatmulBlockShape for runtime dimensions or StaticMatmulBlockShape for
 * compile-time dimensions. For callbacks, use matmul_block_gathered.
 *
 * Before the first call, include compute_kernel_hw_startup.h and call
 * compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out), then
 * matmul_block_init(in0, in1, transpose, subblock_w, subblock_h, block_k).
 * Initialize any SFPU activation at startup with ActivationInitHelper::init().
 * Subsequent calls use the init_mode and reconfig gates.
 *
 * Buffers may be CircularBuffer or DataflowBuffer. in0, in1, and out must be
 * distinct CBs. For each K block, stream M x block_k tiles into in0 and
 * block_k x N tiles into in1. InputPolicy controls their wait/pop lifecycle.
 * interm stores partial sums; it may share out's L1 storage when layouts and
 * capacity are compatible. For a single K block, pass out as the unused interm
 * placeholder: output reconfiguration still reads interm's format.
 *
 * The writer must accept tile_order. For untilized output, target Interm,
 * then use reblock_and_untilize for SubblockMajor or untilize for TileRowMajor.
 * transpose changes each B tile internally; the caller must arrange the tile grid.
 *
 * Activation runs on the packer thread when materializing the completed output.
 * If bias or another downstream operation must precede activation, leave it as
 * NoneActivation here and activate in that operation. OutWithRelu also waits
 * until the full sum is available.
 *
 * Math fidelity and FP32 accumulation come from the host ComputeConfig. Avoid
 * HiFi4 with BF16 inputs and FP32 DST on Wormhole B0 (issue #38306).
 * SKIP_COMPUTE omits only the matmul LLK call, retaining buffer synchronization.
 *
 * @example
 *   CircularBuffer a(cb_in0), b(cb_in1), out(cb_out);
 *   matmul_block<>(a, b, out, out, MatmulBlockShape::of(Mt, Nt, 1, 1, Kt, 1));
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
    typename Shape,
    typename Buf = ::CircularBuffer>
ALWI void matmul_block(Buf& in0_buf, Buf& in1_buf, Buf& out_buf, Buf& interm_buf, const Shape& shape);

}  // namespace compute_kernel_lib

#include "matmul_block_helpers.inl"
