// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t k_blocks = get_compile_time_arg_val(0);
    constexpr bool l1_acc = get_compile_time_arg_val(1);
    constexpr bool row_major = get_compile_time_arg_val(2);
    constexpr uint32_t post_op = get_compile_time_arg_val(3);  // 0: none, 1: bias, 2: ReLU, 3: ReLU6, 4: bias + ReLU6
    constexpr bool static_shape = get_compile_time_arg_val(4);
    const auto shape = [] {
        if constexpr (static_shape) {
            return StaticMatmulBlockShape<2, 2, 2, 2, 1, k_blocks>{};
        } else {
            return MatmulBlockShape::of(2, 2, 2, 2, 1, k_blocks);
        }
    }();
    constexpr auto layout = row_major ? OutputCBLayout::TileRowMajor : OutputCBLayout::SubblockMajor;
    constexpr bool with_bias = post_op == 1 || post_op == 4;
    constexpr auto target =
        with_bias ? LastBlockTarget::Interm : (post_op == 2 ? LastBlockTarget::OutWithRelu : LastBlockTarget::Out);
    constexpr uint32_t in0 = 0, in1 = 1, partials = 2, bias = 3, out = 16;
    CircularBuffer a(in0), b(in1), p(partials), c(out), bias_buf(bias);
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_block_init(in0, in1, false, 2, 2, 1);
    if constexpr (post_op >= 3) {
        ActivationInitHelper<KernelActivation::RELU6>::init();
    }

    // Inputs reside in sharded L1 tensors, ordered by K block by the test.
    a.reserve_back(4 * k_blocks);
    a.push_back(4 * k_blocks);
    b.reserve_back(4 * k_blocks);
    b.push_back(4 * k_blocks);
    if constexpr (with_bias) {
        bias_buf.reserve_back(4);
        bias_buf.push_back(4);
    }
    using MatmulActivation = std::conditional_t<post_op == 3, Relu6Activation<>, NoneActivation>;
    using BiasActivation = std::conditional_t<post_op == 4, Relu6Activation<>, NoneActivation>;
    compute_kernel_lib::matmul_block<
        false,
        l1_acc,
        target,
        layout,
        matmul_config::InitMode::Short,
        InputPolicy::WaitAndPopPerKBlock,
        InputPolicy::WaitAndPopPerKBlock,
        matmul_config::DataFormatReconfig::InputAndOutput,
        MatmulActivation>(a, b, c, p, shape);
    if constexpr (with_bias) {
        bias_buf.wait_front(4);
        pack_reconfig_data_format(out);
        add_bias_bcast_rows<BiasBroadcast::RowBroadcast, layout, bias_add_config::NoPostBias, BiasActivation>(
            p, bias_buf, c, BiasAddShape::of(2, 2, 2, 2));
        bias_buf.pop_front(4);
    }
}
