// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "internal/risc_attribs.h"

using ttnn::operations::matmul::KernelActivation;

namespace compute_kernel_lib {

/**
 * Compile-time activation kind and parameters for matmul and bias helpers.
 * Use the named aliases below or ActivationOp for host-supplied compile-time arguments.
 */
template <KernelActivation ACT, uint32_t P0 = 0, uint32_t P1 = 0, uint32_t P2 = 0>
struct ActivationOp {
    static constexpr KernelActivation activation = ACT;
    static constexpr uint32_t param0 = P0;
    static constexpr uint32_t param1 = P1;
    static constexpr uint32_t param2 = P2;
};

using NoneActivation = ActivationOp<KernelActivation::NONE>;
using SiluActivation = ActivationOp<KernelActivation::SILU>;
using HardsigmoidActivation = ActivationOp<KernelActivation::HARDSIGMOID>;
// TanhActivation / GeluActivation: Fast=0 selects the accurate variant; non-zero selects fast.
template <uint32_t Fast = 0>
using TanhActivation = ActivationOp<KernelActivation::TANH, Fast>;
template <uint32_t Fast = 0>
using GeluActivation = ActivationOp<KernelActivation::GELU, Fast>;
// Relu6Activation: MaxBits is the max value as a uint32_t bit pattern of the float;
// 0 selects the default 6.0f.
template <uint32_t MaxBits = 0>
using Relu6Activation = ActivationOp<KernelActivation::RELU6, MaxBits>;
// SigmoidActivation: VecMode 1=R, 2=C, else RC. Fast non-zero enables fast approximation.
template <uint32_t VecMode = 0, uint32_t Fast = 0>
using SigmoidActivation = ActivationOp<KernelActivation::SIGMOID, VecMode, Fast>;
// HardtanhActivation<low_bits, high_bits>: low/high are uint32_t bit patterns of the floats.
template <uint32_t LowBits, uint32_t HighBits>
using HardtanhActivation = ActivationOp<KernelActivation::HARDTANH, LowBits, HighBits>;
// SeluActivation<alpha_bits, lambda_bits>: alpha/lambda are uint32_t bit patterns.
template <uint32_t AlphaBits, uint32_t LambdaBits>
using SeluActivation = ActivationOp<KernelActivation::SELU, AlphaBits, LambdaBits>;
// SoftplusActivation<beta_bits, threshold_bits, beta_reciprocal_bits>: float bit patterns.
// beta must be non-zero (validated via static_assert in ActivationApplyHelper).
template <uint32_t BetaBits, uint32_t ThresholdBits, uint32_t BetaReciprocalBits>
using SoftplusActivation = ActivationOp<KernelActivation::SOFTPLUS, BetaBits, ThresholdBits, BetaReciprocalBits>;

// Activation init and apply run on the packer thread (TRISC2).
// Initialize at kernel startup; math-thread post-compute hooks remain independent.
template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0>
struct ActivationInitHelper {
    static_assert(
        ACT == KernelActivation::NONE || ACT == KernelActivation::SILU || ACT == KernelActivation::TANH ||
            ACT == KernelActivation::GELU || ACT == KernelActivation::RELU6 || ACT == KernelActivation::SIGMOID ||
            ACT == KernelActivation::HARDSIGMOID || ACT == KernelActivation::HARDTANH ||
            ACT == KernelActivation::SELU || ACT == KernelActivation::SOFTPLUS,
        "Unsupported KernelActivation type for fused activation init");

    FORCE_INLINE static void init() {
        if constexpr (ACT == KernelActivation::SILU) {
            silu_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::TANH) {
            tanh_tile_init_pack<PARAM0 != 0>();
        } else if constexpr (ACT == KernelActivation::GELU) {
            gelu_tile_init_pack<PARAM0 != 0>();
        } else if constexpr (ACT == KernelActivation::RELU6) {
            relu_max_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SIGMOID) {
            sigmoid_tile_init_pack<PARAM1 != 0>();
        } else if constexpr (ACT == KernelActivation::HARDSIGMOID) {
            hardsigmoid_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::HARDTANH) {
            hardtanh_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SELU) {
            selu_tile_init_pack();
        } else if constexpr (ACT == KernelActivation::SOFTPLUS) {
            softplus_tile_init_pack();
        }
    }
};

template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0, uint32_t PARAM2 = 0>
struct ActivationApplyHelper {
    static_assert(
        ACT == KernelActivation::NONE || ACT == KernelActivation::SILU || ACT == KernelActivation::TANH ||
            ACT == KernelActivation::GELU || ACT == KernelActivation::RELU6 || ACT == KernelActivation::SIGMOID ||
            ACT == KernelActivation::HARDSIGMOID || ACT == KernelActivation::HARDTANH ||
            ACT == KernelActivation::SELU || ACT == KernelActivation::SOFTPLUS,
        "Unsupported KernelActivation type for fused activation apply");

    static_assert(
        ACT != KernelActivation::SOFTPLUS || PARAM0 != 0,
        "SOFTPLUS PARAM0 (beta) must be non-zero to avoid division by zero");

    FORCE_INLINE static void apply(uint32_t tile_index) {
        if constexpr (ACT == KernelActivation::SILU) {
            silu_tile_pack(tile_index);
        } else if constexpr (ACT == KernelActivation::TANH) {
            tanh_tile_pack<PARAM0 != 0>(tile_index);
        } else if constexpr (ACT == KernelActivation::GELU) {
            gelu_tile_pack<PARAM0 != 0>(tile_index);
        } else if constexpr (ACT == KernelActivation::RELU6) {
            constexpr uint32_t max = (PARAM0 != 0) ? PARAM0 : 0x40c00000u;
            relu_max_tile_pack(tile_index, max);
        } else if constexpr (ACT == KernelActivation::SIGMOID) {
            constexpr int vec_mode = (PARAM0 == 1) ? VectorMode::R : (PARAM0 == 2) ? VectorMode::C : VectorMode::RC;
            sigmoid_tile_pack<vec_mode, PARAM1 != 0>(tile_index);
        } else if constexpr (ACT == KernelActivation::HARDSIGMOID) {
            hardsigmoid_tile_pack(tile_index);
        } else if constexpr (ACT == KernelActivation::HARDTANH) {
            hardtanh_tile_pack(tile_index, PARAM0, PARAM1);
        } else if constexpr (ACT == KernelActivation::SELU) {
            selu_tile_pack(tile_index, PARAM0, PARAM1);
        } else if constexpr (ACT == KernelActivation::SOFTPLUS) {
            // The SFPU API takes beta reciprocal before threshold.
            softplus_tile_pack(tile_index, PARAM0, PARAM2, PARAM1);
        }
    }
};

// Apply activation to DST on the packer thread, replacing tile_regs_wait().
template <KernelActivation ACT, uint32_t PARAM0 = 0, uint32_t PARAM1 = 0, uint32_t PARAM2 = 0>
FORCE_INLINE void apply_activation_from_pack(uint32_t out_subblock_num_tiles) {
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));

    // Select the packer's DST half.
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        ActivationApplyHelper<ACT, PARAM0, PARAM1, PARAM2>::apply(i);
    }

    // Wait for SFPU completion before packing.
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}

}  // namespace compute_kernel_lib
