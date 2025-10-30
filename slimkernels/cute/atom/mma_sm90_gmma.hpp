
//
// Copyright (c) 2025
//
// This file references CUTLASS CUTE GMMA primitives for SM90.
//
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm90_gmma.hpp>


namespace cute {

namespace SM90::GMMA {

////////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA 64x64x64 TN F16+=E4M3*E4M3
// A matrix is 64X64 2bit packed, 每个线程用2个32bit寄存器存储32个数
// B matrix is 64X64 FP8 elements, 每个线程有2个desc_b分别指向两块 64X32的 smem 空间。
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One
>
struct MMA_64x64x64_F16I2E4M3_RS_TN
{
  using DRegisters = void;
  using ARegisters = uint32_t[2];
  using BRegisters = uint64_t[2];
  using CRegisters = uint32_t[16];

  CUTE_HOST_DEVICE static void
  fma(uint32_t const& a00, uint32_t const& a01,
      uint64_t const& desc_b0, uint64_t const& desc_b1,
      uint32_t      & d00, uint32_t      & d01, uint32_t      & d02, uint32_t      & d03,
      uint32_t      & d04, uint32_t      & d05, uint32_t      & d06, uint32_t      & d07,
      uint32_t      & d08, uint32_t      & d09, uint32_t      & d10, uint32_t      & d11,
      uint32_t      & d12, uint32_t      & d13, uint32_t      & d14, uint32_t      & d15,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
  {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)


    // 获取高16bit
    uint16_t hi16_a00 = static_cast<uint16_t>(a00 >> 16);
    uint16_t hi16_a01 = static_cast<uint16_t>(a01 >> 16);

    // 每2bit取出，模拟转E4M3（这里只是简单搬运2bit到3bit，实际可以按映射需求填充）
    // a00
    uint32_t a00_e4m3_0 =
          ((uint32_t)((hi16_a00 >> 14) & 0x3) << 0)
        | ((uint32_t)((hi16_a00 >> 12) & 0x3) << 8)
        | ((uint32_t)((hi16_a00 >> 10) & 0x3) << 16)
        | ((uint32_t)((hi16_a00 >> 8 ) & 0x3) << 24);
    uint32_t a00_e4m3_1 =
          ((uint32_t)((hi16_a00 >> 6 ) & 0x3) << 0)
        | ((uint32_t)((hi16_a00 >> 4 ) & 0x3) << 8)
        | ((uint32_t)((hi16_a00 >> 2 ) & 0x3) << 16)
        | ((uint32_t)((hi16_a00 >> 0 ) & 0x3) << 24);

    // a01
    uint32_t a01_e4m3_0 =
          ((uint32_t)((hi16_a01 >> 14) & 0x3) << 0)
        | ((uint32_t)((hi16_a01 >> 12) & 0x3) << 8)
        | ((uint32_t)((hi16_a01 >> 10) & 0x3) << 16)
        | ((uint32_t)((hi16_a01 >> 8 ) & 0x3) << 24);
    uint32_t a01_e4m3_1 =
          ((uint32_t)((hi16_a01 >> 6 ) & 0x3) << 0)
        | ((uint32_t)((hi16_a01 >> 4 ) & 0x3) << 8)
        | ((uint32_t)((hi16_a01 >> 2 ) & 0x3) << 16)
        | ((uint32_t)((hi16_a01 >> 0 ) & 0x3) << 24);


    cutlass::arch::synclog_emit_wgmma_reg_smem(__LINE__, desc_b0);
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      "{%16, %17, %18, %19},"
      " %20,"
      " p,   %22, %23;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15)
      :  "r"(a00_e4m3_0),  "r"(a01_e4m3_0),  "r"(a00_e4m3_1),  "r"(a01_e4m3_1),
         "l"(desc_b0),
         "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)));


    // 获取低16bit
    uint16_t lo16_a00 = static_cast<uint16_t>(a00 & 0xFFFF);
    uint16_t lo16_a01 = static_cast<uint16_t>(a01 & 0xFFFF);
    // 每2bit取出，模拟转E4M3（这里只是简单搬运2bit到3bit，实际可以按映射需求填充）
    // a00
    uint32_t a00_e4m3_2 =
          ((uint32_t)((lo16_a00 >> 14) & 0x3) << 0)
        | ((uint32_t)((lo16_a00 >> 12) & 0x3) << 8)
        | ((uint32_t)((lo16_a00 >> 10) & 0x3) << 16)
        | ((uint32_t)((lo16_a00 >> 8 ) & 0x3) << 24);
    uint32_t a00_e4m3_3 =
          ((uint32_t)((lo16_a00 >> 6 ) & 0x3) << 0)
        | ((uint32_t)((lo16_a00 >> 4 ) & 0x3) << 8)
        | ((uint32_t)((lo16_a00 >> 2 ) & 0x3) << 16)
        | ((uint32_t)((lo16_a00 >> 0 ) & 0x3) << 24);

    // a01
    uint32_t a01_e4m3_2 =
          ((uint32_t)((lo16_a01 >> 14) & 0x3) << 0)
        | ((uint32_t)((lo16_a01 >> 12) & 0x3) << 8)
        | ((uint32_t)((lo16_a01 >> 10) & 0x3) << 16)
        | ((uint32_t)((lo16_a01 >> 8 ) & 0x3) << 24);
    uint32_t a01_e4m3_3 =
          ((uint32_t)((lo16_a01 >> 6 ) & 0x3) << 0)
        | ((uint32_t)((lo16_a01 >> 4 ) & 0x3) << 8)
        | ((uint32_t)((lo16_a01 >> 2 ) & 0x3) << 16)
        | ((uint32_t)((lo16_a01 >> 0 ) & 0x3) << 24);

    cutlass::arch::synclog_emit_wgmma_reg_smem(__LINE__, desc_b1);
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      "{%16, %17, %18, %19},"
      " %20,"
      " p,   %22, %23;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15)
      :  "r"(a00_e4m3_2),  "r"(a01_e4m3_2),  "r"(a00_e4m3_3),  "r"(a01_e4m3_3),
          "l"(desc_b1),
          "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)));

#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use MMA_64x64x32_F16E4M3E4M3_RS_TN without CUTE_ARCH_MMA_SM90A_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
}
}