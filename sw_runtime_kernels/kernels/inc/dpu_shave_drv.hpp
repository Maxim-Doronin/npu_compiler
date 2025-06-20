//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <common_types.h>
#include <moviVectorTypes.h>
#include <cstring>
#include "shave_hal.hpp"

#ifndef GET_REG_DWORD_VAL
#define GET_REG_DWORD_VAL(a) (*(volatile uint64_t*)(((unsigned)(a))))
#endif  // GET_REG_DWORD_VAL

#ifndef SET_REG_DWORD
#define SET_REG_DWORD(a, x) ((void)(*(volatile uint64_t*)(((uintptr_t)(a))) = (uint64_t)(x)))
#endif

#ifndef SET_REG_WORD
#define SET_REG_WORD(a, x) ((void)(*(volatile uint32_t*)(((unsigned)(a))) = (uint32_t)(x)))
#endif

#pragma pack(push, 1)

typedef struct {
    uint32_t cmx_slice0_low_addr;
    uint32_t cmx_slice1_low_addr;
    uint32_t cmx_slice2_low_addr;
    uint32_t cmx_slice3_low_addr;
    uint32_t cmx_slice_size;
    uint32_t se_addr;
    uint32_t sparsity_addr;
    uint32_t se_size;  // se_sp_size

    union {
        uint32_t z_config;
        struct {
            uint32_t se_z_split : 4;
            uint32_t num_ses_in_z_dir : 9;
            uint32_t cm_sp_pattern : 16;
            uint32_t npo2_se_z_split_enable : 1;
            uint32_t reserved : 1;
            uint32_t addr_format_sel : 1;
        } z_config_bf;
    } z_config;
    union {
        uint32_t kernel_pad_cfg;
        struct {
            uint32_t mpe_assign : 1;
            uint32_t pad_right_en : 1;
            uint32_t pad_left_en : 1;
            uint32_t pad_bottom_en : 1;
            uint32_t pad_top_en : 1;
            uint32_t kernel_y : 4;
            uint32_t kernel_x : 4;
            uint32_t wt_plt_cfg : 2;
            uint32_t act_dense : 1;
            uint32_t wt_dense : 1;
            uint32_t stride_y_en : 1;
            uint32_t stride_y : 3;
            uint32_t dynamic_bw_en : 1;
            uint32_t dw_wt_sp_ins : 1;
            uint32_t layer1_wt_sp_ins : 1;
            uint32_t layer1_cmp_en : 1;
            uint32_t pool_opt_en : 1;
            uint32_t unused1 : 3;
            uint32_t sp_se_tbl_segment : 1;
            uint32_t rst_ctxt : 1;
            uint32_t unused2 : 1;
        } kernel_pad_cfg_bf;
    } kernel_pad_cfg;
    union {
        uint32_t tensor_size0;
        struct {
            uint32_t tensor_size_x : 14;
            uint32_t tensor_size_y : 14;
            uint32_t unused : 4;
        } tensor_size0_bf;
    } tensor_size0;
    union {
        uint32_t tensor_size1;
        struct {
            uint32_t tensor_size_z : 14;
            uint32_t npo2_se_size : 9;
            uint32_t tensor_cmp_size : 4;    // (next only)
            uint32_t tensor_cmp_offset : 4;  // (next only)
            uint32_t unused : 1;
        } tensor_size1_bf;
    } tensor_size1;
    uint32_t tensor_start;
    union {
        uint32_t tensor_mode;
        struct {
            uint32_t wmode : 4;
            uint32_t amode : 4;
            uint32_t stride : 3;
            uint32_t zm_input : 1;
            uint32_t dw_input : 1;
            uint32_t cm_input : 1;
            uint32_t workload_operation : 2;
            uint32_t pad_value : 16;
        } tensor_mode_bf;
    } tensor_mode;
    uint32_t elops_sparsity_addr;
    uint32_t elops_se_addr;

    union {
        uint32_t elops_wload;
        struct {
            uint32_t elop_wload : 1;
            uint32_t seed_wload : 1;
            uint32_t fifo_wr_wload : 1;
            uint32_t elop_wload_type : 1;
            uint32_t pool_wt_data : 16;
            uint32_t unused1 : 6;
            uint32_t pool_wt_rd_dis : 1;
            uint32_t elop_operation : 3;  // (next only)
            uint32_t repeat_dis : 1;      // (next only)
            uint32_t unused2 : 1;
        } elops_wload_bf;
    } elops_wload;
    uint32_t act_offset[4];
    uint32_t base_offset_a;

    union {
        uint32_t base_offset_b;
        struct {
            uint32_t base_offset2 : 9;
            uint32_t base_offset3 : 9;
            uint32_t dw_opt_offset : 6;
            uint32_t dw_opt_en : 1;
            uint32_t dw_3x3s1_opt_dis : 1;
            uint32_t wt_dense_opt_en : 1;   // (next only)
            uint32_t small_hw_opt_en : 1;   // (next only)
            uint32_t idu_cmx_mux_mode : 2;  // (next only)
            uint32_t reserved_0 : 2;
        } base_offset_b_bf;
    } base_offset_b;

    uint32_t wt_offset;

    union {
        uint32_t odu_cfg;
        struct {
            uint32_t dtype : 3;
            uint32_t wcb_ac_mode : 1;
            uint32_t wcb_sp_mode : 1;
            uint32_t sp_value : 8;
            uint32_t sp_out_en : 1;
            uint32_t cmx_port_muxing_disable : 1;
            uint32_t write_sp : 1;
            uint32_t write_pt : 1;
            uint32_t write_ac : 1;
            uint32_t mode : 2;
            uint32_t grid : 1;
            uint32_t swizzle_key : 3;
            uint32_t wl_bp_on_start_en : 1;
            uint32_t nthw : 2;
            uint32_t permutation : 3;
            uint32_t wcb_stall_avoidance : 1;
            uint32_t wcb_bypass : 1;
        } odu_cfg_bf;
    } odu_cfg;
    uint32_t odu_be_size;
    uint32_t odu_be_cnt;
    uint32_t odu_se_size;
    union {
        uint32_t te_dim0;
        struct {
            uint32_t te_dim_y : 13;
            uint32_t te_dim_z : 13;
            uint32_t unused : 6;
        } te_dim0_bf;
    } te_dim0;
    union {
        uint32_t te_dim1;
        struct {
            uint32_t te_dim_x : 13;
            uint32_t unused : 19;
        } te_dim1_bf;
    } te_dim1;
    uint32_t pt_base;
    uint32_t sp_base;

    union {
        uint32_t mpe_cfg;
        struct {
            uint32_t mpe_wtbias : 8;
            uint32_t mpe_actbias : 8;
            uint32_t mpe_mode : 3;
            uint32_t mpe_dense : 1;
            uint32_t mrm_weight_dense : 1;
            uint32_t mrm_act_dense : 1;
            uint32_t mpe_daz : 1;
            uint32_t mpe_ftz : 1;
            uint32_t unused : 8;
        } mpe_cfg_bf;
    } mpe_cfg;
    uint32_t mpe_bus_data_sel;
    union {
        uint32_t elop_scale;
        struct {
            uint32_t elop_scale_b : 16;
            uint32_t elop_scale_a : 16;
        } elop_scale_bf;
    } elop_scale;
    union {
        uint32_t ppe_cfg;
        struct {
            uint32_t ppe_g8_bias_c : 9;
            uint32_t ppe_g8_bias_b : 9;
            uint32_t ppe_g8_bias_a : 9;
            uint32_t unused : 5;
        } ppe_cfg_bf;
    } ppe_cfg;
    uint32_t ppe_bias;
    union {
        uint32_t ppe_scale;
        struct {
            uint32_t unused : 2;
            uint32_t ppe_scale_shift : 6;
            uint32_t unused1 : 2;
            uint32_t ppe_scale_round : 2;
            uint32_t unused2 : 4;
            uint32_t ppe_scale_mult : 16;
        } ppe_scale_bf;
    } ppe_scale;
    union {
        uint32_t ppe_scale_ctrl;
        struct {
            uint32_t ppe_scale_override : 1;
            uint32_t ppe_fp_scale_override : 1;
            uint32_t unused : 30;
        } ppe_scale_ctrl_bf;
    } ppe_scale_ctrl;
    union {
        uint32_t ppe_prelu;
        struct {
            uint32_t unused : 8;           // 0-7
            uint32_t ppe_prelu_shift : 5;  // 8-12
            uint32_t unused1 : 3;          // 13-15
            uint32_t ppe_prelu_mult : 11;  // 16-26
            uint32_t unused2 : 5;          // 27-31
        } ppe_prelu_bf;
    } ppe_prelu;

    uint32_t ppe_scale_hclamp;
    uint32_t ppe_scale_lclamp;
    union {
        uint32_t ppe_misc;
        struct {
            uint32_t ppe_mode : 2;     // (next only)
            uint32_t ppe_rnd_fd : 2;   // (next only)
            uint32_t ppe_rnd_int : 2;  // (next only)
            uint32_t ppe_fp16_ftz : 1;
            uint32_t ppe_fp16_clamp : 1;
            uint32_t ppe_i32_convert : 2;
            uint32_t ppe_sb_dtype : 2;    // (next only)
            uint32_t ppe_mult2_mode : 1;  // (next only)
            uint32_t unused : 19;
        } ppe_misc_bf;
    } ppe_misc;
    uint32_t ppe_fp_bias;
    uint32_t ppe_fp_scale;
    uint32_t ppe_fp_prelu;
    union {
        uint32_t ppe_fp_cfg;
        struct {
            uint32_t ppe_fp_convert : 3;
            uint32_t ppe_fp_bypass : 1;
            uint32_t ppe_bf16_round : 1;
            uint32_t ppe_fp_prelu_en : 1;
            uint32_t unused : 26;
        } ppe_fp_cfg_bf;
    } ppe_fp_cfg;

    union {
        uint32_t odu_ac_base;
        struct {
            uint32_t unused : 4;
            uint32_t ac_base : 28;
        } odu_ac_base_bf;
    } odu_ac_base;

    union {
        uint32_t hwp_ctrl;
        struct {
            uint32_t hwp_en : 1;
            uint32_t hwp_stat_mode : 3;
            uint32_t local_timer_en : 1;
            uint32_t local_timer_rst : 1;
            uint32_t rsvd : 10;
            uint32_t unique_id : 16;
        } hwp_ctrl_bf;
    } hwp_ctrl;

    union {
        uint32_t hwp_cmx_mem_addr;
        struct {
            uint32_t mem_addr : 27;
            uint32_t rsvd : 5;
        } hwp_cmx_mem_addr_bf;
    } hwp_cmx_mem_addr;

    union {
        uint32_t odu_cast;
        struct {
            uint32_t cast_enable : 1;
            uint32_t reserved : 3;
            uint32_t cast_offset : 28;
        } odu_cast_bf;
    } odu_cast[3];

    uint32_t tensor2_start;  // (next only)

    union {
        uint32_t ppe_lut_ptr;
        struct {
            uint32_t ppe_lut_ptr : 16;
            uint32_t unused_1 : 2;
            uint32_t ppe_lut_ptr_force : 1;
            uint32_t unused_2 : 13;
        } ppe_lut_ptr_bf;
    } ppe_lut_ptr;  // (next only)

    uint32_t nvar_tag;

    uint32_t pallet[8];

    uint32_t se_addr1;
    uint32_t sparsity_addr1;
    uint32_t se_addr2;
    uint32_t sparsity_addr2;
    uint32_t se_addr3;
    uint32_t sparsity_addr3;
    uint32_t se_sp_size1;
    uint32_t se_sp_size2;

} VpuDPUInvariantRegisters;

static_assert(sizeof(VpuDPUInvariantRegisters) == 288, "VpuDPUInvariantRegisters size != 288");

/* VPU$ HALO Support */
#define NUM_HALO_REGIONS 6

typedef struct {
    union {
        uint32_t halo_region_a;
        struct {
            uint32_t sp_adr_offset : 22;
            uint32_t tile_select : 7;
            uint32_t rsvd : 2;
            uint32_t enable : 1;
        } halo_region_a_bf;
    } halo_region_a;

    union {
        uint32_t halo_region_b;
        struct {
            uint32_t ac_adr_offset : 22;
            uint32_t target_width_lsb : 10;
        } halo_region_b_bf;
    } halo_region_b;

    union {
        uint32_t halo_region_c;
        struct {
            uint32_t begin_x : 13;
            uint32_t begin_y : 13;
            uint32_t target_width_msb : 4;
            uint32_t rsvd : 2;
        } halo_region_c_bf;
    } halo_region_c;

    union {
        uint32_t halo_region_d;
        struct {
            uint32_t end_x : 13;
            uint32_t end_y : 13;
            uint32_t rsvd : 6;
        } halo_region_d_bf;
    } halo_region_d;

} halo_region_t;

typedef struct {
    union {
        uint32_t invar_ptr;
        struct {
            uint32_t invar_ptr : 16;
            uint32_t var_tag : 16;
        } invar_ptr_bf;
    } invar_ptr;
    union {
        uint32_t workload_size0;
        struct {
            uint32_t workload_size_x : 14;
            uint32_t workload_size_y : 14;
            uint32_t unused : 4;
        } workload_size0_bf;
    } workload_size0;
    union {
        uint32_t workload_size1;
        struct {
            uint32_t workload_size_z : 14;
            uint32_t pad_count_up : 3;
            uint32_t pad_count_left : 3;
            uint32_t pad_count_down : 3;
            uint32_t pad_count_right : 3;
            uint32_t unused : 6;
        } workload_size1_bf;
    } workload_size1;
    union {
        uint32_t workload_start0;
        struct {
            uint32_t workload_start_x : 14;
            uint32_t workload_start_y : 14;
            uint32_t unused : 4;
        } workload_start0_bf;
    } workload_start0;
    union {
        uint32_t workload_start1;
        struct {
            uint32_t workload_start_z : 14;
            uint32_t unused : 18;
        } workload_start1_bf;
    } workload_start1;
    union {
        uint32_t offset_addr;
        struct {
            uint32_t nthw_ntk : 2;
            uint32_t bin_cfg : 1;
            uint32_t conv_cond : 1;
            uint32_t dense_se : 1;
            uint32_t idx_quad : 1;
            uint32_t swizzle_key : 3;
            uint32_t idu_mrm_clk_en : 1;
            uint32_t odu_clk_en : 1;
            uint32_t mpe_clk_en : 1;
            uint32_t ppe_clk_en : 1;
            uint32_t odu_stat_en : 1;
            uint32_t idu_stat_en : 1;
            uint32_t reserved_1 : 1;
            uint32_t odu_stat_clr_mode : 1;
            uint32_t idu_stat_clr_mode : 1;
            uint32_t se_only_en : 1;  // (next only)
            uint32_t shave_l2_cache_en : 1;
            uint32_t idu_dbg_en : 2;
            uint32_t sb_read_en : 1;         // (next only)
            uint32_t tensor2_act_dense : 1;  // (next only)
            uint32_t reserved_2 : 3;
            uint32_t wt_swizzle_key : 3;
            uint32_t wt_swizzle_sel : 1;
            uint32_t gif_clk_en : 1;  // (next only)
        } offset_addr_bf;
    } offset_addr;

    union {
        uint32_t hwp_wload_id;
        struct {
            uint32_t wload_id : 16;
            uint32_t rsvd : 16;
        } hwp_wload_id_bf;
    } hwp_wload_id;

    union {
        uint32_t var_cfg;
        struct {
            uint32_t reserved_0 : 8;
            uint32_t invar_lut_rd_en : 1;  // (next only)
            uint32_t invar_line_cnt_en : 1;
            uint32_t invar_line_cnt_cnt : 4;
            uint32_t invar_lptr_force : 1;
            uint32_t next_sram_job_valid : 1;
            uint32_t next_sram_job_addr : 16;
        } var_cfg_bf;
    } var_cfg;

    uint64_t cbarrier_lo;
    uint64_t cbarrier_hi;
    uint64_t pbarrier_lo;
    uint64_t pbarrier_hi;

    halo_region_t halo_region[NUM_HALO_REGIONS];

    union {
        uint32_t dpu_cfg;
        struct {
            uint32_t workload_start_odu : 1;
            uint32_t workload_start_idu : 1;
            uint32_t workload_prm_sel : 1;
            uint32_t workload_valid : 1;
            uint32_t workload_shad_odu : 1;
            uint32_t workload_shad_idu : 1;
            uint32_t workload_idu_auto_upd_0 : 1;
            uint32_t workload_idu_auto_upd_1 : 1;
            uint32_t workload_odu_auto_upd : 1;
            uint32_t cfg_Reserved_0 : 1;
            uint32_t cfg_Reserved_1 : 1;
            uint32_t cfg_Reserved_2 : 1;
            uint32_t rst_ctxt_new : 1;
            uint32_t cfg_Reserved_3 : 1;
            uint32_t cfg_Reserved_4 : 1;
            uint32_t odu_stat_clr : 1;
            uint32_t idu_stat_clr : 1;
            uint32_t cfg_Reserved_5 : 1;
            uint32_t cfg_Reserved_6 : 14;
        } dpu_cfg_bf;
    } dpu_cfg;
    union {
        uint32_t te_beg0;
        struct {
            uint32_t te_beg_y : 13;
            uint32_t te_beg_z : 13;
            uint32_t unused : 6;
        } te_beg0_bf;
    } te_beg0;
    union {
        uint32_t te_beg1;
        struct {
            uint32_t te_beg_x : 13;
            uint32_t unused : 19;
        } te_beg1_bf;
    } te_beg1;
    union {
        uint32_t te_end0;
        struct {
            uint32_t te_end_y : 13;
            uint32_t te_end_z : 13;
            uint32_t unused : 6;
        } te_end0_bf;
    } te_end0;
    union {
        uint32_t te_end1;
        struct {
            uint32_t te_end_x : 13;
            uint32_t unused : 19;
        } te_end1_bf;
    } te_end1;

    uint32_t weight_size;
    uint32_t weight_num;
    uint32_t weight_start;

} VpuDPUVariantRegisters;

static_assert(sizeof(VpuDPUVariantRegisters) == 192, "VpuDPUVariantRegisters size != 192");

struct DpuHwpIduOduMode {
    std::uint32_t idu_workload_duration;
    std::uint16_t idu_workload_id;
    std::uint16_t dpu_id_1;
    std::uint32_t idu_timestamp_l;
    std::uint32_t idu_timestamp_h;
    std::uint32_t odu_workload_duration;
    std::uint16_t odu_workload_id;
    std::uint16_t dpu_id_2;
    std::uint32_t odu_timestamp_l;
    std::uint32_t odu_timestamp_h;
};
static_assert(sizeof(DpuHwpIduOduMode) == 32, "DpuHwpIduOduMode size != 32");

//
typedef struct {
    void* desc;
    VpuDPUVariantRegisters* variant;
    VpuDPUInvariantRegisters* invariant;
    DpuHwpIduOduMode* stats;
} DpuTaskDescriptor;

#pragma pack(pop)

void initInvariant(VpuDPUInvariantRegisters* cfg) {
    cfg->cmx_slice0_low_addr = 0x4000000;
    cfg->cmx_slice1_low_addr = 0x4000000;
    cfg->cmx_slice2_low_addr = 0x4000000;
    cfg->cmx_slice3_low_addr = 0x4000000;
    cfg->cmx_slice_size = CMX_SLICE_SIZE;
    cfg->se_addr = 0;
    cfg->sparsity_addr = 0;
    cfg->se_size = 0;
    cfg->z_config.z_config_bf.se_z_split = 0;
    cfg->z_config.z_config_bf.num_ses_in_z_dir = 0;
    cfg->z_config.z_config_bf.cm_sp_pattern = 0;
    cfg->z_config.z_config_bf.npo2_se_z_split_enable = 0;
    cfg->z_config.z_config_bf.reserved = 0;
    cfg->z_config.z_config_bf.addr_format_sel = 1;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.pad_right_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.pad_left_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.pad_bottom_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.pad_top_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y = 1;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x = 1;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.wt_plt_cfg = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.act_dense = 1;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense = 1;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.stride_y = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en = 1;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.pool_opt_en = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment = 0;
    cfg->kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt = 1;
    cfg->tensor_size0.tensor_size0_bf.tensor_size_x = 1;
    cfg->tensor_size0.tensor_size0_bf.tensor_size_y = 1;
    cfg->tensor_size1.tensor_size1_bf.tensor_size_z = 0x80;
    cfg->tensor_size1.tensor_size1_bf.npo2_se_size = 0;
    cfg->tensor_start = 0;
    cfg->tensor_mode.tensor_mode_bf.wmode = 0;
    cfg->tensor_mode.tensor_mode_bf.amode = 0;
    cfg->tensor_mode.tensor_mode_bf.stride = 0;
    cfg->tensor_mode.tensor_mode_bf.zm_input = 1;
    cfg->tensor_mode.tensor_mode_bf.dw_input = 0;
    cfg->tensor_mode.tensor_mode_bf.cm_input = 0;
    cfg->tensor_mode.tensor_mode_bf.workload_operation = 0;
    cfg->tensor_mode.tensor_mode_bf.pad_value = 0;
    cfg->elops_sparsity_addr = 0;
    cfg->elops_se_addr = 0;
    cfg->elops_wload.elops_wload_bf.elop_wload = 0;
    cfg->elops_wload.elops_wload_bf.seed_wload = 0;
    cfg->elops_wload.elops_wload_bf.fifo_wr_wload = 0;
    cfg->elops_wload.elops_wload_bf.elop_wload_type = 0;
    cfg->elops_wload.elops_wload_bf.pool_wt_data = 0;
    cfg->elops_wload.elops_wload_bf.pool_wt_rd_dis = 0;
    cfg->act_offset[0] = 0x18700;
    cfg->act_offset[1] = 0x18700;
    cfg->act_offset[2] = 0x18700;
    cfg->act_offset[3] = 0x18700;
    cfg->base_offset_a = 0x200;
    cfg->base_offset_b.base_offset_b_bf.base_offset2 = 0;
    cfg->base_offset_b.base_offset_b_bf.base_offset3 = 0;
    cfg->base_offset_b.base_offset_b_bf.dw_opt_offset = 0;
    cfg->base_offset_b.base_offset_b_bf.dw_opt_en = 0;
    cfg->base_offset_b.base_offset_b_bf.dw_3x3s1_opt_dis = 0;
    cfg->wt_offset = 0x18000;
    cfg->odu_cfg.odu_cfg_bf.dtype = 4;
    cfg->odu_cfg.odu_cfg_bf.wcb_ac_mode = 0;
    cfg->odu_cfg.odu_cfg_bf.wcb_sp_mode = 0;
    cfg->odu_cfg.odu_cfg_bf.sp_value = 0;
    cfg->odu_cfg.odu_cfg_bf.sp_out_en = 0;
    cfg->odu_cfg.odu_cfg_bf.cmx_port_muxing_disable = 0;
    cfg->odu_cfg.odu_cfg_bf.write_sp = 0;
    cfg->odu_cfg.odu_cfg_bf.write_pt = 0;
    cfg->odu_cfg.odu_cfg_bf.write_ac = 1;
    cfg->odu_cfg.odu_cfg_bf.mode = 0;
    cfg->odu_cfg.odu_cfg_bf.grid = 0;
    cfg->odu_cfg.odu_cfg_bf.swizzle_key = 0;
    cfg->odu_cfg.odu_cfg_bf.wl_bp_on_start_en = 0;
    cfg->odu_cfg.odu_cfg_bf.nthw = 1;
    cfg->odu_cfg.odu_cfg_bf.permutation = 0;
    cfg->odu_cfg.odu_cfg_bf.wcb_stall_avoidance = 0;
    cfg->odu_cfg.odu_cfg_bf.wcb_bypass = 0;
    cfg->odu_be_size = 0;
    cfg->odu_be_cnt = 0;
    cfg->odu_se_size = 0;
    cfg->te_dim0.te_dim0_bf.te_dim_y = 0;
    cfg->te_dim0.te_dim0_bf.te_dim_z = 0x1FF;
    cfg->te_dim1.te_dim1_bf.te_dim_x = 0;
    cfg->pt_base = 0;
    cfg->sp_base = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mpe_wtbias = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mpe_actbias = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mpe_mode = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mpe_dense = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mrm_weight_dense = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mrm_act_dense = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mpe_daz = 0;
    cfg->mpe_cfg.mpe_cfg_bf.mpe_ftz = 0;
    cfg->mpe_bus_data_sel = 0;
    cfg->elop_scale.elop_scale_bf.elop_scale_b = 0;
    cfg->elop_scale.elop_scale_bf.elop_scale_a = 0;
    cfg->ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c = 0;
    cfg->ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b = 0;
    cfg->ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a = 0;
    cfg->ppe_bias = 0;
    cfg->ppe_scale.ppe_scale_bf.ppe_scale_shift = 0;
    cfg->ppe_scale.ppe_scale_bf.ppe_scale_round = 0;
    cfg->ppe_scale.ppe_scale_bf.ppe_scale_mult = 1;
    cfg->ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override = 1;
    cfg->ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override = 0;
    cfg->ppe_prelu.ppe_prelu_bf.ppe_prelu_shift = 0;
    cfg->ppe_prelu.ppe_prelu_bf.ppe_prelu_mult = 1;
    cfg->ppe_scale_hclamp = 0x7FFFFFFF;
    cfg->ppe_scale_lclamp = 0x80000000;
    cfg->ppe_misc.ppe_misc_bf.ppe_fp16_ftz = 0;
    cfg->ppe_misc.ppe_misc_bf.ppe_fp16_clamp = 1;
    cfg->ppe_misc.ppe_misc_bf.ppe_i32_convert = 0;
    cfg->ppe_fp_bias = 0;
    cfg->ppe_fp_scale = 0;
    cfg->ppe_fp_prelu = 0;
    cfg->ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert = 1;
    cfg->ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass = 0;
    cfg->ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round = 0;
    cfg->ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en = 0;
    cfg->odu_ac_base.odu_ac_base_bf.ac_base = 0;
    cfg->hwp_ctrl.hwp_ctrl_bf.hwp_en = 1;
    cfg->hwp_ctrl.hwp_ctrl_bf.hwp_stat_mode = 3;
    cfg->hwp_ctrl.hwp_ctrl_bf.local_timer_en = 0;
    cfg->hwp_ctrl.hwp_ctrl_bf.local_timer_rst = 0;
    cfg->hwp_ctrl.hwp_ctrl_bf.unique_id = 0xA55A;
    cfg->hwp_cmx_mem_addr.hwp_cmx_mem_addr = 0;
    cfg->odu_cast[0].odu_cast_bf.cast_enable = 0;
    cfg->odu_cast[0].odu_cast_bf.cast_offset = 0;
    cfg->odu_cast[1].odu_cast_bf.cast_enable = 0;
    cfg->odu_cast[1].odu_cast_bf.cast_offset = 0;
    cfg->odu_cast[2].odu_cast_bf.cast_enable = 0;
    cfg->odu_cast[2].odu_cast_bf.cast_offset = 0;
    cfg->nvar_tag = 1;
    cfg->pallet[0] = 0;
    cfg->pallet[0] = 0;
    cfg->pallet[1] = 0;
    cfg->pallet[1] = 0;
    cfg->pallet[2] = 0;
    cfg->pallet[2] = 0;
    cfg->pallet[3] = 0;
    cfg->pallet[3] = 0;
    cfg->pallet[4] = 0;
    cfg->pallet[4] = 0;
    cfg->pallet[5] = 0;
    cfg->pallet[5] = 0;
    cfg->pallet[6] = 0;
    cfg->pallet[6] = 0;
    cfg->pallet[7] = 0;
    cfg->pallet[7] = 0;
    cfg->se_addr1 = 0;
    cfg->sparsity_addr1 = 0;
    cfg->se_addr2 = 0;
    cfg->sparsity_addr2 = 0;
    cfg->se_addr3 = 0;
    cfg->sparsity_addr3 = 0;
    cfg->se_sp_size1 = 0;
    cfg->se_sp_size2 = 0;
}

void initVariant(VpuDPUVariantRegisters* cfg) {
    cfg->invar_ptr.invar_ptr_bf.invar_ptr = 0x335;
    cfg->invar_ptr.invar_ptr_bf.var_tag = 1;
    cfg->workload_size0.workload_size0_bf.workload_size_x = 1;
    cfg->workload_size0.workload_size0_bf.workload_size_y = 1;
    cfg->workload_size1.workload_size1_bf.workload_size_z = 0x80;
    cfg->workload_size1.workload_size1_bf.pad_count_up = 0;
    cfg->workload_size1.workload_size1_bf.pad_count_left = 0;
    cfg->workload_size1.workload_size1_bf.pad_count_down = 0;
    cfg->workload_size1.workload_size1_bf.pad_count_right = 0;
    cfg->workload_start0.workload_start0_bf.workload_start_x = 0;
    cfg->workload_start0.workload_start0_bf.workload_start_y = 0;
    cfg->workload_start1.workload_start1_bf.workload_start_z = 0;
    cfg->offset_addr.offset_addr_bf.nthw_ntk = 3;
    cfg->offset_addr.offset_addr_bf.bin_cfg = 0;
    cfg->offset_addr.offset_addr_bf.conv_cond = 0;
    cfg->offset_addr.offset_addr_bf.dense_se = 1;
    cfg->offset_addr.offset_addr_bf.idx_quad = 0;
    cfg->offset_addr.offset_addr_bf.swizzle_key = 0;
    cfg->offset_addr.offset_addr_bf.idu_mrm_clk_en = 0;
    cfg->offset_addr.offset_addr_bf.odu_clk_en = 0;
    cfg->offset_addr.offset_addr_bf.mpe_clk_en = 0;
    cfg->offset_addr.offset_addr_bf.ppe_clk_en = 0;
    cfg->offset_addr.offset_addr_bf.odu_stat_en = 1;
    cfg->offset_addr.offset_addr_bf.idu_stat_en = 1;
    cfg->offset_addr.offset_addr_bf.odu_stat_clr_mode = 0;
    cfg->offset_addr.offset_addr_bf.idu_stat_clr_mode = 0;
    cfg->offset_addr.offset_addr_bf.shave_l2_cache_en = 0;
    cfg->offset_addr.offset_addr_bf.idu_dbg_en = 0;
    cfg->offset_addr.offset_addr_bf.wt_swizzle_key = 0;
    cfg->offset_addr.offset_addr_bf.wt_swizzle_sel = 1;
    cfg->hwp_wload_id.hwp_wload_id_bf.wload_id = 0;
    cfg->var_cfg.var_cfg_bf.invar_line_cnt_en = 0;
    cfg->var_cfg.var_cfg_bf.invar_line_cnt_cnt = 0;
    cfg->var_cfg.var_cfg_bf.invar_lptr_force = 1;
    cfg->var_cfg.var_cfg_bf.next_sram_job_valid = 0;
    cfg->var_cfg.var_cfg_bf.next_sram_job_addr = 0;
    cfg->cbarrier_lo = 0;
    cfg->cbarrier_hi = 0;
    cfg->pbarrier_lo = 0;
    cfg->pbarrier_hi = 0;
    memset(&cfg->halo_region, 0, sizeof(cfg->halo_region));
    cfg->dpu_cfg.dpu_cfg_bf.workload_start_odu = 1;
    cfg->dpu_cfg.dpu_cfg_bf.workload_start_idu = 1;
    cfg->dpu_cfg.dpu_cfg_bf.workload_prm_sel = 0;
    cfg->dpu_cfg.dpu_cfg_bf.workload_valid = 0;
    cfg->dpu_cfg.dpu_cfg_bf.workload_shad_odu = 0;
    cfg->dpu_cfg.dpu_cfg_bf.workload_shad_idu = 0;
    cfg->dpu_cfg.dpu_cfg_bf.workload_idu_auto_upd_0 = 1;
    cfg->dpu_cfg.dpu_cfg_bf.workload_idu_auto_upd_1 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.workload_odu_auto_upd = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_0 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_1 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_2 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.rst_ctxt_new = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_3 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_4 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.odu_stat_clr = 0;
    cfg->dpu_cfg.dpu_cfg_bf.idu_stat_clr = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_5 = 0;
    cfg->dpu_cfg.dpu_cfg_bf.cfg_Reserved_6 = 0;
    cfg->te_beg0.te_beg0_bf.te_beg_y = 0;
    cfg->te_beg0.te_beg0_bf.te_beg_z = 0;
    cfg->te_beg1.te_beg1_bf.te_beg_x = 0;
    cfg->te_end0.te_end0_bf.te_end_y = 0;
    cfg->te_end0.te_end0_bf.te_end_z = 0x1FF;
    cfg->te_end1.te_end1_bf.te_end_x = 0;
    cfg->weight_size = 0x80;
    cfg->weight_num = 0x200;
    cfg->weight_start = 0x18200;
}

// Fifo and Barrier control:
#define VPU_NCE_NCE_SPINE_CMX_CTRL_BASE 0x2f000000U

#define VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_0_OFFSET (0x00000000U)
#define VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_STAT_0_OFFSET (0x00000600U)
#define FIFO_REG_ADDR_INDEX_SHIFT 5
#define WL_FIFO_PTR_SHIFT 5

#define VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_0_ADR \
    (((VPU_NCE_NCE_SPINE_CMX_CTRL_BASE) + (VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_0_OFFSET)))
#define VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_STAT_0_ADR \
    (((VPU_NCE_NCE_SPINE_CMX_CTRL_BASE) + (VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_STAT_0_OFFSET)))
#define FIFO_SEND_ADR_DPU VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_0_ADR
#define FIFO_STAT_ADR_DPU VPU_NCE_NCE_SPINE_CMX_CTRL_NCE_FIFO_STAT_0_ADR

typedef enum { BARRIER_PRODUCER = 0, BARRIER_CONSUMER, BARRIER_PROCESS_NUM } BarrierType;
typedef union {
    struct {
        uint64_t nce_barrier_count_p_cnt : 8;
        ///< Producer count. Cleared by write of 1 to corresponding bit in NCE_BARRIER_PRESET.
        ///< AccessType="RW" BitOffset="0" ResetValue="0x0"
        uint64_t nce_barrier_count_p_int_en : 1;
        ///< Barrier producer interrupt enable
        ///< AccessType="RO" BitOffset="8" ResetValue="0x0"
        uint64_t reserved_0 : 7;
        ///< Reserved
        ///< AccessType="RO" BitOffset="9" ResetValue="None"
        uint64_t nce_barrier_count_c_cnt : 8;
        ///< Consumer count. Cleared by write of 1 to corresponding bit in NCE_BARRIER_CRESET.
        ///< AccessType="RW" BitOffset="16" ResetValue="0x0"
        uint64_t nce_barrier_count_c_int_en : 1;
        ///< Barrier consumer interrupt enable
        ///< AccessType="RO" BitOffset="24" ResetValue="0x0"
        uint64_t reserved_1 : 39;
        ///< Reserved
        ///< AccessType="RO" BitOffset="25" ResetValue="None"
    };
    uint64_t whole;
} BarrierCountRegStruct;

void fifoDynamicSendDPU(unsigned int tile, void* p) {
    // Remove the CMX base address from the workload
    uint32_t ptrVal = static_cast<uint32_t>(reinterpret_cast<uint64_t>(p) & 0x1FFFFF);
    // Right shift the workload pointer by WL_FIFO_PTR_SHIFT (currently defined to be 5) because the FIFO
    // is 16 bits large. The pointer will be left shifted by the same amount once it reaches the DPU/ACT-SHV.
    ptrVal = ptrVal >> WL_FIFO_PTR_SHIFT;
    SET_REG_WORD(FIFO_SEND_ADR_DPU + (tile << FIFO_REG_ADDR_INDEX_SHIFT), ptrVal);
}

void startDpu(DpuTaskDescriptor* dpuTaskDescriptor) {
    // clean stats
    dpuTaskDescriptor->stats->odu_workload_duration = 0;
    __asm volatile("NOP 10 \n" ::: "memory");
    // Write in DPU FIFO the descriptor
    fifoDynamicSendDPU(getTileId(), dpuTaskDescriptor->desc);
}

void waitDpuTask(DpuTaskDescriptor* dpuTaskDescriptor) {
    // wait for fifo produced to be updated, dpu finish execution
    // wait for HWP to produce data
    while (*((volatile std::uint32_t*)&(dpuTaskDescriptor->stats->odu_workload_duration)) == 0)
        __asm volatile("NOP 10 \n" ::: "memory");
}
