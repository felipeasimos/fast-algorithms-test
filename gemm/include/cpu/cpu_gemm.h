#ifndef CPU_GEMM_H
#define CPU_GEMM_H

#include <stdint.h>
#include "common.h"

void gemm_rrc_naive(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked_without_packing(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_ccr_blocked_avx(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_to_rrr_blocked_avx(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked_avx(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked_avx_and_omp(void* userdata, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);

#endif
