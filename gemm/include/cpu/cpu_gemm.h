#ifndef GEMM_H
#define GEMM_H

#include <stdint.h>

typedef float dtype_t;

void gemm_rrc_naive(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked_without_packing(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_ccr_blocked_avx(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_to_rrr_blocked_avx(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked_avx(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);
void gemm_rrc_blocked_avx_and_omp(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);

#endif
