#include <stdint.h>
#include <string.h>
#include <omp.h>
#include<immintrin.h>

#include "cpu/cpu_gemm.h"

#define MIN(a, b) a < b ? a : b;
#define MAX(a, b) a < b ? b : a;

#define BLOCKSIZE 64

void gemm_rrc_naive(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk) {
	// C is row major
	// A is row major
	// B is column major
	for(uint32_t i = 0; i < ni; i++) {
		for(uint32_t j = 0; j < nj; j++) {
			for(uint32_t k = 0; k < nk; k++) {
				// B is column major, so this is k row and j column
				C[i * nj + j] += A[i * nk + k] * B[k + j * nk];
			}
		}
	}
}

void gemm_rrc_blocked_without_packing(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk) {
	// C is row major
	// A is row major
	// B is column major
	for(uint32_t bi = 0; bi < ni; bi += BLOCKSIZE) {
		uint32_t I = MIN(BLOCKSIZE, ni - bi);
		for(uint32_t bj = 0; bj < nj; bj += BLOCKSIZE) {
			uint32_t J = MIN(BLOCKSIZE, nj - bj);
			for(uint32_t bk = 0; bk < nk; bk += BLOCKSIZE) {
				uint32_t K = MIN(BLOCKSIZE, nk - bk);
				for(uint32_t ii = 0; ii < I; ii++) {
					for(uint32_t ij = 0; ij < J; ij++) {
						for(uint32_t ik = 0; ik < K; ik++) {
							uint32_t i = bi + ii;
							uint32_t j = bj + ij;
							uint32_t k = bk + ik;
							// B is column major, so this is k row and j column
							C[i * nj + j] += A[i * nk + k] * B[k + j * nk];
						}
					}
				}
			}
		}
	}
}


void gemm_rrc_blocked_with_packing(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk) {
	// C is row major
	// A is row major
	// B is column major
	dtype_t block_a[BLOCKSIZE * BLOCKSIZE];
	dtype_t block_b[BLOCKSIZE * BLOCKSIZE];

	for(uint32_t bi = 0; bi < ni; bi += BLOCKSIZE) {
		uint32_t I = MIN(BLOCKSIZE, ni - bi);
		for(uint32_t bk = 0; bk < nk; bk += BLOCKSIZE) {
			uint32_t K = MIN(BLOCKSIZE, nk - bk);
			// --- Pack A block (row-major) ---
			for(uint32_t ii = 0; ii < I; ii++) {
				memcpy(&block_a[ii * K], &A[(bi + ii) * nk + bk], K * sizeof(dtype_t));
			}
			for(uint32_t bj = 0; bj < nj; bj += BLOCKSIZE) {
				uint32_t J = MIN(BLOCKSIZE, nj - bj);
				// --- Pack B block (column-major) ---
				for(uint32_t ij = 0; ij < J; ij++) {
					memcpy(&block_b[ij * K], &B[bk + (bj + ij) * nk], K * sizeof(dtype_t));
				}
				for(uint32_t ii = 0; ii < I; ii++) {
					for(uint32_t ij = 0; ij < J; ij++) {
						for(uint32_t ik = 0; ik < K; ik++) {
							// B is column major, so this is k row and j column
							C[(bi + ii) * nj + (bj + ij)] += block_a[ii * K + ik] * block_b[ik + ij * K];
						}
					}
				}
			}
		}
	}
}


// void gemm_rrc_blocked_with_packing_and_avx(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk) {
// 	uint8_t n_avx = 32 / sizeof(dtype_t);
// 	// C is row major
// 	// A is row major
// 	// B is column major
// 	dtype_t block_a[BLOCKSIZE * BLOCKSIZE];
// 	dtype_t block_b[BLOCKSIZE * BLOCKSIZE];
//
// 	for(uint32_t bi = 0; bi < ni; bi += BLOCKSIZE) {
// 		uint32_t I = MIN(BLOCKSIZE, ni - bi);
// 		for(uint32_t bk = 0; bk < nk; bk += BLOCKSIZE) {
// 			uint32_t K = MIN(BLOCKSIZE, nk - bk);
// 			// --- Pack A block (row-major) ---
// 			for(uint32_t ii = 0; ii < I; ii++) {
// 				memcpy(&block_a[ii * K], &A[(bi + ii) * nk + bk], K * sizeof(dtype_t));
// 			}
// 			for(uint32_t bj = 0; bj < nj; bj += BLOCKSIZE) {
// 				uint32_t J = MIN(BLOCKSIZE, nj - bj);
// 				// --- Pack B block (column-major) ---
// 				for(uint32_t ij = 0; ij < J; ij++) {
// 					memcpy(&block_b[ij * K], &B[bk + (bj + ij) * nk], K * sizeof(dtype_t));
// 				}
// 				for(uint32_t ii = 0; ii < I; ii++) {
// 					for(uint32_t ij = 0; ij < J; ij++) {
// 						uint64_t ik;
// 						uint32_t c_index = (bi + ii) * nj + (bj + ij);
// 						for ( ik = 0; ik <= nk - n_avx; ik += n_avx ) {
// 							// broadcast A, because doing A * B and reducing to a single C means extra instructions (inefficient)
// 							__m256 a_scalar = _mm256_set1_ps(block_a[ii * K + ik]);
// 							__m256 b_vec = _mm256_loadu_ps(&block_b[ik + ij * K]);
// 							__m256 c_v = _mm256_set1_ps(C[c_index]);
// 							cv = _mm256_fmadd_ps(av, bv, cv);
// 							_mm256_storeu_ps(&C[c_index], cv);
// 						}
// 						for (; ik < K; ik++)  C[c_index] += block_a[ii * K + ik] * block_b[ik + ij * K];    
// 					}
// 				}
// 			}
// 		}
// 	}
// }

#include <stdio.h>
void gemm_rrr_blocked_with_packing_and_avx(dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk) {
	uint8_t n_avx = 32 / sizeof(dtype_t);
	dtype_t block_a[BLOCKSIZE * BLOCKSIZE];
	dtype_t block_b[BLOCKSIZE * BLOCKSIZE];

	for(uint32_t bi = 0; bi < ni; bi += BLOCKSIZE) {
		uint32_t I = MIN(BLOCKSIZE, ni - bi);
		for(uint32_t bk = 0; bk < nk; bk += BLOCKSIZE) {
			uint32_t K = MIN(BLOCKSIZE, nk - bk);
			// --- Pack A block (row-major) ---
			for(uint32_t ii = 0; ii < I; ii++) {
				memcpy(&block_a[ii * K], &A[(bi + ii) * nk + bk], K * sizeof(dtype_t));
			}
			for(uint32_t bj = 0; bj < nj; bj += BLOCKSIZE) {
				uint32_t J = MIN(BLOCKSIZE, nj - bj);
				// --- Pack B block (row-major) ---
				for(uint32_t ik = 0; ik < K; ik++) {
					memcpy(&block_b[ik * J], &B[(bk + ik) * nj + bj], J * sizeof(dtype_t));
				}
				for(uint32_t ii = 0; ii < I; ii++) {
					for(uint32_t ij = 0; ij < J; ij++) {
						uint64_t ik;
						uint32_t c_index = (bi + ii) * nj + (bj + ij);
						uint32_t aligned_K = MIN(K - n_avx, K);
						for ( ik = 0; ik <= aligned_K; ik += n_avx ) {
							printf("b_index: %lu, a_index: %lu, J: %u, K: %u, nj: %u, nk: %u, n_avx: %u, aligned_K: %u\n", ik * J + ij, ii * K + ik, J, K, nj, nk, n_avx, aligned_K);
							__m256 a_scalar = _mm256_set1_ps(block_a[ii * K + ik]);
							__m256 b_vec = _mm256_loadu_ps(&block_b[ik * J + ij]);
							__m256 c_vec = _mm256_loadu_ps(&C[c_index]);
							c_vec = _mm256_fmadd_ps(a_scalar, b_vec, c_vec);
							_mm256_storeu_ps(&C[c_index], c_vec);
						}
						for (; ik < K; ik++)  C[c_index] += block_a[ii * K + ik] * block_b[ik + ij * K];    
					}
				}
			}
		}
	}
}
