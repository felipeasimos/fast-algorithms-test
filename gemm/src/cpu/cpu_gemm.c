#include <stdint.h>
#include <string.h>
#include "cpu/cpu_gemm.h"

#define MIN(a, b) a < b ? a : b;
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
