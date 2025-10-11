#ifndef GPU_GEMM_H
#define GPU_GEMM_H

#include <stdint.h>
#include "common.h"
#include "gpu.h"

void gemm_gpu(GPUData* gpu, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk);

#endif
