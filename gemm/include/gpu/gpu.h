#ifndef GPU_H
#define GPU_H

#include "webgpu.h"

typedef struct {
	WGPUInstance instance;
	WGPUAdapter adapter;
	WGPUDevice device;
	WGPUQueue queue;
	double time;
} GPUData;

GPUData initGPUData(void);
void freeGPUData(GPUData gpu_data);

#endif
