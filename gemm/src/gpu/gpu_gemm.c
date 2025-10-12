#include "gpu/gpu_gemm.h"
#include "common.h"
#include "gpu/webgpu.h"
#include <string.h>
#include <stdio.h>
#include <omp.h>

typedef struct {
	dtype_t* C;
	uint64_t size;
	int* signal;
} ReadbackData;

void onBufferMapped(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* _) {
	ReadbackData* data = (ReadbackData*)userdata1;
	if (status != WGPUMapAsyncStatus_Success) {
		fprintf(stderr, "Buffer mapping failed: %s\n", message.data);
		*(data->signal) = 2;
		return;
	}
	*(data->signal) = 0;
}

void waitForGPUWork(WGPUQueueWorkDoneStatus status, WGPU_NULLABLE void* signal_ptr, WGPU_NULLABLE void* _) {
	if (status != WGPUQueueWorkDoneStatus_Success) {
		fprintf(stderr, "Error on wgpu: %u\n", status);
		*(int*)signal_ptr = 2;
		return;
	}
	*(int*)signal_ptr = 0;
}
void gemm_gpu(GPUData* gpu, dtype_t* C, dtype_t* A, dtype_t* B, uint32_t ni, uint32_t nj, uint32_t nk) {
	// sizes in bytes
	const uint64_t a_size = (uint64_t)ni * nk * sizeof(dtype_t);
	const uint64_t b_size = (uint64_t)nk * nj * sizeof(dtype_t);
	const uint64_t c_size = (uint64_t)ni * nj * sizeof(dtype_t);

	// Create GPU buffers (device-local usage + copy-dst for uploads)
	WGPUBuffer a_buf = wgpuDeviceCreateBuffer(gpu->device, &(WGPUBufferDescriptor){
		.size = a_size,
		.usage = (WGPUBufferUsage)(WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage),
		.label = {"a_buffer", WGPU_STRLEN},
		.mappedAtCreation = 0,
	});

	WGPUBuffer b_buf = wgpuDeviceCreateBuffer(gpu->device, &(WGPUBufferDescriptor){
		.size = b_size,
		.usage = (WGPUBufferUsage)(WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage),
		.label = {"b_buffer", WGPU_STRLEN},
		.mappedAtCreation = 0,
	});

	WGPUBuffer c_buf = wgpuDeviceCreateBuffer(gpu->device, &(WGPUBufferDescriptor){
		.size = c_size,
		.usage = (WGPUBufferUsage)(WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage),
		.label = {"c_buffer", WGPU_STRLEN},
		.mappedAtCreation = 0, // make sure the buffer is zero-initialized
	});
	WGPUBuffer staging_buffer = wgpuDeviceCreateBuffer(gpu->device, &(WGPUBufferDescriptor){
		.size = c_size,
		.usage = (WGPUBufferUsage)(WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead),
		.label = {"staging_buffer", WGPU_STRLEN},
		.mappedAtCreation = 0, // make sure the buffer is zero-initialized
	});

	// Upload A and B via Queue::writeBuffer (uses an internal staging buffer)
	wgpuQueueWriteBuffer(gpu->queue, a_buf, 0, A, (size_t)a_size);
	wgpuQueueWriteBuffer(gpu->queue, b_buf, 0, B, (size_t)b_size);

	// Uniform buffer with dims (pad to 16 bytes)
	uint32_t dims_data[4] = { ni, nj, nk, 0u }; // last element is padding
	WGPUBufferDescriptor dims_desc = {
		.size = sizeof(dims_data),
		.usage = (WGPUBufferUsage)(WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform),
		.label = {"dims_data", WGPU_STRLEN},
		.mappedAtCreation = 0,
	};
	WGPUBuffer dims_buf = wgpuDeviceCreateBuffer(gpu->device, &dims_desc);
	wgpuQueueWriteBuffer(gpu->queue, dims_buf, 0, dims_data, sizeof(dims_data));

	// WGSL compute shader (simple naive matmul)
	const char* wgsl_code = 
		"struct Dims { n: u32, m: u32, k: u32, pad: u32, };\n"
		"@group(0) @binding(0) var<storage, read> A: array<f32>;\n"
		"@group(0) @binding(1) var<storage, read> B: array<f32>;\n"
		"@group(0) @binding(2) var<storage, read_write> C: array<f32>;\n"
		"@group(0) @binding(3) var<uniform> dims: Dims;\n"
		"@compute @workgroup_size(16,16)\n"
		"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
		"  let row = gid.y;\n"
		"  let col = gid.x;\n"
		"  if (row >= dims.n || col >= dims.m) { return; }\n"
		"  var sum: f32 = 0.0;\n"
		"  var k: u32 = 0u;\n"
		"  loop {\n"
		"    if (k >= dims.k) { break; }\n"
		"    let a = A[row * dims.k + k];\n"
		"    let b = B[k * dims.m + col];\n"
		"    sum = sum + a * b;\n"
		"    k = k + 1u;\n"
		"  }\n"
		"  C[row * dims.m + col] = sum;\n"
		"}\n";

	WGPUShaderModule shader = wgpuDeviceCreateShaderModule(
		gpu->device,
		&(const WGPUShaderModuleDescriptor){
			.label = {"shader_module_descriptor", WGPU_STRLEN},
			.nextInChain = (const WGPUChainedStruct *)&(const WGPUShaderSourceWGSL){
				.chain = (const WGPUChainedStruct){
					.sType = WGPUSType_ShaderSourceWGSL,
				},
				.code = {wgsl_code, WGPU_STRLEN},
			}
                }
	);

	WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(
		gpu->device,
		&(const WGPUComputePipelineDescriptor){
			.label = {"compute_pipeline", WGPU_STRLEN},
			.compute = (const WGPUProgrammableStageDescriptor){
				.module = shader,
				.entryPoint = {"main", WGPU_STRLEN},
			}
		}
	);
	WGPUBindGroupLayout bind_group_layout = wgpuComputePipelineGetBindGroupLayout(compute_pipeline, 0);
	WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
		gpu->device,
		&(const WGPUBindGroupDescriptor){
			.label = {"bind_group", WGPU_STRLEN},
			.layout = bind_group_layout,
			.entryCount = 4,
			.entries = (const WGPUBindGroupEntry[]){
				(const WGPUBindGroupEntry){
					.nextInChain = NULL,
					.binding = 0,
					.buffer = a_buf,
					.offset = 0,
					.size = a_size,
					.sampler = NULL,
					.textureView = NULL,
				},
				(const WGPUBindGroupEntry){
					.nextInChain = NULL,
					.binding = 1,
					.buffer = b_buf,
					.offset = 0,
					.size = b_size,
					.sampler = NULL,
					.textureView = NULL,
				},
				(const WGPUBindGroupEntry){
					.nextInChain = NULL,
					.binding = 2,
					.buffer = c_buf,
					.offset = 0,
					.size = c_size,
					.sampler = NULL,
					.textureView = NULL,
				},
				(const WGPUBindGroupEntry){
					.nextInChain = NULL,
					.binding = 3,
					.buffer = dims_buf,
					.offset = 0,
					.size = sizeof(dims_data),
					.sampler = NULL,
					.textureView = NULL,
				}
			}
		});
	WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(
		gpu->device,
		&(const WGPUCommandEncoderDescriptor){
			.label = {"command_encoder", WGPU_STRLEN},
		}
	);
	WGPUComputePassEncoder compute_pass_encoder = wgpuCommandEncoderBeginComputePass(
		command_encoder,
		&(const WGPUComputePassDescriptor){
			.label = {"compute_pass", WGPU_STRLEN}
		}
	);
	wgpuComputePassEncoderSetPipeline(compute_pass_encoder, compute_pipeline);
	wgpuComputePassEncoderSetBindGroup(compute_pass_encoder, 0, bind_group, 0, NULL);
	wgpuComputePassEncoderDispatchWorkgroups(compute_pass_encoder, 256, 1, 1);
	wgpuComputePassEncoderEnd(compute_pass_encoder);
	wgpuComputePassEncoderRelease(compute_pass_encoder);
	wgpuCommandEncoderCopyBufferToBuffer(command_encoder, c_buf, 0, staging_buffer, 0, c_size);

	WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(
		command_encoder,
		&(const WGPUCommandBufferDescriptor){
                        .label = {"command_buffer", WGPU_STRLEN},
                });
	double start = omp_get_wtime();
	wgpuQueueSubmit(gpu->queue, 1, &command_buffer);
	int signal = 1;
	wgpuQueueOnSubmittedWorkDone(gpu->queue, (WGPUQueueWorkDoneCallbackInfo){
		.callback = waitForGPUWork,
		.userdata1 = &signal,
		.userdata2 = NULL,
	});
	while(signal) {
		wgpuInstanceProcessEvents(gpu->instance);
	}
	double stop = omp_get_wtime();
	gpu->time = stop-start;
	ReadbackData readback = {.C = C, .size = c_size, .signal = &signal};
	signal = 1;
	wgpuBufferMapAsync(staging_buffer, WGPUMapMode_Read, 0, c_size, (WGPUBufferMapCallbackInfo){
		.callback = onBufferMapped,
		.userdata1 = &readback,
	});
	while(signal) {
		wgpuInstanceProcessEvents(gpu->instance);
	}
	if(signal == 0) {
		const void* mapped_data = wgpuBufferGetConstMappedRange(staging_buffer, 0, c_size);
		memcpy(C, mapped_data, c_size);
	}
	wgpuBufferUnmap(staging_buffer);
	wgpuBufferRelease(a_buf);
	wgpuBufferRelease(b_buf);
	wgpuBufferRelease(c_buf);
	wgpuBufferRelease(staging_buffer);
	wgpuBufferRelease(dims_buf);
	wgpuShaderModuleRelease(shader);
	wgpuComputePipelineRelease(compute_pipeline);
	wgpuBindGroupLayoutRelease(bind_group_layout);
	wgpuBindGroupRelease(bind_group);
	wgpuCommandEncoderRelease(command_encoder);
	wgpuCommandBufferRelease(command_buffer);
}
