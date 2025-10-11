#include "gpu/gpu.h"
#include <stdio.h>

// Callback for device request
void onDeviceRequest(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, WGPU_NULLABLE void* device_ptr, WGPU_NULLABLE void* signal_ptr) {
	if (status != WGPURequestDeviceStatus_Success) {
		fprintf(stderr, "Failed to create device: %s\n", message.data);
		*((WGPUDevice*)device_ptr) = NULL;
		return;
	}
	*((WGPUDevice*)device_ptr) = device;
	*(int*)signal_ptr = 0;
	printf("created device\n");
}

// Callback for adapter request
void onAdapterRequest(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, WGPU_NULLABLE void* adapter_ptr, WGPU_NULLABLE void* signal_ptr) {
	if (status != WGPURequestAdapterStatus_Success) {
		fprintf(stderr, "Failed to get adapter: %s\n", message.data);
		*((WGPUAdapter*)adapter_ptr) = NULL;
		return;
	}
	*((WGPUAdapter*)adapter_ptr) = adapter;
	*(int*)signal_ptr = 0;
	printf("created adapter\n");
}

GPUData initGPUData(void) {
	// Create instance
	WGPUInstance instance = wgpuCreateInstance(&(WGPUInstanceDescriptor){0});

	// Request adapter
	WGPUAdapter adapter = {0};
	WGPUDevice device = {0};
	WGPUQueue queue = {0};

	int signal = 1;
	WGPURequestAdapterOptions options = {
		.powerPreference = WGPUPowerPreference_HighPerformance,
		.compatibleSurface = NULL, // compute only
	};
	WGPURequestAdapterCallbackInfo adapter_callback_info = {
		.callback = onAdapterRequest,
		.userdata1 = &adapter,
		.userdata2 = &signal,
	};
	wgpuInstanceRequestAdapter(instance, &options, adapter_callback_info);

	// wait for adapter
	while(signal) {
		wgpuInstanceProcessEvents(instance);
	};
	signal = 1;

	const WGPUDeviceDescriptor desc = {0};
	WGPURequestDeviceCallbackInfo device_callback_info = {
		.callback = onDeviceRequest,
		.userdata1 = &device,
		.userdata2 = &signal,
	};
	wgpuAdapterRequestDevice(adapter, &desc, device_callback_info);
	while(signal) {
		wgpuInstanceProcessEvents(instance);
	};
	signal = 1;

	queue = wgpuDeviceGetQueue(device);

	printf("Device and queue are ready!\n");

	GPUData data = {
		.device = device,
		.adapter = adapter,
		.instance = instance,
	};
	return data;
}

void freeGPUData(GPUData gpu_data) {
	// Cleanup
	wgpuDeviceRelease(gpu_data.device);
	wgpuAdapterRelease(gpu_data.adapter);
	wgpuInstanceRelease(gpu_data.instance);
}
