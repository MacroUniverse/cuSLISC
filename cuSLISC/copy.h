#pragma once
#include "global.h"
namespace slisc {

// set elements to same value
template <typename T> __global__
void cumemset(T *dest, const T val, Long_I n)
{
	Int i, ind, stride;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i = ind; i < n; i += stride)
		dest[i] = val;
}

// copy elements using kernel instead of cudaMemcpyDeviceToDevice
// advantage unknown
template <typename T> __global__
void cumemcpy(T *dest, const T *src, Long_I n)
{
	Int i, ind, stride;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i = ind; i < n; i += stride)
		dest[i] = src[i];
}

} // namespace slisc
