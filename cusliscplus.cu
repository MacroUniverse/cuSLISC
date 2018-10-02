#include "cusliscplus.h"

__global__ void test_kernel()
{
	printf("in block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

//v1 is a vector, size = Nblock
//sum v1 in cpu to get norm2
// __global__
// void norm2_kernel(Doub *v1, Comp *v, Int N)
// {
// 	__shared__ Doub cache[Nth_norm2];
// 	Int i, ind, stride, cacheIdx;
// 	Doub sum{}, temp;

// 	cacheIdx = threadIdx.x;
// 	ind = blockIdx.x * blockDim.x + threadIdx.x;
// 	stride = gridDim.x * blockDim.x;

// 	for (i=ind; i<N; i+=stride) {
// 		temp = abs(v[i]);
// 		sum += temp*temp;
// 	}
// 	cache[threadIdx.x] = sum;
// 	__syncthreads();

// 	// reduction, see CUDA by Example P85
// 	i = blockDim.x/2;
// 	while (i != 0) {
// 		if (cacheIdx < i)
// 			cache[cacheIdx] += cache[cacheIdx+i];
// 		__syncthreads();
// 		i /= 2;
// 	}
// 	if (cacheIdx == 0)
// 		v1[blockIdx.x] = cache[0];
// }
