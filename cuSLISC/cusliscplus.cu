#include "cusliscplus.h"

__global__ void test_kernel()
{
	printf("in block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

//sum v1 in cpu to get norm2, size(v1) = Nblock
__global__
void norm2_kernel(Doub *v1, Cump_I *v, Long N)
{
	__shared__ Doub cache[Nth_sum];
	Long i, ind, stride, cacheIdx;
	Doub s = 0.;
	cacheIdx = threadIdx.x;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;

	for (i=ind; i<N; i+=stride)
		s += norm(v[i]);
	
	cache[cacheIdx] = s;
	__syncthreads();

	// reduction
	i = blockDim.x/2;
	while (i != 0) {
		if (cacheIdx < i)
			cache[cacheIdx] += cache[cacheIdx+i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIdx == 0)
		v1[blockIdx.x] = cache[0];
}
