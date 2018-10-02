#pragma once
#include "cuslisc.h"
using std::cout; using std::endl;

__global__ void test_kernel();

template <class T, class T1, class T2>
__global__ void plus0_kerel(T *v, T1 *v1, T2 *v2, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] = v1[i] + v2[i];
}

template <class T, class T1, class T2>
inline void plus(CUbase<T> &v, const CUbase<T1> &v1, const CUbase<T2> &v2)
{
	plus0_kerel<<<2,32>>>(v.ptr(), v1.ptr(), v2.ptr(), v.size());
}

//sum v1 in cpu to get total sum, size(v1) = Nblock
template <class T>
__global__
void sum_kernel(T *v1, const T *v, Long N)
{
	__shared__ T cache[Nth_sum];
	Long i, ind, stride, cacheIdx;
	T s{};

	cacheIdx = threadIdx.x;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;

	for (i=ind; i<N; i+=stride)
		s += v[i];
	
	cache[threadIdx.x] = s;
	__syncthreads();

	// reduction, see CUDA by Example P85
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

template <class T>
inline T sum(const CUbase<T> &gv)
{
	Long N = gv.size();
	Int Nbl = nbl(Nbl_sum, Nth_sum, N);
	CUvector<T> gv1(Nbl);
	NRvector<T> v1(Nbl);
	cout << "sum() block num = " << nbl(Nbl_sum, Nth_sum, N) << endl;
	cout << "sum() thread num = " << Nth_sum << endl;
	sum_kernel<<<Nbl, Nth_sum>>>(gv1.ptr(), gv.ptr(), N);
	gv1.get(v1);
	disp(v1);
	return sum(v1);
}

//sum(abs(v(:)). ^ 2) for complex numbers
// inline Doub norm2(CUbase<Comp> &v)
// {	
// 	Int N = v.size();
// 	//Int Nth = 32;
// 	//Int Nbl = 320;
// 	//Int Nbl = min(320, (N + Nth - 1)/Nth);
// 	GvecDoub gv1(Nbl_norm2); VecDoub v1(Nbl_norm2);
// 	cunorm2_kernel<<<Nbl_norm2, Nth_norm2>>>(v1.ptr(), v.ptr(), v.size());
// 	gv1.get(v1);
// 	return sum(v1);
// }
