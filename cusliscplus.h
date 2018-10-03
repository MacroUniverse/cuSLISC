#pragma once
#include "cuslisc.h"
using std::cout; using std::endl;

// compare size
template <class T1, class T2>
Bool shape_cmp(const CUvector<T1> &v1, const CUvector<T2> &v2)
{ return v1.size() == v2.size(); }

template <class T1, class T2>
Bool shape_cmp(const CUmatrix<T1> &a1, const CUmatrix<T2> &a2)
{ return (a1.nrows() == a2.nrows()) && (a1.ncols() == a2.ncols()); }

template <class T1, class T2>
Bool shape_cmp(const CUmat3d<T1> &a1, const CUmat3d<T2> &a2)
{ return (a1.dim1() == a2.dim1()) && (a1.dim2() == a2.dim2()) && (a1.dim3() == a2.dim3()); }

// a very simple kernel
__global__ void test_kernel();

// v += v
template <class T, class T1>
__global__ void plus_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] += v1[i];
}

template <class T, class T1>
inline void operator+=(CUvector<T> &v, const CUvector<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals0, Nth_plus_equals0, N);
	plus_equals0_kernel<<<Nbl, Nth_plus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator+=(CUmatrix<T> &v, const CUmatrix<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals0, Nth_plus_equals0, N);
	plus_equals0_kernel<<<Nbl, Nth_plus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator+=(CUmat3d<T> &v, const CUmat3d<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals0, Nth_plus_equals0, N);
	plus_equals0_kernel<<<Nbl, Nth_plus_equals0>>>(v.ptr(), v1.ptr(), N);
}

// v -= v
template <class T, class T1>
__global__ void minus_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] -= v1[i];
}

template <class T, class T1>
inline void operator-=(CUvector<T> &v, const CUvector<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals0, Nth_minus_equals0, N);
	minus_equals0_kernel<<<Nbl, Nth_minus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator-=(CUmatrix<T> &v, const CUmatrix<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals0, Nth_minus_equals0, N);
	minus_equals0_kernel<<<Nbl, Nth_minus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator-=(CUmat3d<T> &v, const CUmat3d<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals0, Nth_minus_equals0, N);
	minus_equals0_kernel<<<Nbl, Nth_minus_equals0>>>(v.ptr(), v1.ptr(), N);
}

// v *= v
template <class T, class T1>
__global__ void times_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] *= v1[i];
}

template <class T, class T1>
inline void operator*=(CUvector<T> &v, const CUvector<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals0, Nth_times_equals0, N);
	times_equals0_kernel<<<Nbl, Nth_times_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator*=(CUmatrix<T> &v, const CUmatrix<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals0, Nth_times_equals0, N);
	times_equals0_kernel<<<Nbl, Nth_times_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator*=(CUmat3d<T> &v, const CUmat3d<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals0, Nth_times_equals0, N);
	times_equals0_kernel<<<Nbl, Nth_times_equals0>>>(v.ptr(), v1.ptr(), N);
}

// v /= v
template <class T, class T1>
__global__ void divide_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] /= v1[i];
}

template <class T, class T1>
inline void operator/=(CUvector<T> &v, const CUvector<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals0, Nth_divide_equals0, N);
	divide_equals0_kernel<<<Nbl, Nth_divide_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator/=(CUmatrix<T> &v, const CUmatrix<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals0, Nth_divide_equals0, N);
	divide_equals0_kernel<<<Nbl, Nth_divide_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <class T, class T1>
inline void operator/=(CUmat3d<T> &v, const CUmat3d<T1> &v1)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1)) error("wrong shape!")
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals0, Nth_divide_equals0, N);
	divide_equals0_kernel<<<Nbl, Nth_divide_equals0>>>(v.ptr(), v1.ptr(), N);
}

// v += s
template <class T, class T1>
__global__ void plus_equals1_kernel(T *v, T1 s, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] += s;
}

template <class T, class T1>
inline void operator+=(CUvector<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals1, Nth_plus_equals1, N);
	plus_equals1_kernel<<<Nbl, Nth_plus_equals1>>>(v.ptr(), s, N);
}

template <class T, class T1>
inline void operator+=(CUmatrix<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals1, Nth_plus_equals1, N);
	plus_equals1_kernel<<<Nbl, Nth_plus_equals1>>>(v.ptr(), s, N);
}

template <class T, class T1>
inline void operator+=(CUmat3d<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals1, Nth_plus_equals1, N);
	plus_equals1_kernel<<<Nbl, Nth_plus_equals1>>>(v.ptr(), s, N);
}

// v -= s
template <class T, class T1>
__global__ void minus_equals1_kernel(T *v, T1 s, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] -= s;
}

template <class T, class T1>
inline void operator-=(CUvector<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals1, Nth_minus_equals1, N);
	minus_equals1_kernel<<<Nbl, Nth_minus_equals1>>>(v.ptr(), s, N);
}

template <class T, class T1>
inline void operator-=(CUmatrix<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals1, Nth_minus_equals1, N);
	minus_equals1_kernel<<<Nbl, Nth_minus_equals1>>>(v.ptr(), s, N);
}

template <class T, class T1>
inline void operator-=(CUmat3d<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals1, Nth_minus_equals1, N);
	minus_equals1_kernel<<<Nbl, Nth_minus_equals1>>>(v.ptr(), s, N);
}

// v *= s
template <class T, class T1>
__global__ void times_equals1_kernel(T *v, T1 s, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] *= s;
}

template <class T, class T1>
inline void operator*=(CUvector<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals1, Nth_times_equals1, N);
	times_equals1_kernel<<<Nbl, Nth_times_equals1>>>(v.ptr(), s, N);
}

template <class T, class T1>
inline void operator*=(CUmatrix<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals1, Nth_times_equals1, N);
	times_equals1_kernel<<<Nbl, Nth_times_equals1>>>(v.ptr(), s, N);
}

template <class T, class T1>
inline void operator*=(CUmat3d<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals1, Nth_times_equals1, N);
	times_equals1_kernel<<<Nbl, Nth_times_equals1>>>(v.ptr(), s, N);
}

// v /= s
// only works for floating point types
template <class T, class T1>
__global__ void divide_equals1_kernel(T *v, T1 sInv, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] *= sInv;
}

template <class T, class T1>
inline void operator/=(CUvector<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals1, Nth_divide_equals1, N);
	divide_equals1_kernel<<<Nbl, Nth_divide_equals1>>>(v.ptr(), 1./s, N);
}

template <class T, class T1>
inline void operator/=(CUmatrix<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals1, Nth_divide_equals1, N);
	divide_equals1_kernel<<<Nbl, Nth_divide_equals1>>>(v.ptr(), 1./s, N);
}

template <class T, class T1>
inline void operator/=(CUmat3d<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals1, Nth_divide_equals1, N);
	divide_equals1_kernel<<<Nbl, Nth_divide_equals1>>>(v.ptr(), 1./s, N);
}

template <class T, class T1, class T2>
__global__ void plus1_kerel(T *v, T1 *v1, T2 *v2, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] = v1[i] + v2[i];
}

template <class T, class T1, class T2>
inline void plus(CUvector<T> &v, const CUvector<T1> &v1, const CUvector<T2> &v2)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1) || !shape_cmp(v, v2)) error("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus, Nth_plus, N);
	plus1_kerel<<<Nbl,Nth_plus>>>(v.ptr(), v1.ptr(), v2.ptr(), N);
}

template <class T, class T1, class T2>
inline void plus(CUmatrix<T> &v, const CUmatrix<T1> &v1, const CUmatrix<T2> &v2)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1) || !shape_cmp(v, v2)) error("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus, Nth_plus, N);
	plus1_kerel<<<Nbl,Nth_plus>>>(v.ptr(), v1.ptr(), v2.ptr(), N);
}

template <class T, class T1, class T2>
inline void plus(CUmat3d<T> &v, const CUmat3d<T1> &v1, const CUmat3d<T2> &v2)
{
#ifdef _CHECKBOUNDS_
	if (!shape_cmp(v, v1) || !shape_cmp(v, v2)) error("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus, Nth_plus, N);
	plus1_kerel<<<Nbl,Nth_plus>>>(v.ptr(), v1.ptr(), v2.ptr(), N);
}

//sum v1 in cpu to get total sum, size(v1) = Nblock
template <class T> __global__
void sum_kernel(T *v1, const T *v, Long N)
{
	__shared__ T cache[Nth_sum];
	Long i, ind, stride, cacheIdx;
	T s; s = 0.;
	cacheIdx = threadIdx.x;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;

	for (i=ind; i<N; i+=stride)
		s += v[i];
	cache[cacheIdx] = s;
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
	sum_kernel<<<Nbl, Nth_sum>>>(gv1.ptr(), gv.ptr(), N);
	gv1.get(v1);
	return sum(v1);
}

//sum v1 in cpu to get norm2, size(v1) = Nblock
// works only for non-complex types
template <class T> __global__
void norm2_kernel(T *v1, const T *v, Long N)
{
	__shared__ T cache[Nth_sum];
	Long i, ind, stride, cacheIdx;
	T s{}, temp;
	cacheIdx = threadIdx.x;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;

	for (i=ind; i<N; i+=stride) {
		temp = v[i];
		s += temp*temp;
	}
	
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

// norm2 for Comp
__global__ void norm2_kernel(Doub *v1, const Comp *v, Long N);

template <class T>
inline Doub norm2(const CUbase<T> &gv)
{
	Long N = gv.size();
	Int Nbl = nbl(Nbl_norm2, Nth_norm2, N);
	GvecDoub gv1(Nbl);
	VecDoub v1(Nbl);
	norm2_kernel<<<Nbl, Nth_norm2>>>(gv1.ptr(), gv.ptr(), N);
	gv1.get(v1);
	return sum(v1);
}
