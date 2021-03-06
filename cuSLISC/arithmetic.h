#pragma once
#include "cuslisc.h"
#include "../SLISC/arithmetic.h"

namespace slisc {

// compare size
template <typename T1, typename T2>
Bool shape_cmp(const Gvector<T1> &v1, const Gvector<T2> &v2)
{ return v1.size() == v2.size(); }

template <typename T1, typename T2>
Bool shape_cmp(const Gmatrix<T1> &a1, const Gmatrix<T2> &a2)
{ return (a1.n1() == a2.n1()) && (a1.n2() == a2.n2()); }

template <typename T1, typename T2>
Bool shape_cmp(const Gmat3d<T1> &a1, const Gmat3d<T2> &a2)
{ return (a1.n1() == a2.n1()) && (a1.n2() == a2.n2()) && (a1.dim3() == a2.dim3()); }

// v += v
template <typename T, typename T1>
__global__ void plus_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] += v1[i];
}

template <typename T, typename T1>
inline void operator+=(Gvector<T> &v, const Gvector<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals0, Nth_plus_equals0, N);
	plus_equals0_kernel<<<Nbl, Nth_plus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <typename T, typename T1>
inline void operator+=(Gmatrix<T> &v, const Gmatrix<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals0, Nth_plus_equals0, N);
	plus_equals0_kernel<<<Nbl, Nth_plus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <typename T, typename T1>
inline void operator+=(Gmat3d<T> &v, const Gmat3d<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals0, Nth_plus_equals0, N);
	plus_equals0_kernel<<<Nbl, Nth_plus_equals0>>>(v.ptr(), v1.ptr(), N);
}

// v -= v
template <typename T, typename T1>
__global__ void minus_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] -= v1[i];
}

template <typename T, typename T1>
inline void operator-=(Gvector<T> &v, const Gvector<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals0, Nth_minus_equals0, N);
	minus_equals0_kernel<<<Nbl, Nth_minus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <typename T, typename T1>
inline void operator-=(Gmatrix<T> &v, const Gmatrix<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals0, Nth_minus_equals0, N);
	minus_equals0_kernel<<<Nbl, Nth_minus_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <typename T, typename T1>
inline void operator-=(Gmat3d<T> &v, const Gmat3d<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals0, Nth_minus_equals0, N);
	minus_equals0_kernel<<<Nbl, Nth_minus_equals0>>>(v.ptr(), v1.ptr(), N);
}

// v *= v
template <typename T, typename T1>
__global__ void times_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] *= v1[i];
}

template <typename T, typename T1>
inline void times_equals0(Gbase<T> &v, const Gbase<T1> &v1)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals0, Nth_times_equals0, N);
	times_equals0_kernel<<<Nbl, Nth_times_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <typename T, typename T1>
inline void operator*=(Gvector<T> &v, const Gvector<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	times_equals0(v, v1);
}

template <typename T, typename T1>
inline void operator*=(Gmatrix<T> &v, const Gmatrix<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	times_equals0(v, v1);
}

template <typename T, typename T1>
inline void operator*=(Gmat3d<T> &v, const Gmat3d<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	times_equals0(v, v1);
}

// v /= v
template <typename T, typename T1>
__global__ void divide_equals0_kernel(T *v, T1 *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] /= v1[i];
}

template <typename T, typename T1>
inline void divide_equals0(Gbase<T> &v, const Gbase<T1> &v1)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_divide_equals0, Nth_divide_equals0, N);
	divide_equals0_kernel<<<Nbl, Nth_divide_equals0>>>(v.ptr(), v1.ptr(), N);
}

template <typename T, typename T1>
inline void operator/=(Gvector<T> &v, const Gvector<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	divide_equals0(v, v1);
}

template <typename T, typename T1>
inline void operator/=(Gmatrix<T> &v, const Gmatrix<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	divide_equals0(v, v1);
}

template <typename T, typename T1>
inline void operator/=(Gmat3d<T> &v, const Gmat3d<T1> &v1)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	divide_equals0(v, v1);
}

// v += s
template <typename T, typename T1>
__global__ void plus_equals1_kernel(T *v, T1 s, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] += s;
}

template <typename T, typename T1>
inline void plus_equals1(Gbase<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus_equals1, Nth_plus_equals1, N);
	plus_equals1_kernel<<<Nbl, Nth_plus_equals1>>>(v.ptr(), s, N);
}

template <typename T>
inline void plus_equals1(Gbase<T> &v, Comp_I &s)
{ plus_equals1(v, toCump(s)); }

template <typename T, typename T1>
inline void operator+=(Gbase<T> &v, const T1 &s)
{ plus_equals1(v, s); }

// template <typename T, typename T1>
// inline void operator+=(Gmatrix<T> &v, const T1 &s)
// { plus_equals1(v, s); }

// template <typename T, typename T1>
// inline void operator+=(Gmat3d<T> &v, const T1 &s)
// { plus_equals1(v, s); }

// v -= s
template <typename T, typename T1>
__global__ void minus_equals1_kernel(T *v, T1 s, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] -= s;
}

template <typename T, typename T1>
inline void minus_equals1(Gbase<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus_equals1, Nth_minus_equals1, N);
	minus_equals1_kernel<<<Nbl, Nth_minus_equals1>>>(v.ptr(), s, N);
}

template <typename T>
inline void minus_equals1(Gbase<T> &v, Comp_I &s)
{ minus_equals1(v, toCump(s)); }

template <typename T, typename T1>
inline void operator-=(Gbase<T> &v, const T1 &s)
{ minus_equals1(v, s); }

// v *= s
template <typename T, typename T1>
__global__ void times_equals1_kernel(T *v, T1 s, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] *= s;
}

template <typename T, typename T1>
inline void times_equals1(Gbase<T> &v, const T1 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_times_equals1, Nth_times_equals1, N);
	times_equals1_kernel<<<Nbl, Nth_times_equals1>>>(v.ptr(), s, N);
}

template <typename T>
inline void times_equals1(Gbase<T> &v, Comp_I &s)
{ times_equals1(v, toCump(s)); }

template <typename T, typename T1>
inline void operator*=(Gbase<T> &v, const T1 &s)
{ times_equals1(v, s); }

// v /= s
// only works for floating point types

template <typename T, typename T1>
inline void operator/=(Gbase<T> &v, const T1 &s)
{ v *= 1./s; }

// plus(v, v1, s), plus(v, s, v1)
template <typename T, typename T1, typename T2> __global__
void plus0_kernel(T *v, const T1 *v1, const T2 s, const Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] = v1[i] + s;
}

template <typename T, typename T1, typename T2>
inline void plus0(Gbase<T> &v, const Gbase<T1> &v1, const T2 &s)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus0, Nth_plus0, N);
	plus0_kernel<<<Nbl, Nth_plus0>>>(v.ptr(), v1.ptr(), s, N);
}

template <typename T, typename T1, typename T2>
inline void plus0(Gbase<T> &v, const Gbase<T1> &v1, Comp_I &s)
{ plus0(v.ptr(), v1.ptr(), toCump(s)); }

template <typename T, typename T1, typename T2>
inline void plus(Gvector<T> &v, const Gvector<T1> &v1, const T2 &s)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	plus0(v, v1, s);
}

template <typename T, typename T1, typename T2>
inline void plus(Gmatrix<T> &v, const Gmatrix<T1> &v1, const T2 &s)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	plus0(v, v1, s);
}

template <typename T, typename T1, typename T2>
inline void plus(Gmat3d<T> &v, const Gmat3d<T1> &v1, const T2 &s)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1))
		SLS_ERR("wrong shape!");
#endif
	plus0(v, v1, s);
}

template <typename T, typename T1, typename T2>
inline void plus(Gvector<T> &v, const T1 &s, const Gvector<T2> &v1)
{ plus(v, v1, s); }

template <typename T, typename T1, typename T2>
inline void plus(Gmatrix<T> &v, const T1 &s, const Gmatrix<T2> &v1)
{ plus(v, v1, s); }

template <typename T, typename T1, typename T2>
inline void plus(Gmat3d<T> &v, const T1 &s, const Gmat3d<T2> &v1)
{ plus(v, v1, s); }

// plus(v, v1, v2)
template <typename T, typename T1, typename T2> __global__
void plus1_kerel(T *v, const T1 *v1, const T2 *v2, const Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] = v1[i] + v2[i];
}

template <typename T, typename T1, typename T2>
inline void plus1(Gbase<T> &v, const Gbase<T1> &v1, const Gbase<T2> &v2)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_plus1, Nth_plus1, N);
	plus1_kerel<<<Nbl,Nth_plus1>>>(v.ptr(), v1.ptr(), v2.ptr(), N);
}

template <typename T, typename T1, typename T2>
inline void plus(Gvector<T> &v, const Gvector<T1> &v1, const Gvector<T2> &v2)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1) || !shape_cmp(v, v2))
		SLS_ERR("wrong shape!");
#endif
	plus1(v, v1, v2);
}

template <typename T, typename T1, typename T2>
inline void plus(Gmatrix<T> &v, const Gmatrix<T1> &v1, const Gmatrix<T2> &v2)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1) || !shape_cmp(v, v2))
		SLS_ERR("wrong shape!");
#endif
	plus1(v, v1, v2);
}

template <typename T, typename T1, typename T2>
inline void plus(Gmat3d<T> &v, const Gmat3d<T1> &v1, const Gmat3d<T2> &v2)
{
#ifdef CUSLS_CHECKBOUNDS
	if (!shape_cmp(v, v1) || !shape_cmp(v, v2))
		SLS_ERR("wrong shape!");
#endif
	plus1(v, v1, v2);
}

//minus(v)
template <typename T> __global__
void minus0_kernel(T *v, const Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] = -v[i];
}

template <typename T>
inline void minus(Gbase<T> &v)
{
	Int N = v.size();
	Int Nbl = nbl(Nbl_minus0, Nth_minus0, N);
	minus0_kernel<<<Nbl,Nth_minus0>>>(v.ptr(), N);
}

//sum v1 in cpu to get total sum, size(v1) = Nblock
template <typename T> __global__
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

template <typename T>
inline T sum(const Gbase<T> &gv)
{
	Long N = gv.size();
	Int Nbl = nbl(Nbl_sum, Nth_sum, N);
	Gvector<T> gv1(Nbl);
	Vector<T> v1(Nbl);
	sum_kernel<<<Nbl, Nth_sum>>>(gv1.ptr(), gv.ptr(), N);
	v1 = gv1;
	return sum(v1);
}

template <>
inline Comp sum(const Gbase<Comp> &gv)
{
	Long N = gv.size();
	Int Nbl = nbl(Nbl_sum, Nth_sum, N);
	Gvector<Comp> gv1(Nbl);
	Vector<Comp> v1(Nbl);
	sum_kernel<<<Nbl, Nth_sum>>>((Cump*)gv1.ptr(), (Cump*)gv.ptr(), N);
	v1 = gv1;
	return sum(v1);
}

//sum v1 in cpu to get norm2, size(v1) = Nblock
// works only for non-complex types
template <typename T> __global__
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

// norm2_kernel for Cump
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

template <typename T>
inline Doub norm2(const Gbase<T> &gv)
{
	Long N = gv.size();
	Int Nbl = nbl(Nbl_norm2, Nth_norm2, N);
	GvecDoub gv1(Nbl);
	VecDoub v1(Nbl);
	norm2_kernel<<<Nbl, Nth_norm2>>>(gv1.ptr(), gv.ptr(), N);
	v1 = gv1;
	return sum(v1);
}

template <typename T>
inline Doub norm(const Gbase<T> &gv)
{ return std::sqrt(norm2(gv)); }

} // namespace slisc
