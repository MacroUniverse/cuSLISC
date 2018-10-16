// classes for cuda matrix
#pragma once
#include "../SLISC/slisc.h"
#include "blocks_threads.h"
#include "complex.h"
//#include "complex/complex.h"

namespace slisc {

// complex type for cuda kernel
// should not be used in host code
typedef const slisc::complex<Doub> Cump_I;
typedef slisc::complex<Doub> Cump, Cump_O, Cump_IO;

inline Cump& toCump(Comp &s) { return reinterpret_cast<Cump&>(s); }
inline Cump_I & toCump(Comp_I &s) { return reinterpret_cast<Cump_I&>(s); }

// specialized overloading of operator== for Comp
// so that "CUref<Comp>" can be implicitly converted to "Comp" when used as argument
// because template version in STL does not support such conversion
inline Bool operator==(Comp_I &s1, Comp_I &s2)
{ return real(s1)==real(s2) && imag(s1)==imag(s2); }

inline Bool operator!=(Comp_I &s1, Comp_I &s2)
{ return !(s1 == s2); } // this is the STL implementation

// these should never be needed
// inline Bool operator==(Comp_I &s1, Cump_I &s2)
// { return toCump(s1) == s2; }
// inline Bool operator==(Cump_I &s1, Comp_I &s2) { return s2 == s1; }
// inline Bool operator!=(Comp_I &s1, Cump_I &s2)
// { return toCump(s1) != s2; }
// inline Bool operator!=(Cump_I &s1, Comp_I &s2) { return s2 != s1; }

// get device global variable
template <typename T>
inline T getsym(const T &sym)
{
	T val;
	cudaMemcpyFromSymbol(&val, sym, sizeof(T));
	return val;
}

inline Comp getsym(Cump_I &sym)
{
	Comp val;
	cudaMemcpyFromSymbol(&val, sym, sizeof(Comp));
	return val;
}

// set device global variable

// this might be unnecessary
// template <typename T, typename T1>
// inline void setsym(T &sym, const T1 &val)
// {
// 	T val1; val1 = (T)val;
// 	cudaMemcpyToSymbol(sym, &val1, sizeof(T));
// #ifdef _CHECKSETSYS_
// 	if (getsym(sym) != val1) error("failed!");
// #endif
// }

template <typename T>
inline void setsym(T &sym, const T &val)
{
	cudaMemcpyToSymbol(sym, &val, sizeof(T));
#ifdef _CHECKSETSYS_
	if (getsym(sym) != val) error("failed!");
#endif
}

inline void setsym(Cump &sym, Comp_I &val)
{
	cudaMemcpyToSymbol(sym, &val, sizeof(Comp));
#ifdef _CHECKSETSYS_
	if ( getsym(sym) != val ) error("failed!");
#endif
}

// calculate number of CUDA blocks needed
inline Int nbl(Int NblMax, Int Nth, Int N)
{ return min(NblMax, (N + Nth - 1)/Nth); }

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

// convert from host type to corresponding device type
template <typename T>
class KerT
{ public: typedef T type; };

template<>
class KerT<Comp>
{ public: typedef Cump type; };

// reference type to CUbase element
template <typename T>
class CUref
{
protected:
	typedef typename KerT<T>::type T1;
	T1 * p;
public:
	CUref() {};
	explicit CUref(T1* ptr) : p(ptr) {}
	void bind(T1* ptr) { p = ptr; };
	T1* ptr() { return p; }
	const T1* ptr() const { return p; }
	inline operator T() const;
	inline CUref& operator=(const T& rhs);
	template <typename T1>
	inline CUref& operator+=(const T1& rhs) { *this = T(*this) + rhs; return *this; }
	template <typename T1>
	inline CUref& operator-=(const T1& rhs) { *this = T(*this) - rhs; return *this; }
	template <typename T1>
	inline CUref& operator*=(const T1& rhs) { *this = T(*this) * rhs; return *this; }
	template <typename T1>
	inline CUref& operator/=(const T1& rhs) { *this = T(*this) / rhs; return *this; }
};

template <typename T>
inline CUref<T>::operator T() const
{
	T val;
	cudaMemcpy(&val, p, sizeof(T), cudaMemcpyDeviceToHost);
	return val;
}

template <typename T>
inline CUref<T>& CUref<T>::operator=(const T& rhs)
{
	cudaMemcpy(p, &rhs, sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

// pointer type to CUbase element
// note : const CUptr is top level const
// TODO : create a class for low level const, or is CUptr<const T> a low level const??
template <typename T>
class CUptr
{
protected:
	typedef typename KerT<T>::type T1;
	T1* p;
public:
	CUptr() : p(nullptr) {}
	CUptr(T1 *ptr) : p(ptr) {}
	T1* ptr() const { return p; }
	inline CUref<T> & operator*() const; // dereference
	CUref<T> operator[](Long_I i) const { return CUref<T>(p+i); }
	CUptr & operator=(const CUptr &rhs) { p = rhs.ptr(); return *this; } // copy assignment
	CUptr & operator=(T1* ptr) { p = ptr; return *this; } // T1* assignment
	void operator+=(Long_I i) { p += i; }
	void operator-=(Long_I i) { p -= i; }
};

template <typename T>
inline CUref<T> & CUptr<T>::operator*() const
{ return reinterpret_cast<CUref<T>&>(*const_cast<CUptr<T>*>(this)); }

template <typename T>
inline CUptr<T> operator+(const CUptr<T> &pcu, Long_I i) { return CUptr<T>(pcu.ptr()+i); }

template <typename T>
inline CUptr<T> operator-(const CUptr<T> &pcu, Long_I i) { return CUptr<T>(pcu.ptr()-i); }

// scalar class
template <typename T>
class CUscalar : public CUref<T>
{
private:
	typedef typename KerT<T>::type T1;
public:
	typedef CUref<T> Base;
	using Base::p;
	using Base::operator=;
	CUscalar() { cudaMalloc(&p, sizeof(T)); }
	explicit CUscalar(const T &s) : CUscalar() { *this = s; }
};

// base class for CUvector, CUmatrix, CUmat3d
template <typename T>
class CUbase
{
private:
	// it is weired that I cannot use "kernel_type*" instead of "p_kernel_type", although they have the same typeid!
	typedef typename KerT<T>::type kernel_type;
	typedef typename KerT<T>::type* p_kernel_type;
	typedef const typename KerT<T>::type* p_const_kernel_type;
protected:
	p_kernel_type p; // pointer to the first element
	Long N; // number of elements
public:
	CUbase() : N(0), p(nullptr) {}
	explicit CUbase(Long_I n) : N(n) { cudaMalloc(&p, N*sizeof(T)); }
	p_kernel_type ptr() { return p; } // get pointer
	p_const_kernel_type ptr() const { return p; }
	Long_I size() const { return N; }
	inline void resize(Long_I n);
	inline CUref<T> operator()(Long_I i);
	inline const CUref<T> operator()(Long_I i) const;
	inline CUref<T> end(); // last element
	inline const CUref<T> end() const;
	inline CUbase & operator=(const T &rhs); // set scalar
	~CUbase() { if (p) cudaFree(p); }
};

template <typename T>
inline void CUbase<T>::resize(Long_I n)
{
	if (n != N) {
		if (p != nullptr) cudaFree(p);
		N = n;
		if (n > 0)
			cudaMalloc(&p, N*sizeof(T));
		else
			p = nullptr;
	}
}

template <typename T>
inline CUref<T> CUbase<T>::operator()(Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUbase subscript out of bounds!");
#endif
	return CUref<T>(p+i);
}

template <typename T>
inline const CUref<T> CUbase<T>::operator()(Long_I i) const
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUbase subscript out of bounds!");
#endif
	return CUref<T>(p+i);
}

template <typename T>
inline CUref<T> CUbase<T>::end()
{
#ifdef _CHECKBOUNDS_
	if (N < 1)
		error("Using end() for empty object!");
#endif
	return CUref<T>(p+N-1);
}

template <typename T>
inline const CUref<T> CUbase<T>::end() const
{
#ifdef _CHECKBOUNDS_
	if (N < 1)
		error("Using end() for empty object!");
#endif
	return CUref<T>(p+N-1);
}

template <typename T>
inline CUbase<T> & CUbase<T>::operator=(const T &rhs)
{
	if (N) cumemset<<<nbl(Nbl_cumemset,Nth_cumemset,N), Nth_cumemset>>>(p, (kernel_type&)rhs, N);
	return *this;
}

// Vector Class

template <typename T>
class CUvector : public CUbase<T>
{
public:
	typedef CUbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	CUvector() {};
	explicit CUvector(Long_I n) : Base(n) {}
	CUvector(Long_I n, const T &a);	//initialize to constant value
	CUvector(NRvector<T> &v); // initialize from cpu vector
	CUvector(const CUvector &rhs);	// Copy constructor forbidden
	template <typename T1>
	inline void get(NRvector<T1> &v) const; // copy to cpu vector
	inline CUvector & operator=(const CUvector &rhs);	// copy assignment
	inline CUvector & operator=(const NRvector<T> &v); // NR assignment
	inline CUref<T> operator[](Long_I i); //i'th element
	inline const CUref<T> operator[](Long_I i) const;
	inline void resize(Long_I n); // resize (contents not preserved)
	template <typename T1>
	inline void resize(const CUvector<T1> &v);
	template <typename T1>
	inline void resize(const NRvector<T1> &v);
};

template <typename T>
CUvector<T>::CUvector(Long_I n, const T &a) : CUvector(n)
{ *this = a; }

template <typename T>
CUvector<T>::CUvector(NRvector<T> &v) : CUvector(v.size())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
CUvector<T>::CUvector(const CUvector<T> &rhs)
{
	error("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline CUvector<T> & CUvector<T>::operator=(const CUvector &rhs)
{
	if (this == &rhs) error("self assignment is forbidden!");
	if (rhs.size() != N) error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline CUvector<T> & CUvector<T>::operator=(const NRvector<T> &v)
{
	if (v.size() != N)
		error("size mismatch!");
	cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline CUref<T> CUvector<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUvector subscript out of bounds");
#endif
	return CUref<T>(p+i);
}

template <typename T>
inline const CUref<T> CUvector<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=N)
		error("CUvector subscript out of bounds");
#endif
	return CUref<T>(p+i);
}

template <typename T> template <typename T1>
inline void CUvector<T>::get(NRvector<T1> &v) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		error("wrong type size!");
#endif
	v.resize(N);
	cudaMemcpy(v.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline void CUvector<T>::resize(Long_I n)
{ Base::resize(n); }

template<typename T> template<typename T1>
inline void CUvector<T>::resize(const CUvector<T1>& v)
{ resize(v.size()); }

template<typename T> template<typename T1>
inline void CUvector<T>::resize(const NRvector<T1>& v)
{ resize(v.size()); }

// Matrix Class

template <typename T>
class CUmatrix : public CUbase<T>
{
private:
	Long nn, mm;
	CUptr<T> *v;
	inline CUptr<T>* v_alloc();
public:
	typedef CUbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	CUmatrix();
	CUmatrix(Long_I n, Long_I m);
	CUmatrix(Long_I n, Long_I m, const T &a); //Initialize to constant
	CUmatrix(NRmatrix<T> &v); // initialize from cpu matrix
	CUmatrix(const CUmatrix &rhs);		// Copy constructor forbidden
	inline Long nrows() const;
	inline Long ncols() const;
	template <typename T1>
	inline void get(NRmatrix<T1> &v) const; // copy to cpu vector
	inline CUmatrix & operator=(const CUmatrix &rhs); //copy assignment
	inline CUmatrix & operator=(const NRmatrix<T> &rhs); //NR assignment
	inline CUptr<T> operator[](Long_I i);  //subscripting: pointer to row i
	// TODO: should return low level const
	inline CUptr<T> operator[](Long_I i) const;
	inline void resize(Long_I n, Long_I m); // resize (contents not preserved)
	template <typename T1>
	inline void resize(const CUmatrix<T1> &v);
	template <typename T1>
	inline void resize(const NRmatrix<T1> &v);
	~CUmatrix();
};

template <typename T>
inline CUptr<T>* CUmatrix<T>::v_alloc()
{
	if (N == 0) return nullptr;
	CUptr<T> *v = new CUptr<T>[nn];
	v[0] = p;
	for (Long i = 1; i<nn; i++)
		v[i] = v[i-1] + mm;
	return v;
}

template <typename T>
CUmatrix<T>::CUmatrix() : nn(0), mm(0), v(nullptr) {}

template <typename T>
CUmatrix<T>::CUmatrix(Long_I n, Long_I m) : Base(n*m), nn(n), mm(m), v(v_alloc()) {}

template <typename T>
CUmatrix<T>::CUmatrix(Long_I n, Long_I m, const T &s) : CUmatrix(n, m)
{ *this = s; }

template <typename T>
CUmatrix<T>::CUmatrix(NRmatrix<T> &v) : CUmatrix(v.nrows(), v.ncols())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
CUmatrix<T>::CUmatrix(const CUmatrix<T> &rhs)
{
	error("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!")
}

template <typename T>
inline Long CUmatrix<T>::nrows() const
{ return nn; }

template <typename T>
inline Long CUmatrix<T>::ncols() const
{ return mm; }

template <typename T> template <typename T1>
inline void CUmatrix<T>::get(NRmatrix<T1> &a) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		error("wrong type size!");
#endif
	a.resize(nn, mm);
	cudaMemcpy(a.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline CUmatrix<T> & CUmatrix<T>::operator=(const CUmatrix &rhs)
{
	if (this == &rhs) error("self assignment is forbidden!");
	if (rhs.nrows() != nn || rhs.ncols() != mm) error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline CUmatrix<T> & CUmatrix<T>::operator=(const NRmatrix<T> &rhs)
{
	if (rhs.nrows() != nn || rhs.ncols() != mm) error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline CUptr<T> CUmatrix<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn)
		error("CUmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline CUptr<T> CUmatrix<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn)
		error("CUmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline void CUmatrix<T>::resize(Long_I n, Long_I m)
{
	if (n != nn || m != mm) {
		Base::resize(n*m);
		nn = n; mm = m;
		if (v) delete v;
		v = v_alloc();
	}
}

template<typename T> template<typename T1>
inline void CUmatrix<T>::resize(const CUmatrix<T1>& v)
{ resize(v.nrows(), v.ncols()); }

template<typename T> template<typename T1>
inline void CUmatrix<T>::resize(const NRmatrix<T1>& v)
{ resize(v.nrows(), v.ncols()); }

template <typename T>
CUmatrix<T>::~CUmatrix()
{ if(v) delete v; }


// 3D Matrix Class

template <typename T>
class CUmat3d : public CUbase<T>
{
private:
	Long nn;
	Long mm;
	Long kk;
	CUptr<T> **v;
	inline CUptr<T>** v_alloc();
	inline void v_free();
public:
	typedef CUbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	CUmat3d();
	CUmat3d(Long_I n, Long_I m, Long_I k);
	CUmat3d(Long_I n, Long_I m, Long_I k, const T &a); //Initialize to constant
	CUmat3d(NRmat3d<T> &v); // initialize from cpu matrix
	CUmat3d(const CUmat3d &rhs);   // Copy constructor forbidden
	inline Long dim1() const;
	inline Long dim2() const;
	inline Long dim3() const;
	template <typename T1>
	inline void get(NRmat3d<T1> &v) const; // copy to cpu matrix
	inline CUmat3d & operator=(const CUmat3d &rhs);	//copy assignment
	inline CUmat3d & operator=(const NRmat3d<T> &rhs); //NR assignment
	inline CUptr<T>* operator[](Long_I i);	//subscripting: pointer to row i
	// TODO: should return pointer to low level const
	inline CUptr<T>* operator[](Long_I i) const;
	inline void resize(Long_I n, Long_I m, Long_I k);
	template <typename T1>
	inline void resize(const CUmat3d<T1> &v);
	template <typename T1>
	inline void resize(const NRmat3d<T1> &v);
	~CUmat3d();
};

template <typename T>
inline CUptr<T>** CUmat3d<T>::v_alloc()
{
	if (N == 0) return nullptr;
	Long i;
	Long nnmm = nn*mm;
	CUptr<T> *v0 = new CUptr<T>[nnmm]; v0[0] = p;
	for (i = 1; i < nnmm; ++i)
		v0[i] = v0[i - 1] + kk;
	CUptr<T> **v = new CUptr<T>*[nn]; v[0] = v0;
	for(i = 1; i < nn; ++i)
		v[i] = v[i-1] + mm;
	return v;
}

template <typename T>
inline void CUmat3d<T>::v_free()
{
	if (v != nullptr) {
		delete v[0]; delete v;
	}
}

template <typename T>
CUmat3d<T>::CUmat3d() : nn(0), mm(0), kk(0), v(nullptr) {}

template <typename T>
CUmat3d<T>::CUmat3d(Long_I n, Long_I m, Long_I k) :
Base(n*m*k), nn(n), mm(m), kk(k), v(v_alloc()) {}

template <typename T>
CUmat3d<T>::CUmat3d(Long_I n, Long_I m, Long_I k, const T &s) : CUmat3d(n, m, k)
{ *this = s; }

template <typename T>
CUmat3d<T>::CUmat3d(NRmat3d<T> &v) : CUmat3d(v.dim1(), v.dim2(), v.dim3())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
CUmat3d<T>::CUmat3d(const CUmat3d<T> &rhs)
{
	error("Copy constructor or move constructor is forbidden, "
	"use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline Long CUmat3d<T>::dim1() const
{ return nn; }

template <typename T>
inline Long CUmat3d<T>::dim2() const
{ return mm; }

template <typename T>
inline Long CUmat3d<T>::dim3() const
{ return kk; }

template <typename T> template <typename T1>
inline void CUmat3d<T>::get(NRmat3d<T1> &a) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		error("wrong type size");
#endif
	a.resize(nn, mm, kk);
	cudaMemcpy(a.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline CUmat3d<T> & CUmat3d<T>::operator=(const CUmat3d &rhs)
{
	if (this == &rhs) error("self assignment is forbidden!");
	if (rhs.dim1() != nn || rhs.dim2() != mm || rhs.dim3() != kk)
		error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline CUmat3d<T> & CUmat3d<T>::operator=(const NRmat3d<T> &rhs)
{
	if (rhs.dim1() != nn || rhs.dim2() != mm || rhs.dim3() != kk)
		error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline CUptr<T>* CUmat3d<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn)
		error("CUmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline CUptr<T>* CUmat3d<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn)
		error("CUmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline void CUmat3d<T>::resize(Long_I n, Long_I m, Long_I k)
{
	if (n != nn || m != mm || k != kk) {
		Base::resize(n*m*k);
		nn = n; mm = m; kk = k;
		if (v) delete v;
		v = v_alloc();
	}
}

template<typename T> template<typename T1>
inline void CUmat3d<T>::resize(const CUmat3d<T1>& v)
{ resize(v.dim1(), v.dim2(), v.dim3()); }

template<typename T> template<typename T1>
inline void CUmat3d<T>::resize(const NRmat3d<T1>& v)
{ resize(v.dim1(), v.dim2(), v.dim3()); }

template <typename T>
CUmat3d<T>::~CUmat3d()
{ v_free(); }

// Scalar, vector and matrix types

typedef const CUscalar<Int> Gint_I;
typedef CUscalar<Int> Gint, Gint_O, Gint_IO;
typedef const CUscalar<Uint> Guint_I;
typedef CUscalar<Uint> Guint, Guint_O, Guint_IO;

typedef const CUscalar<Long> Glong_I;
typedef CUscalar<Long> Glong, Glong_O, Glong_IO;

typedef const CUscalar<Llong> Gllong_I;
typedef CUscalar<Llong> Gllong, Gllong_O, Gllong_IO;
typedef const CUscalar<Ullong> Gullong_I;
typedef CUscalar<Ullong> Gullong, Gullong_O, Gullong_IO;

typedef const CUscalar<Char> Gchar_I;
typedef CUscalar<Char> Gchar, Gchar_O, Gchar_IO;
typedef const CUscalar<Uchar> Guchar_I;
typedef CUscalar<Uchar> Guchar, Guchar_O, Guchar_IO;

typedef const CUscalar<Doub> Gdoub_I;
typedef CUscalar<Doub> Gdoub, Gdoub_O, Gdoub_IO;
typedef const CUscalar<Ldoub> Gldoub_I;
typedef CUscalar<Ldoub> Gldoub, Gldoub_O, Gldoub_IO;

typedef const CUscalar<Comp> Gcomp_I;
typedef CUscalar<Comp> Gcomp, Gcomp_O, Gcomp_IO;

typedef const CUscalar<Bool> Gbool_I;
typedef CUscalar<Bool> Gbool, Gbool_O, Gbool_IO;

typedef const CUvector<Int> GvecInt_I;
typedef CUvector<Int> GvecInt, GvecInt_O, GvecInt_IO;

typedef const CUvector<Uint> GvecUint_I;
typedef CUvector<Uint> GvecUint, GvecUint_O, GvecUint_IO;

typedef const CUvector<Long> GvecLong_I;
typedef CUvector<Long> GvecLong, GvecLong_O, GvecLong_IO;

typedef const CUvector<Llong> GvecLlong_I;
typedef CUvector<Llong> GvecLlong, GvecLlong_O, GvecLlong_IO;

typedef const CUvector<Ullong> GvecUllong_I;
typedef CUvector<Ullong> GvecUllong, GvecUllong_O, GvecUllong_IO;

typedef const CUvector<Char> GvecChar_I;
typedef CUvector<Char> GvecChar, GvecChar_O, GvecChar_IO;

typedef const CUvector<Char*> GvecCharp_I;
typedef CUvector<Char*> GvecCharp, GvecCharp_O, GvecCharp_IO;

typedef const CUvector<Uchar> GvecUchar_I;
typedef CUvector<Uchar> GvecUchar, GvecUchar_O, GvecUchar_IO;

typedef const CUvector<Doub> GvecDoub_I;
typedef CUvector<Doub> GvecDoub, GvecDoub_O, GvecDoub_IO;

typedef const CUvector<Doub*> GvecDoubp_I;
typedef CUvector<Doub*> GvecDoubp, GvecDoubp_O, GvecDoubp_IO;

typedef const CUvector<Comp> GvecComp_I;
typedef CUvector<Comp> GvecComp, GvecComp_O, GvecComp_IO;

typedef const CUvector<Bool> GvecBool_I;
typedef CUvector<Bool> GvecBool, GvecBool_O, GvecBool_IO;

typedef const CUmatrix<Int> GmatInt_I;
typedef CUmatrix<Int> GmatInt, GmatInt_O, GmatInt_IO;

typedef const CUmatrix<Uint> GmatUint_I;
typedef CUmatrix<Uint> GmatUint, GmatUint_O, GmatUint_IO;

typedef const CUmatrix<Llong> GmatLlong_I;
typedef CUmatrix<Llong> GmatLlong, GmatLlong_O, GmatLlong_IO;

typedef const CUmatrix<Ullong> GmatUllong_I;
typedef CUmatrix<Ullong> GmatUllong, GmatUllong_O, GmatUllong_IO;

typedef const CUmatrix<Char> GmatChar_I;
typedef CUmatrix<Char> GmatChar, GmatChar_O, GmatChar_IO;

typedef const CUmatrix<Uchar> GmatUchar_I;
typedef CUmatrix<Uchar> GmatUchar, GmatUchar_O, GmatUchar_IO;

typedef const CUmatrix<Doub> GmatDoub_I;
typedef CUmatrix<Doub> GmatDoub, GmatDoub_O, GmatDoub_IO;

typedef const CUmatrix<Comp> GmatComp_I;
typedef CUmatrix<Comp> GmatComp, GmatComp_O, GmatComp_IO;

typedef const CUmatrix<Bool> GmatBool_I;
typedef CUmatrix<Bool> GmatBool, GmatBool_O, GmatBool_IO;

typedef const CUmat3d<Doub> Gmat3Doub_I;
typedef CUmat3d<Doub> Gmat3Doub, Gmat3Doub_O, Gmat3Doub_IO;

typedef const CUmat3d<Comp> Gmat3Comp_I;
typedef CUmat3d<Comp> Gmat3Comp, Gmat3Comp_O, Gmat3Comp_IO;

} // namespace slisc
