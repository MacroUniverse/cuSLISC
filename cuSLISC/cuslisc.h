// classes for cuda matrix
#pragma once
#include "../SLISC/time.h"
#include "../SLISC/matrix.h"
#include "blocks_threads.h"
#include "complex.h"
//#include "complex/complex.h"

namespace slisc {

// complex type for cuda kernel
// should not be used in host code
typedef cuslisc::complex<Doub> Cump;
typedef const Cump Cump_I;
typedef Cump &Cump_O, &Cump_IO;

inline Cump& toCump(Comp &s) { return reinterpret_cast<Cump&>(s); }
inline Cump_I & toCump(Comp_I &s) { return reinterpret_cast<Cump_I&>(s); }

// specialized overloading of operator== for Comp
// so that "Gref<Comp>" can be implicitly converted to "Comp" when used as argument
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
// 	if (getsym(sym) != val1)
//		SLS_ERR("failed!");
// #endif
// }

template <typename T>
inline void setsym(T &sym, const T &val)
{
	cudaMemcpyToSymbol(sym, &val, sizeof(T));
#ifdef _CHECKSETSYS_
	if (getsym(sym) != val)
		SLS_ERR("failed!");
#endif
}

inline void setsym(Cump &sym, Comp_I &val)
{
	cudaMemcpyToSymbol(sym, &val, sizeof(Comp));
#ifdef _CHECKSETSYS_
	if ( getsym(sym) != val )
		SLS_ERR("failed!");
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

// reference type to Gbase element
template <typename T>
class Gref
{
protected:
	typedef typename KerT<T>::type T1;
	T1 * p;
public:
	Gref() {};
	explicit Gref(T1* ptr) : p(ptr) {}
	void bind(T1* ptr) { p = ptr; };
	T1* ptr() { return p; }
	const T1* ptr() const { return p; }
	inline operator T() const;
	inline Gref& operator=(const T& rhs);
	template <typename T1>
	inline Gref& operator+=(const T1& rhs) { *this = T(*this) + rhs; return *this; }
	template <typename T1>
	inline Gref& operator-=(const T1& rhs) { *this = T(*this) - rhs; return *this; }
	template <typename T1>
	inline Gref& operator*=(const T1& rhs) { *this = T(*this) * rhs; return *this; }
	template <typename T1>
	inline Gref& operator/=(const T1& rhs) { *this = T(*this) / rhs; return *this; }
};

template <typename T>
inline Gref<T>::operator T() const
{
	T val;
	cudaMemcpy(&val, p, sizeof(T), cudaMemcpyDeviceToHost);
	return val;
}

template <typename T>
inline Gref<T>& Gref<T>::operator=(const T& rhs)
{
	cudaMemcpy(p, &rhs, sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

// pointer type to Gbase element
// note : const Gptr is top level const
// TODO : create a class for low level const, or is Gptr<const T> a low level const??
template <typename T>
class Gptr
{
protected:
	typedef typename KerT<T>::type T1;
	T1* p;
public:
	Gptr() : p(nullptr) {}
	Gptr(T1 *ptr) : p(ptr) {}
	T1* ptr() const { return p; }
	inline Gref<T> & operator*() const; // dereference
	Gref<T> operator[](Long_I i) const { return Gref<T>(p+i); }
	Gptr & operator=(const Gptr &rhs) { p = rhs.ptr(); return *this; } // copy assignment
	Gptr & operator=(T1* ptr) { p = ptr; return *this; } // T1* assignment
	void operator+=(Long_I i) { p += i; }
	void operator-=(Long_I i) { p -= i; }
};

template <typename T>
inline Gref<T> & Gptr<T>::operator*() const
{ return reinterpret_cast<Gref<T>&>(*const_cast<Gptr<T>*>(this)); }

template <typename T>
inline Gptr<T> operator+(const Gptr<T> &pcu, Long_I i) { return Gptr<T>(pcu.ptr()+i); }

template <typename T>
inline Gptr<T> operator-(const Gptr<T> &pcu, Long_I i) { return Gptr<T>(pcu.ptr()-i); }

// scalar class
template <typename T>
class Gscalar : public Gref<T>
{
private:
	typedef typename KerT<T>::type T1;
public:
	typedef Gref<T> Base;
	using Base::p;
	using Base::operator=;
	Gscalar() { cudaMalloc(&p, sizeof(T)); }
	explicit Gscalar(const T &s) : Gscalar() { *this = s; }
};

// base class for Gvector, Gmatrix, Gmat3d
template <typename T>
class Gbase
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
	Gbase() : N(0), p(nullptr) {}
	explicit Gbase(Long_I n) : N(n) { cudaMalloc(&p, N*sizeof(T)); }
	p_kernel_type ptr() { return p; } // get pointer
	p_const_kernel_type ptr() const { return p; }
	Long_I size() const { return N; }
	inline void resize(Long_I n);
	inline Gref<T> operator()(Long_I i);
	inline const Gref<T> operator()(Long_I i) const;
	inline Gref<T> end(); // last element
	inline const Gref<T> end() const;
	inline Gbase & operator=(const T &rhs); // set scalar
	~Gbase() { if (p) cudaFree(p); }
};

template <typename T>
inline void Gbase<T>::resize(Long_I n)
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
inline Gref<T> Gbase<T>::operator()(Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	SLS_ERR("Gbase subscript out of bounds!");
#endif
	return Gref<T>(p+i);
}

template <typename T>
inline const Gref<T> Gbase<T>::operator()(Long_I i) const
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	SLS_ERR("Gbase subscript out of bounds!");
#endif
	return Gref<T>(p+i);
}

template <typename T>
inline Gref<T> Gbase<T>::end()
{
#ifdef _CHECKBOUNDS_
	if (N < 1)
		SLS_ERR("Using end() for empty object!");
#endif
	return Gref<T>(p+N-1);
}

template <typename T>
inline const Gref<T> Gbase<T>::end() const
{
#ifdef _CHECKBOUNDS_
	if (N < 1)
		SLS_ERR("Using end() for empty object!");
#endif
	return Gref<T>(p+N-1);
}

template <typename T>
inline Gbase<T> & Gbase<T>::operator=(const T &rhs)
{
	if (N) cumemset<<<nbl(Nbl_cumemset,Nth_cumemset,N), Nth_cumemset>>>(p, (kernel_type&)rhs, N);
	return *this;
}

// Vector Class

template <typename T>
class Gvector : public Gbase<T>
{
public:
	typedef Gbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	Gvector() {};
	explicit Gvector(Long_I n) : Base(n) {}
	Gvector(Long_I n, const T &a);	//initialize to constant value
	Gvector(Vector<T> &v); // initialize from cpu vector
	Gvector(const Gvector &rhs);	// Copy constructor forbidden
	template <typename T1>
	inline void get(Vector<T1> &v) const; // copy to cpu vector
	inline Gvector & operator=(const Gvector &rhs);	// copy assignment
	inline Gvector & operator=(const Vector<T> &v); // NR assignment
	inline Gref<T> operator[](Long_I i); //i'th element
	inline const Gref<T> operator[](Long_I i) const;
	inline void resize(Long_I n); // resize (contents not preserved)
	template <typename T1>
	inline void resize(const Gvector<T1> &v);
	template <typename T1>
	inline void resize(const Vector<T1> &v);
};

template <typename T>
Gvector<T>::Gvector(Long_I n, const T &a) : Gvector(n)
{ *this = a; }

template <typename T>
Gvector<T>::Gvector(Vector<T> &v) : Gvector(v.size())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
Gvector<T>::Gvector(const Gvector<T> &rhs)
{
	SLS_ERR("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline Gvector<T> & Gvector<T>::operator=(const Gvector &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.size() != N)
		SLS_ERR("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gvector<T> & Gvector<T>::operator=(const Vector<T> &v)
{
	if (v.size() != N)
		SLS_ERR("size mismatch!");
	cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline Gref<T> Gvector<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	SLS_ERR("Gvector subscript out of bounds");
#endif
	return Gref<T>(p+i);
}

template <typename T>
inline const Gref<T> Gvector<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=N)
		SLS_ERR("Gvector subscript out of bounds");
#endif
	return Gref<T>(p+i);
}

template <typename T> template <typename T1>
inline void Gvector<T>::get(Vector<T1> &v) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size!");
#endif
	v.resize(N);
	cudaMemcpy(v.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline void Gvector<T>::resize(Long_I n)
{ Base::resize(n); }

template<typename T> template<typename T1>
inline void Gvector<T>::resize(const Gvector<T1>& v)
{ resize(v.size()); }

template<typename T> template<typename T1>
inline void Gvector<T>::resize(const Vector<T1>& v)
{ resize(v.size()); }

// Matrix Class

template <typename T>
class Gmatrix : public Gbase<T>
{
private:
	Long m_N1, m_N2;
	Gptr<T> *v;
	inline Gptr<T>* v_alloc();
public:
	typedef Gbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	Gmatrix();
	Gmatrix(Long_I n, Long_I m);
	Gmatrix(Long_I n, Long_I m, const T &a); //Initialize to constant
	Gmatrix(Matrix<T> &v); // initialize from cpu matrix
	Gmatrix(const Gmatrix &rhs);		// Copy constructor forbidden
	inline Long n1() const;
	inline Long n2() const;
	template <typename T1>
	inline void get(Matrix<T1> &v) const; // copy to cpu vector
	inline Gmatrix & operator=(const Gmatrix &rhs); //copy assignment
	inline Gmatrix & operator=(const Matrix<T> &rhs); //NR assignment
	inline Gptr<T> operator[](Long_I i);  //subscripting: pointer to row i
	// TODO: should return low level const
	inline Gptr<T> operator[](Long_I i) const;
	inline void resize(Long_I n, Long_I m); // resize (contents not preserved)
	template <typename T1>
	inline void resize(const Gmatrix<T1> &v);
	template <typename T1>
	inline void resize(const Matrix<T1> &v);
	~Gmatrix();
};

template <typename T>
inline Gptr<T>* Gmatrix<T>::v_alloc()
{
	if (N == 0) return nullptr;
	Gptr<T> *v = new Gptr<T>[m_N1];
	v[0] = p;
	for (Long i = 1; i<m_N1; i++)
		v[i] = v[i-1] + m_N2;
	return v;
}

template <typename T>
Gmatrix<T>::Gmatrix() : m_N1(0), m_N2(0), v(nullptr) {}

template <typename T>
Gmatrix<T>::Gmatrix(Long_I n, Long_I m) : Base(n*m), m_N1(n), m_N2(m), v(v_alloc()) {}

template <typename T>
Gmatrix<T>::Gmatrix(Long_I n, Long_I m, const T &s) : Gmatrix(n, m)
{ *this = s; }

template <typename T>
Gmatrix<T>::Gmatrix(Matrix<T> &v) : Gmatrix(v.n1(), v.n2())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
Gmatrix<T>::Gmatrix(const Gmatrix<T> &rhs)
{
	SLS_ERR("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline Long Gmatrix<T>::n1() const
{ return m_N1; }

template <typename T>
inline Long Gmatrix<T>::n2() const
{ return m_N2; }

template <typename T> template <typename T1>
inline void Gmatrix<T>::get(Matrix<T1> &a) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size!");
#endif
	a.resize(m_N1, m_N2);
	cudaMemcpy(a.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline Gmatrix<T> & Gmatrix<T>::operator=(const Gmatrix &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2)
		SLS_ERR("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gmatrix<T> & Gmatrix<T>::operator=(const Matrix<T> &rhs)
{
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2)
		SLS_ERR("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline Gptr<T> Gmatrix<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=m_N1)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline Gptr<T> Gmatrix<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=m_N1)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline void Gmatrix<T>::resize(Long_I n, Long_I m)
{
	if (n != m_N1 || m != m_N2) {
		Base::resize(n*m);
		m_N1 = n; m_N2 = m;
		if (v) delete v;
		v = v_alloc();
	}
}

template<typename T> template<typename T1>
inline void Gmatrix<T>::resize(const Gmatrix<T1>& v)
{ resize(v.n1(), v.n2()); }

template<typename T> template<typename T1>
inline void Gmatrix<T>::resize(const Matrix<T1>& v)
{ resize(v.n1(), v.n2()); }

template <typename T>
Gmatrix<T>::~Gmatrix()
{ if(v) delete v; }


// 3D Matrix Class

template <typename T>
class Gmat3d : public Gbase<T>
{
private:
	Long m_N1;
	Long m_N2;
	Long m_N3;
	Gptr<T> **v;
	inline Gptr<T>** v_alloc();
	inline void v_free();
public:
	typedef Gbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	Gmat3d();
	Gmat3d(Long_I n, Long_I m, Long_I k);
	Gmat3d(Long_I n, Long_I m, Long_I k, const T &a); //Initialize to constant
	Gmat3d(Mat3d<T> &v); // initialize from cpu matrix
	Gmat3d(const Gmat3d &rhs);   // Copy constructor forbidden
	inline Long n1() const;
	inline Long n2() const;
	inline Long n3() const;
	template <typename T1>
	inline void get(Mat3d<T1> &v) const; // copy to cpu matrix
	inline Gmat3d & operator=(const Gmat3d &rhs);	//copy assignment
	inline Gmat3d & operator=(const Mat3d<T> &rhs); //NR assignment
	inline Gptr<T>* operator[](Long_I i);	//subscripting: pointer to row i
	// TODO: should return pointer to low level const
	inline Gptr<T>* operator[](Long_I i) const;
	inline void resize(Long_I n, Long_I m, Long_I k);
	template <typename T1>
	inline void resize(const Gmat3d<T1> &v);
	template <typename T1>
	inline void resize(const Mat3d<T1> &v);
	~Gmat3d();
};

template <typename T>
inline Gptr<T>** Gmat3d<T>::v_alloc()
{
	if (N == 0) return nullptr;
	Long i;
	Long nnmm = m_N1*m_N2;
	Gptr<T> *v0 = new Gptr<T>[nnmm]; v0[0] = p;
	for (i = 1; i < nnmm; ++i)
		v0[i] = v0[i - 1] + m_N3;
	Gptr<T> **v = new Gptr<T>*[m_N1]; v[0] = v0;
	for(i = 1; i < m_N1; ++i)
		v[i] = v[i-1] + m_N2;
	return v;
}

template <typename T>
inline void Gmat3d<T>::v_free()
{
	if (v != nullptr) {
		delete v[0]; delete v;
	}
}

template <typename T>
Gmat3d<T>::Gmat3d() : m_N1(0), m_N2(0), m_N3(0), v(nullptr) {}

template <typename T>
Gmat3d<T>::Gmat3d(Long_I n, Long_I m, Long_I k) :
Base(n*m*k), m_N1(n), m_N2(m), m_N3(k), v(v_alloc()) {}

template <typename T>
Gmat3d<T>::Gmat3d(Long_I n, Long_I m, Long_I k, const T &s) : Gmat3d(n, m, k)
{ *this = s; }

template <typename T>
Gmat3d<T>::Gmat3d(Mat3d<T> &v) : Gmat3d(v.n1(), v.n2(), v.n3())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
Gmat3d<T>::Gmat3d(const Gmat3d<T> &rhs)
{
	SLS_ERR("Copy constructor or move constructor is forbidden, "
	"use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline Long Gmat3d<T>::n1() const
{ return m_N1; }

template <typename T>
inline Long Gmat3d<T>::n2() const
{ return m_N2; }

template <typename T>
inline Long Gmat3d<T>::n3() const
{ return m_N3; }

template <typename T> template <typename T1>
inline void Gmat3d<T>::get(Mat3d<T1> &a) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size");
#endif
	a.resize(m_N1, m_N2, m_N3);
	cudaMemcpy(a.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline Gmat3d<T> & Gmat3d<T>::operator=(const Gmat3d &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2 || rhs.n3() != m_N3)
		SLS_ERR("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gmat3d<T> & Gmat3d<T>::operator=(const Mat3d<T> &rhs)
{
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2 || rhs.n3() != m_N3)
		SLS_ERR("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline Gptr<T>* Gmat3d<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=m_N1)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline Gptr<T>* Gmat3d<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=m_N1)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <typename T>
inline void Gmat3d<T>::resize(Long_I n, Long_I m, Long_I k)
{
	if (n != m_N1 || m != m_N2 || k != m_N3) {
		Base::resize(n*m*k);
		m_N1 = n; m_N2 = m; m_N3 = k;
		if (v) delete v;
		v = v_alloc();
	}
}

template<typename T> template<typename T1>
inline void Gmat3d<T>::resize(const Gmat3d<T1>& v)
{ resize(v.n1(), v.n2(), v.n3()); }

template<typename T> template<typename T1>
inline void Gmat3d<T>::resize(const Mat3d<T1>& v)
{ resize(v.n1(), v.n2(), v.n3()); }

template <typename T>
Gmat3d<T>::~Gmat3d()
{ v_free(); }

// Scalar, vector and matrix types

typedef const Gscalar<Int> Gint_I;
typedef Gscalar<Int> Gint, Gint_O, Gint_IO;
typedef const Gscalar<Uint> Guint_I;
typedef Gscalar<Uint> Guint, Guint_O, Guint_IO;

typedef const Gscalar<Long> Glong_I;
typedef Gscalar<Long> Glong, Glong_O, Glong_IO;

typedef const Gscalar<Llong> Gllong_I;
typedef Gscalar<Llong> Gllong, Gllong_O, Gllong_IO;
typedef const Gscalar<Ullong> Gullong_I;
typedef Gscalar<Ullong> Gullong, Gullong_O, Gullong_IO;

typedef const Gscalar<Char> Gchar_I;
typedef Gscalar<Char> Gchar, Gchar_O, Gchar_IO;
typedef const Gscalar<Uchar> Guchar_I;
typedef Gscalar<Uchar> Guchar, Guchar_O, Guchar_IO;

typedef const Gscalar<Doub> Gdoub_I;
typedef Gscalar<Doub> Gdoub, Gdoub_O, Gdoub_IO;
typedef const Gscalar<Ldoub> Gldoub_I;
typedef Gscalar<Ldoub> Gldoub, Gldoub_O, Gldoub_IO;

typedef const Gscalar<Comp> Gcomp_I;
typedef Gscalar<Comp> Gcomp, Gcomp_O, Gcomp_IO;

typedef const Gscalar<Bool> Gbool_I;
typedef Gscalar<Bool> Gbool, Gbool_O, Gbool_IO;

typedef const Gvector<Int> GvecInt_I;
typedef Gvector<Int> GvecInt, GvecInt_O, GvecInt_IO;

typedef const Gvector<Uint> GvecUint_I;
typedef Gvector<Uint> GvecUint, GvecUint_O, GvecUint_IO;

typedef const Gvector<Long> GvecLong_I;
typedef Gvector<Long> GvecLong, GvecLong_O, GvecLong_IO;

typedef const Gvector<Llong> GvecLlong_I;
typedef Gvector<Llong> GvecLlong, GvecLlong_O, GvecLlong_IO;

typedef const Gvector<Ullong> GvecUllong_I;
typedef Gvector<Ullong> GvecUllong, GvecUllong_O, GvecUllong_IO;

typedef const Gvector<Char> GvecChar_I;
typedef Gvector<Char> GvecChar, GvecChar_O, GvecChar_IO;

typedef const Gvector<Char*> GvecCharp_I;
typedef Gvector<Char*> GvecCharp, GvecCharp_O, GvecCharp_IO;

typedef const Gvector<Uchar> GvecUchar_I;
typedef Gvector<Uchar> GvecUchar, GvecUchar_O, GvecUchar_IO;

typedef const Gvector<Doub> GvecDoub_I;
typedef Gvector<Doub> GvecDoub, GvecDoub_O, GvecDoub_IO;

typedef const Gvector<Doub*> GvecDoubp_I;
typedef Gvector<Doub*> GvecDoubp, GvecDoubp_O, GvecDoubp_IO;

typedef const Gvector<Comp> GvecComp_I;
typedef Gvector<Comp> GvecComp, GvecComp_O, GvecComp_IO;

typedef const Gvector<Bool> GvecBool_I;
typedef Gvector<Bool> GvecBool, GvecBool_O, GvecBool_IO;

typedef const Gmatrix<Int> GmatInt_I;
typedef Gmatrix<Int> GmatInt, GmatInt_O, GmatInt_IO;

typedef const Gmatrix<Uint> GmatUint_I;
typedef Gmatrix<Uint> GmatUint, GmatUint_O, GmatUint_IO;

typedef const Gmatrix<Llong> GmatLlong_I;
typedef Gmatrix<Llong> GmatLlong, GmatLlong_O, GmatLlong_IO;

typedef const Gmatrix<Ullong> GmatUllong_I;
typedef Gmatrix<Ullong> GmatUllong, GmatUllong_O, GmatUllong_IO;

typedef const Gmatrix<Char> GmatChar_I;
typedef Gmatrix<Char> GmatChar, GmatChar_O, GmatChar_IO;

typedef const Gmatrix<Uchar> GmatUchar_I;
typedef Gmatrix<Uchar> GmatUchar, GmatUchar_O, GmatUchar_IO;

typedef const Gmatrix<Doub> GmatDoub_I;
typedef Gmatrix<Doub> GmatDoub, GmatDoub_O, GmatDoub_IO;

typedef const Gmatrix<Comp> GmatComp_I;
typedef Gmatrix<Comp> GmatComp, GmatComp_O, GmatComp_IO;

typedef const Gmatrix<Bool> GmatBool_I;
typedef Gmatrix<Bool> GmatBool, GmatBool_O, GmatBool_IO;

typedef const Gmat3d<Doub> Gmat3Doub_I;
typedef Gmat3d<Doub> Gmat3Doub, Gmat3Doub_O, Gmat3Doub_IO;

typedef const Gmat3d<Comp> Gmat3Comp_I;
typedef Gmat3d<Comp> Gmat3Comp, Gmat3Comp_O, Gmat3Comp_IO;

} // namespace slisc
