// classes for cuda matrix
#include "nr3.h"

Int NblMax, Nth0; // <<<nbl0(N),Nth0>>> for element-wise kernels

// always call this function first
// TODO: get GPU properties
void cuInit()
{
	NblMax = 320; Nth0 = 32;
}

// calculate number of CUDA blocks needed
inline Int nbl0(Int N)
{ return min(NblMax, (N + Nth0 - 1)/Nth0); }

// set elements to same value
template <class T>
__global__
void cumemset(T *dest, const T val, Long_I n)
{
	Int i, ind, stride;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i = ind; i < n; i += stride)
		dest[i] = val;
}

// copy elements using CUDA instead of cudaMemcpyDeviceToDevice
// advantage unknown
template <class T>
__global__
void cumemcpy(T *dest, const T *src, Long_I n)
{
	Int i, ind, stride;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i = ind; i < n; i += stride)
		dest[i] = src[i];
}

// reference type to CUbase element
template <class T>
class CUref
{
protected:
	T* p;
public:
	CUref(T* ptr);
	inline T* ptr() {return p;}
	inline const T* ptr() const {return p;}
	inline operator T() const;
	inline CUref& operator=(const T& rhs);
	inline CUref& operator+=(const T& rhs);
	inline CUref& operator-=(const T& rhs);
	inline CUref& operator*=(const T& rhs);
	inline CUref& operator/=(const T& rhs);
};

template <class T>
inline CUref<T>::CUref(T* ptr): p(ptr) {}

template <class T>
inline CUref<T>::operator T() const
{
	T val;
	cudaMemcpy(&val, p, sizeof(T), cudaMemcpyDeviceToHost);
	return val;
}

template <class T>
inline CUref<T>& CUref<T>::operator=(const T& rhs)
{
	cudaMemcpy(p, &rhs, sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <class T>
inline CUref<T>& CUref<T>::operator+=(const T& rhs)
{
	*this = *this + rhs;
	return *this;
}

template <class T>
inline CUref<T>& CUref<T>::operator-=(const T& rhs)
{
	*this = *this - rhs;
	return *this;
}

template <class T>
inline CUref<T>& CUref<T>::operator*=(const T& rhs)
{
	*this = *this * rhs;
	return *this;
}

template <class T>
inline CUref<T>& CUref<T>::operator/=(const T& rhs)
{
	*this = *this / rhs;
	return *this;
}

// pointer type to CUbase element
// note : const CUptr is top level const
// TODO : implement ConstCUptr
template <class T>
class CUptr
{
protected:
	T* p;
public:
	CUptr() : p(nullptr) {}
	CUptr(T *ptr) : p(ptr) {}
	T* ptr() const {return p;}
	CUref<T> operator*() const {return CUref<T>(p);} // dereference
	CUref<T> operator[](Long_I i) const {return CUref<T>(p+i);}
	CUptr & operator=(const CUptr &rhs) {p = rhs.ptr(); return *this;} // copy assignment
	CUptr & operator=(T* ptr) {p = ptr; return *this;} // T* assignment
	void operator+=(Long_I i) { p += i; }
	void operator-=(Long_I i) { p -= i; }
};

template <class T>
CUptr<T> operator+(const CUptr<T> &pcu, Long_I i) { return CUptr<T>(pcu.ptr()+i); }
template <class T>
CUptr<T> operator-(const CUptr<T> &pcu, Long_I i) { return CUptr<T>(pcu.ptr()-i); }

// device memory
template <class T>
class CUbase
{
protected:
	Long N;// number of elements
	T* p; // pointer to the first element
public:
	CUbase();
	CUbase(Long_I N);
	inline T* ptr();// get pointer
	inline const T* ptr() const;
	inline Long_I size() const;
	inline void resize(Long_I n);
	inline CUref<T> operator()(Long_I i);
	inline const CUref<T> operator()(Long_I i) const;
	inline CUref<T> end(); // last element
	inline const CUref<T> end() const;
	inline CUbase & operator=(const T &rhs); // set scalar
	~CUbase();
};

template <class T>
CUbase<T>::CUbase(): N(0), p(nullptr) {}

template <class T>
CUbase<T>::CUbase(Long_I n) : N(n)
{ cudaMalloc(&p, N*sizeof(T)); }

template <class T>
inline T* CUbase<T>::ptr()
{ return p; }

template <class T>
inline const T* CUbase<T>::ptr() const
{ return p; }

template <class T>
inline Long_I CUbase<T>::size() const
{ return N; }

template <class T>
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

template <class T>
inline CUref<T> CUbase<T>::operator()(Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUbase subscript out of bounds")
#endif
	return CUref<T>(p+i);
}

template <class T>
inline const CUref<T> CUbase<T>::operator()(Long_I i) const
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUbase subscript out of bounds");
#endif
	return CUref<T>(p+i);
}

template <class T>
inline CUref<T> CUbase<T>::end()
{
#ifdef _CHECKBOUNDS_
	if (N < 1)
		error("Using end() for empty object")
#endif
	return CUref<T>(p+N-1);
}

template <class T>
inline const CUref<T> CUbase<T>::end() const
{
#ifdef _CHECKBOUNDS_
	if (N < 1)
		error("Using end() for empty object")
#endif
	T val;
	cudaMemcpy(&val, p+N-1, sizeof(T), cudaMemcpyDeviceToHost);
	return CUref<T>(p+N-1);
}

template <class T>
inline CUbase<T> & CUbase<T>::operator=(const T &rhs)
{
	if (N) cumemset<<<nbl0(N), Nth0>>>(p, rhs, N);
	return *this;
}

template <class T>
CUbase<T>::~CUbase()
{ if (p) cudaFree(p); }


// Vector Class

template <class T>
class CUvector : public CUbase<T>
{
public:
	typedef CUbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
	CUvector();
	explicit CUvector(Long_I n);
	CUvector(Long_I n, const T &a);	//initialize to constant value
	CUvector(NRvector<T> &v); // initialize from cpu vector
	inline void get(NRvector<T> &v) const; // copy to cpu vector
	CUvector(const CUvector &rhs);	// Copy constructor forbidden
	inline CUvector & operator=(const CUvector &rhs);	// copy assignment
	inline CUvector & operator=(const NRvector<T> &v); // NR assignment
	inline CUref<T> operator[](Long_I i); //i'th element
	inline const CUref<T> operator[](Long_I i) const;
	inline void resize(Long_I newn); // resize (contents not preserved)
	template <class T1>
	inline void resize(const CUvector<T1> &v);
	template <class T1>
	inline void resize(const NRvector<T1> &v);
};

template <class T>
CUvector<T>::CUvector() {}

template <class T>
CUvector<T>::CUvector(Long_I n) : Base(n) {}

template <class T>
CUvector<T>::CUvector(Long_I n, const T &a) : CUvector(n)
{ cumemset<<<nbl0(N), Nth0>>>(p, a, N); }

template <class T>
CUvector<T>::CUvector(NRvector<T> &v) : CUvector(v.size())
{ cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice); }

template <class T>
CUvector<T>::CUvector(const CUvector<T> &rhs)
{
	error("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!")
}

template <class T>
inline CUvector<T> & CUvector<T>::operator=(const CUvector &rhs)
{
	if (this == &rhs) error("self assignment is forbidden!");
	if (rhs.size() != N) error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <class T>
inline CUvector<T> & CUvector<T>::operator=(const NRvector<T> &v)
{
	if (v.size() != N)
		error("size mismatch!");
	cudaMemcpy(p, v.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <class T>
inline CUref<T> CUvector<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUvector subscript out of bounds")
#endif
	return CUref<T>(p+i);
}

template <class T>
inline const CUref<T> CUvector<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=N)
	error("CUvector subscript out of bounds");
#endif
	return CUref<T>(p+i);
}

template <class T>
inline void CUvector<T>::get(NRvector<T> &v) const
{
	v.resize(N);
	cudaMemcpy(v.ptr(), p, N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <class T>
inline void CUvector<T>::resize(Long_I n)
{ Base::resize(n); }

template<class T> template<class T1>
inline void CUvector<T>::resize(const CUvector<T1>& v)
{ resize(v.size()); }

template<class T> template<class T1>
inline void CUvector<T>::resize(const NRvector<T1>& v)
{ resize(v.size()); }

// Matrix Class

template <class T>
class CUmatrix : public CUbase<T>
{
	typedef CUbase<T> Base;
	using Base::p;
	using Base::N;
	using Base::operator=;
private:
	Long nn, mm;
	CUptr<T> *v;
	inline CUptr<T>* v_alloc();
public:
	CUmatrix();
	CUmatrix(Long_I n, Long_I m);
	CUmatrix(Long_I n, Long_I m, const T &a);	//Initialize to constant
	CUmatrix(const CUmatrix &rhs);		// Copy constructor forbidden
	inline Long nrows() const;
	inline Long ncols() const;
	inline CUmatrix & operator=(const CUmatrix &rhs); //copy assignment
	inline CUmatrix & operator=(const NRmatrix<T> &rhs); //NR assignment
	inline CUmatrix & operator=(const T &rhs); // scalar assignment
	inline CUptr<T> operator[](Long_I i);	//subscripting: pointer to row i
	inline const T* operator[](Long_I i) const;
	inline void resize(Long_I newn, Long_I newm); // resize (contents not preserved)
	~CUmatrix();
};

template <class T>
inline CUptr<T>* CUmatrix<T>::v_alloc()
{
	if (N == 0) return nullptr;
	CUptr<T> *v = new CUptr<T>[nn];
	v[0] = p;
	for (Long i = 1; i<nn; i++)
		v[i] = v[i-1] + mm;
	return v;
}

template <class T>
CUmatrix<T>::CUmatrix() : nn(0), mm(0), v(nullptr) {}

template <class T>
CUmatrix<T>::CUmatrix(Long_I n, Long_I m) : Base(n*m), nn(n), mm(m), v(v_alloc()) {}

template <class T>
CUmatrix<T>::CUmatrix(Long_I n, Long_I m, const T &s) : CUmatrix(n, m)
{ cumemset<<<nbl0(N), Nth0>>>(p, s, N); }

template <class T>
CUmatrix<T>::CUmatrix(const CUmatrix<T> &rhs)
{
	error("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!")
}

template <class T>
inline Long CUmatrix<T>::nrows() const
{ return nn; }

template <class T>
inline Long CUmatrix<T>::ncols() const
{ return mm; }

template <class T>
inline CUmatrix<T> & CUmatrix<T>::operator=(const CUmatrix &rhs)
{
	if (this == &rhs) error("self assignment is forbidden!");
	if (rhs.rows() != nn || rhs.cols() != mm) error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <class T>
inline CUmatrix<T> & CUmatrix<T>::operator=(const NRmatrix<T> &rhs)
{
	if (rhs.rows() != nn || rhs.cols() != mm) error("size mismatch!");
	cudaMemcpy(p, rhs.ptr(), N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <class T>
inline CUptr<T> CUmatrix<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=nn)
		error("CUmatrix subscript out of bounds!");
#endif
	return v[i];
}

template <class T>
CUmatrix<T>::~CUmatrix()
{ if(v) delete v; }

// Matric and vector types

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

// typedef const CUmat3d<Doub> Gmat3Doub_I;
// typedef CUmat3d<Doub> Gmat3Doub, Gmat3Doub_O, Gmat3Doub_IO;

// typedef const CUmat3d<Comp> Gmat3Comp_I;
// typedef CUmat3d<Comp> Gmat3Comp, Gmat3Comp_O, Gmat3Comp_IO;
