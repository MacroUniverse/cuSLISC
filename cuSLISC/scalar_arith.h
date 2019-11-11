#pragma once
#include "global.h"

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
// #ifdef CUSLS_CHECKSETSYS
// 	if (getsym(sym) != val1)
//		SLS_ERR("failed!");
// #endif
// }

template <typename T>
inline void setsym(T &sym, const T &val)
{
	cudaMemcpyToSymbol(sym, &val, sizeof(T));
#ifdef CUSLS_CHECKSETSYS
	if (getsym(sym) != val)
		SLS_ERR("failed!");
#endif
}

inline void setsym(Cump &sym, Comp_I &val)
{
	cudaMemcpyToSymbol(sym, &val, sizeof(Comp));
#ifdef CUSLS_CHECKSETSYS
	if ( getsym(sym) != val )
		SLS_ERR("failed!");
#endif
}

// calculate number of CUDA blocks needed
inline Int nbl(Int NblMax, Int Nth, Int N)
{ return min(NblMax, (N + Nth - 1)/Nth); }

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
	inline operator T() const; // convert to type T
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

} // namespace slisc
