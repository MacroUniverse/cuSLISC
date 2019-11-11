#pragma once
#include "scalar_arith.h"
#include "copy.h"

namespace slisc {

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
	p_kernel_type m_p; // pointer to the first element
	Long m_N; // number of elements
public:
	Gbase() : m_N(0), m_p(nullptr) {}
	explicit Gbase(Long_I n) : m_N(n) { cudaMalloc(&m_p, m_N*sizeof(T)); }
	p_kernel_type ptr() { return m_p; } // get pointer
	p_const_kernel_type ptr() const { return m_p; }
	Long_I size() const { return m_N; }
	inline void resize(Long_I n);
	inline Gref<T> operator[](Long_I i);
	// TODO: should return low level const
	inline const Gref<T> operator[](Long_I i) const;
	inline Gref<T> operator()(Long_I i);
	inline const Gref<T> operator()(Long_I i) const;
	inline Gref<T> end(); // last element
	inline const Gref<T> end() const;
	inline Gbase & operator=(const T &rhs); // set scalar
	~Gbase() { if (m_p) cudaFree(m_p); }
};

template <typename T>
inline void Gbase<T>::resize(Long_I n)
{
	if (n != m_N) {
		if (m_p != nullptr) cudaFree(m_p);
		m_N = n;
		if (n > 0)
			cudaMalloc(&m_p, m_N*sizeof(T));
		else
			m_p = nullptr;
	}
}

template <typename T>
inline Gref<T> Gbase<T>::operator[](Long_I i)
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=m_N)
	SLS_ERR("Gbase subscript out of bounds!");
#endif
	return Gref<T>(m_p+i);
}

template <typename T>
inline const Gref<T> Gbase<T>::operator[](Long_I i) const
{
#ifdef _CHECKBOUNDS_
if (i<0 || i>=m_N)
	SLS_ERR("Gbase subscript out of bounds!");
#endif
	return Gref<T>(m_p+i);
}

template <typename T>
inline Gref<T> Gbase<T>::operator()(Long_I i)
{
	return (*this)[i];
}

template <typename T>
inline const Gref<T> Gbase<T>::operator()(Long_I i) const
{
	return (*this)[i];
}

template <typename T>
inline Gref<T> Gbase<T>::end()
{
#ifdef _CHECKBOUNDS_
	if (m_N < 1)
		SLS_ERR("Using end() for empty object!");
#endif
	return Gref<T>(m_p+m_N-1);
}

template <typename T>
inline const Gref<T> Gbase<T>::end() const
{
#ifdef _CHECKBOUNDS_
	if (m_N < 1)
		SLS_ERR("Using end() for empty object!");
#endif
	return Gref<T>(m_p+m_N-1);
}

template <typename T>
inline Gbase<T> & Gbase<T>::operator=(const T &rhs)
{
	if (m_N) cumemset<<<nbl(Nbl_cumemset,Nth_cumemset,m_N), Nth_cumemset>>>(m_p, (kernel_type&)rhs, m_N);
	return *this;
}

// Vector Class

template <typename T>
class Gvector : public Gbase<T>
{
public:
	typedef Gbase<T> Base;
	using Base::m_p;
	using Base::m_N;
	using Base::operator=;
	using Base::operator();
	using Base::operator[];
	Gvector() {};
	explicit Gvector(Long_I n) : Base(n) {}
	Gvector(Long_I n, const T &a);	//initialize to constant value
	Gvector(Vector<T> &v); // initialize from cpu vector
	Gvector(const Gvector &rhs);	// Copy constructor forbidden
	template <typename T1>
	inline void get(Vector<T1> &v) const; // copy to cpu vector
	inline Gvector & operator=(const Gvector &rhs);	// copy assignment
	inline Gvector & operator=(const Vector<T> &v); // NR assignment
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
{ cudaMemcpy(m_p, v.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice); }

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
	if (rhs.size() != m_N)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gvector<T> & Gvector<T>::operator=(const Vector<T> &v)
{
	if (v.size() != m_N)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, v.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T> template <typename T1>
inline void Gvector<T>::get(Vector<T1> &v) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size!");
#endif
	v.resize(m_N);
	cudaMemcpy(v.ptr(), m_p, m_N*sizeof(T), cudaMemcpyDeviceToHost);
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

} // namespace slisc
