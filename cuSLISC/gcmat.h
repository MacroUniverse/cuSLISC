// Column major matrix Class

#pragma once
#include "gvector.h"

namespace slisc {

template <typename T>
class Gcmat : public Gbase<T>
{
private:
	Long m_N1, m_N2;
public:
	typedef Gbase<T> Base;
	using Base::m_p;
	using Base::m_N;
	using Base::operator=;
	using Base::operator[];
	using Base::operator();
	Gcmat();
	Gcmat(Long_I N1, Long_I N2);
	Gcmat(Long_I N1, Long_I N2, const T &s); //Initialize to constant
	Gcmat(Cmat<T> &v); // initialize from cpu matrix
	Gcmat(const Gcmat &rhs);		// Copy constructor forbidden
	inline Long n1() const;
	inline Long n2() const;
	template <typename T1>
	inline void get(Cmat<T1> &v) const; // copy to cpu vector
	inline Gcmat & operator=(const Gcmat &rhs); //copy assignment
	inline Gcmat & operator=(const Cmat<T> &rhs); //NR assignment
	inline Gref<T> operator()(Long_I i, Long_I j);  //subscripting: pointer to row i
	inline const Gref<T> operator()(Long_I i, Long_I j) const;
	inline void resize(Long_I N1, Long_I N2); // resize (contents not preserved)
	template <typename T1>
	inline void resize(const Gcmat<T1> &v);
	template <typename T1>
	inline void resize(const Cmat<T1> &v);
	~Gcmat();
};

template <typename T>
Gcmat<T>::Gcmat() : m_N1(0), m_N2(0) {}

template <typename T>
Gcmat<T>::Gcmat(Long_I n, Long_I m) : Base(n*m), m_N1(n), m_N2(m) {}

template <typename T>
Gcmat<T>::Gcmat(Long_I n, Long_I m, const T &s) : Gcmat(n, m)
{ *this = s; }

template <typename T>
Gcmat<T>::Gcmat(Cmat<T> &v) : Gcmat(v.n1(), v.n2())
{ cudaMemcpy(m_p, v.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
Gcmat<T>::Gcmat(const Gcmat<T> &rhs)
{
	SLS_ERR("Copy constructor or move constructor is forbidden, use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline Long Gcmat<T>::n1() const
{ return m_N1; }

template <typename T>
inline Long Gcmat<T>::n2() const
{ return m_N2; }

template <typename T> template <typename T1>
inline void Gcmat<T>::get(Cmat<T1> &a) const
{
#ifdef CUSLS_CHECKTYPE
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size!");
#endif
	a.resize(m_N1, m_N2);
	cudaMemcpy(a.ptr(), m_p, m_N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline Gcmat<T> & Gcmat<T>::operator=(const Gcmat &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gcmat<T> & Gcmat<T>::operator=(const Cmat<T> &rhs)
{
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline Gref<T> Gcmat<T>::operator()(Long_I i, Long_I j)
{
#ifdef CUSLS_CHECKBOUNDS
	if (i<0 || i>=m_N1 || j<0 || j>=m_N2)
		SLS_ERR("Gcmat subscript out of bounds!");
#endif
	return (*this)[i+m_N1*j];
}

template <typename T>
inline const Gref<T> Gcmat<T>::operator()(Long_I i, Long_I j) const
{
#ifdef CUSLS_CHECKBOUNDS
	if (i<0 || i>=m_N1 || j<0 || j>=m_N2)
		SLS_ERR("Gcmat subscript out of bounds!");
#endif
	return (*this)[i+m_N1*j];
}

template <typename T>
inline void Gcmat<T>::resize(Long_I n, Long_I m)
{
	if (n != m_N1 || m != m_N2) {
		Base::resize(n*m);
		m_N1 = n; m_N2 = m;
	}
}

template<typename T> template<typename T1>
inline void Gcmat<T>::resize(const Gcmat<T1>& v)
{ resize(v.n1(), v.n2()); }

template<typename T> template<typename T1>
inline void Gcmat<T>::resize(const Cmat<T1>& v)
{ resize(v.n1(), v.n2()); }

template <typename T>
Gcmat<T>::~Gcmat()
{}

} // namespace slisc
