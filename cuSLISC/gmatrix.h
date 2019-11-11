// Matrix Class

#pragma once
#include "gvector.h"

namespace slisc {

template <typename T>
class Gmatrix : public Gbase<T>
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
	inline Gref<T> operator()(Long_I i, Long_I j);  //subscripting: pointer to row i
	inline const Gref<T> operator()(Long_I i, Long_I j) const;
	inline void resize(Long_I n, Long_I m); // resize (contents not preserved)
	template <typename T1>
	inline void resize(const Gmatrix<T1> &v);
	template <typename T1>
	inline void resize(const Matrix<T1> &v);
	~Gmatrix();
};

template <typename T>
Gmatrix<T>::Gmatrix() : m_N1(0), m_N2(0) {}

template <typename T>
Gmatrix<T>::Gmatrix(Long_I n, Long_I m) : Base(n*m), m_N1(n), m_N2(m) {}

template <typename T>
Gmatrix<T>::Gmatrix(Long_I n, Long_I m, const T &s) : Gmatrix(n, m)
{ *this = s; }

template <typename T>
Gmatrix<T>::Gmatrix(Matrix<T> &v) : Gmatrix(v.n1(), v.n2())
{ cudaMemcpy(m_p, v.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice); }

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
#ifdef CUSLS_CHECKTYPE
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size!");
#endif
	a.resize(m_N1, m_N2);
	cudaMemcpy(a.ptr(), m_p, m_N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline Gmatrix<T> & Gmatrix<T>::operator=(const Gmatrix &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gmatrix<T> & Gmatrix<T>::operator=(const Matrix<T> &rhs)
{
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline Gref<T> Gmatrix<T>::operator()(Long_I i, Long_I j)
{
#ifdef CUSLS_CHECKBOUNDS
	if (i<0 || i>=m_N1 || j<0 || j>=m_N2)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return (*this)[m_N2*i+j];
}

template <typename T>
inline const Gref<T> Gmatrix<T>::operator()(Long_I i, Long_I j) const
{
#ifdef CUSLS_CHECKBOUNDS
	if (i<0 || i>=m_N1 || j<0 || j>=m_N2)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return (*this)[m_N2*i+j];
}

template <typename T>
inline void Gmatrix<T>::resize(Long_I n, Long_I m)
{
	if (n != m_N1 || m != m_N2) {
		Base::resize(n*m);
		m_N1 = n; m_N2 = m;
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
{}

} // namespace slisc
