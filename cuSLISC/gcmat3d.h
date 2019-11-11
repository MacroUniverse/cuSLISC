// 3D Matrix Class
#pragma once
#include "gvector.h"

namespace slisc {

template <typename T>
class Gcmat3d : public Gbase<T>
{
private:
	Long m_N1;
	Long m_N2;
	Long m_N3;
public:
	typedef Gbase<T> Base;
	using Base::m_p;
	using Base::m_N;
	using Base::operator=;
	using Base::operator();
	using Base::operator[];
	Gcmat3d();
	Gcmat3d(Long_I n, Long_I m, Long_I k);
	Gcmat3d(Long_I n, Long_I m, Long_I k, const T &a); //Initialize to constant
	Gcmat3d(Cmat3d<T> &v); // initialize from cpu matrix
	Gcmat3d(const Gcmat3d &rhs);   // Copy constructor forbidden
	inline Long n1() const;
	inline Long n2() const;
	inline Long n3() const;
	template <typename T1>
	inline void get(Cmat3d<T1> &v) const; // copy to cpu matrix
	inline Gcmat3d & operator=(const Gcmat3d &rhs);	//copy assignment
	inline Gcmat3d & operator=(const Cmat3d<T> &rhs); //NR assignment
	inline Gref<T> operator()(Long_I i, Long_I j, Long_I k);	//subscripting: pointer to row i
	inline const Gref<T> operator()(Long_I i, Long_I j, Long_I k) const;
	inline void resize(Long_I n, Long_I m, Long_I k);
	template <typename T1>
	inline void resize(const Gcmat3d<T1> &v);
	template <typename T1>
	inline void resize(const Cmat3d<T1> &v);
	~Gcmat3d();
};

template <typename T>
Gcmat3d<T>::Gcmat3d() : m_N1(0), m_N2(0), m_N3(0) {}

template <typename T>
Gcmat3d<T>::Gcmat3d(Long_I n, Long_I m, Long_I k) :
Base(n*m*k), m_N1(n), m_N2(m), m_N3(k) {}

template <typename T>
Gcmat3d<T>::Gcmat3d(Long_I n, Long_I m, Long_I k, const T &s) : Gcmat3d(n, m, k)
{ *this = s; }

template <typename T>
Gcmat3d<T>::Gcmat3d(Cmat3d<T> &v) : Gcmat3d(v.n1(), v.n2(), v.n3())
{ cudaMemcpy(m_p, v.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice); }

template <typename T>
Gcmat3d<T>::Gcmat3d(const Gcmat3d<T> &rhs)
{
	SLS_ERR("Copy constructor or move constructor is forbidden, "
	"use reference argument for function input or output, and use \"=\" to copy!");
}

template <typename T>
inline Long Gcmat3d<T>::n1() const
{ return m_N1; }

template <typename T>
inline Long Gcmat3d<T>::n2() const
{ return m_N2; }

template <typename T>
inline Long Gcmat3d<T>::n3() const
{ return m_N3; }

template <typename T> template <typename T1>
inline void Gcmat3d<T>::get(Cmat3d<T1> &a) const
{
#ifdef _CHECKTYPE_
	if (sizeof(T) != sizeof(T1))
		SLS_ERR("wrong type size");
#endif
	a.resize(m_N1, m_N2, m_N3);
	cudaMemcpy(a.ptr(), m_p, m_N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline Gcmat3d<T> & Gcmat3d<T>::operator=(const Gcmat3d &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2 || rhs.n3() != m_N3)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gcmat3d<T> & Gcmat3d<T>::operator=(const Cmat3d<T> &rhs)
{
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2 || rhs.n3() != m_N3)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice);
	return *this;
}

template <typename T>
inline Gref<T> Gcmat3d<T>::operator()(Long_I i, Long_I j, Long_I k)
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=m_N1 || j<0 || j>=m_N2 || k<0 || k>=m_N3)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return (*this)[i + m_N1*j + m_N1*m_N2*k];
}

template <typename T>
inline const Gref<T> Gcmat3d<T>::operator()(Long_I i, Long_I j, Long_I k) const
{
#ifdef _CHECKBOUNDS_
	if (i<0 || i>=m_N1 || j<0 || j>=m_N2 || k<0 || k>=m_N3)
		SLS_ERR("Gmatrix subscript out of bounds!");
#endif
	return (*this)[i + m_N1*j + m_N1*m_N2*k];
}

template <typename T>
inline void Gcmat3d<T>::resize(Long_I n, Long_I m, Long_I k)
{
	if (n != m_N1 || m != m_N2 || k != m_N3) {
		Base::resize(n*m*k);
		m_N1 = n; m_N2 = m; m_N3 = k;
	}
}

template<typename T> template<typename T1>
inline void Gcmat3d<T>::resize(const Gcmat3d<T1>& v)
{ resize(v.n1(), v.n2(), v.n3()); }

template<typename T> template<typename T1>
inline void Gcmat3d<T>::resize(const Cmat3d<T1>& v)
{ resize(v.n1(), v.n2(), v.n3()); }

template <typename T>
Gcmat3d<T>::~Gcmat3d()
{}

} // namespace slisc
