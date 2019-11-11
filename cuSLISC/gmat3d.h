// 3D Matrix Class
#pragma once
#include "gvector.h"

namespace slisc {

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
	using Base::m_p;
	using Base::m_N;
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
	if (m_N == 0) return nullptr;
	Long i;
	Long nnmm = m_N1*m_N2;
	Gptr<T> *v0 = new Gptr<T>[nnmm]; v0[0] = m_p;
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
{ cudaMemcpy(m_p, v.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice); }

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
	cudaMemcpy(a.ptr(), m_p, m_N*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
inline Gmat3d<T> & Gmat3d<T>::operator=(const Gmat3d &rhs)
{
	if (this == &rhs)
		SLS_ERR("self assignment is forbidden!");
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2 || rhs.n3() != m_N3)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyDeviceToDevice);
	return *this;
}

template <typename T>
inline Gmat3d<T> & Gmat3d<T>::operator=(const Mat3d<T> &rhs)
{
	if (rhs.n1() != m_N1 || rhs.n2() != m_N2 || rhs.n3() != m_N3)
		SLS_ERR("size mismatch!");
	cudaMemcpy(m_p, rhs.ptr(), m_N*sizeof(T), cudaMemcpyHostToDevice);
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

} // namespace slisc
