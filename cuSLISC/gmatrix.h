// Matrix Class

#pragma once
#include "gvector.h"

namespace slisc {

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

} // namespace slisc
