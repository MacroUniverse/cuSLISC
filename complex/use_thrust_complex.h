#include <thrust/complex.h>

namespace cuslisc
{

template<typename T> __host__ __device__
inline T abs(const complex<T>& z) {
	return abs((thrust::complex<T>&)z);
}

// pow() is different!

template <typename T> __host__ __device__
inline complex<T> exp(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = exp((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> sqrt(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = sqrt((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> log(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = log((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> sin(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = sin((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> cos(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = cos((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> sinh(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = sinh((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> cosh(const complex<T>& z) {
	thrust::complex<T> ret;
	ret = cosh((thrust::complex<T>&)z);
	return reinterpret_cast<complex<T>&>(ret);
}

} // namespace cuslisc
