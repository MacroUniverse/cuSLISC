#include <thrust/complex.h>

namespace slisc
{

template<typename T> __host__ __device__
inline T abs(const complex<T>& z) {
	return abs((thrust::complex<T>&)z);
}

template <typename T> __host__ __device__
inline complex<T> pow(const complex<T>& x, const complex<T>& y) {
	thrust::complex<T> ret;
	ret = pow((thrust::complex<T>&)x, (thrust::complex<T>&)y);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> pow(const complex<T>& x, const T& y) {
	thrust::complex<T> ret;
	ret = pow((thrust::complex<T>&)x, y);
	return reinterpret_cast<complex<T>&>(ret);
}

template <typename T> __host__ __device__
inline complex<T> pow(const T& x, const complex<T>& y) {
	thrust::complex<T> ret;
	ret = pow(x, (thrust::complex<T>&)y);
	return reinterpret_cast<complex<T>&>(ret);
}

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

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits>&
operator<<(std::basic_ostream<charT, traits>& os, const complex<T>& z) {
	return os << (thrust::complex<T>&)z;
}

} // namespace slisc
