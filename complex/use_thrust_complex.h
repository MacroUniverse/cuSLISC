#include <thrust/complex.h>

namespace cuslisc
{
	template<typename T> __host__ __device__
	inline T abs(const complex<T>& z) {
		return abs((thrust::complex<T>&)z);
	}
}
