#include "../cuSLISC/complex.h"

__global__
void test_complex_kernel(Cump *v, Long N, Cump s1, Cump s2)
{
	__shared__ Cump cache[4];
	cache[0] = s1 + s2; v[0] = cache[0];
	cache[1] = s1 - s2; v[1] = cache[1];
	cache[2] = s1 * s2; v[2] = cache[2];
	cache[3] = s1 / s2; v[3] = cache[3];
	v[4] = abs(s1);
	v[5] = pow(s1, 2.);
	v[6] = pow(s1, s2);
	v[7] = exp(s1);
	v[8] = real(s1);
	v[9] = imag(s1);
}

void test_complex()
{
	Int N = 10;
	GvecComp gv(N, 0.); Cump s1(1.1, 2.2), s2(2.2, 4.4);
	test_complex_kernel<<<1,1>>>(gv.ptr(), N, s1, s2);
	VecComp v; v = gv;
	if (v[0] != (Comp)s1 + (Comp)s2) error("failed!");
	if (v[1] != (Comp)s1 - (Comp)s2) error("failed!");
	if (v[2] != (Comp)s1 * (Comp)s2) error("failed!");
	if (v[3] != (Comp)s1 / (Comp)s2) error("failed!");
	if (v[4] != abs((Comp)s1)) error("failed!");
	if (abs(v[5] - pow((Comp)s1, 2.)) > 1e-15) error("failed!");
	if (abs(v[6] - pow((Comp)s1, (Comp)s2)) > 1e-15) error("failed!");
	if (v[7] != exp((Comp)s1)) error("failed!");
	if (v[8] != real((Comp)s1)) error("failed!");
	if (v[9] != imag((Comp)s1)) error("failed!");
}
