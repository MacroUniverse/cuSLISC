// test all cuda function
#include "test/test_global.h"
#include "test/test_complex.h"
#include "test/test_gvector.h"
#include "test/test_gmatrix.h"
#include "test/test_gcmat.h"
#include "test/test_gmat3d.h"
#include "test/test_arithmetic.h"

using std::cout; using std::endl;

__global__ void test_kernel()
{
	printf("in block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

// temporary test
void test()
{
}

int main()
{
	cout << "test_kernel()" << endl;
	test_kernel<<<2,2>>>();
	cudaDeviceSynchronize();
	cout << "done!\n\n" << endl;

	// temporary test
	test();

	// systematic tests
	cout << "test_global()" << endl;
	test_global();
	cout << "test_gvector()" << endl;
	test_gvector();
	cout << "test_complex()" << endl;
	test_complex();
	cout << "test_gmatrix()" << endl;
	test_gmatrix();
	cout << "test_gcmat()" << endl;
	test_gcmat();
	cout << "test_gmat3d()" << endl;
	test_gmat3d();
	cout << "test_arithmetic()" << endl;
	test_arithmetic();
	cout << "done testing!" << endl;
}
