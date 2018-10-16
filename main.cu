// test all cuda function
#include <iostream>
#include "test/test_global.h"
#include "test/test_vector.h"
#include "test/test_complex.h"
#include "test/test_matrix.h"
#include "test/test_mat3d.h"
#include "test/test_basic.h"

using std::cout; using std::endl; using std::string;
using std::ifstream; using std::to_string;

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
	cout << "test gpu global var..." << endl;
	test_global();
	cout << "test scalar and vector..." << endl;
	test_vector();
	cout << "test Cump..." << endl;
	test_complex();
	cout << "test matrix..." << endl;
	test_matrix();
	cout << "test mat3d..." << endl;
	test_mat3d();
	cout << "test basic..." << endl;
	test_basic();
	cout << "done testing!" << endl;
}
