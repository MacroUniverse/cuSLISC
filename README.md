# cuSLISC
cuSLISC is SLISC (Scientific Library In Simple C++) for CUDA, fully compatible with SLISC, and the grammar is mostly simmilar. cuSLISC is tested with gcc in Ubuntu, with CUDA capability 6.0.

A simple example :

```cpp
#include "cuslisc.h"
using std::cout; using std::endl;

int main()
{
	// create 2x2 CPU matrix, double type, row major.
	MatDoub a(2,2);
	// let elements linearly spaced from 0 to 3
	linspace(a, 0., 3.);
	// print matrix
	cout << "a = \n"; disp(a);
	// create 2x2 GPU matrix
	GmatDoub ga(2,2);
	// copy CPU matrix to GPU matrix
	ga = a;
	// change first element
	ga[0][0] += 1.;
	// use "()" for single indexing, use ".end()" to access the last element
	ga.end() = ga(2) + 1;
	// copy GPU matrix back to CPU matrix and print
	a = ga;	cout << "a = \n"; disp(a);
	// resize GPU matrix
	ga.resize(2,3);
	// set all elements of GPU matrix
	ga = 3.1;
	// create another 2x3 complex CPU and GPU matrix
	MatComp a1(2,3); linspace(a1, 0., Comp(5.,5.));
	GmatComp ga1(2,3); ga1 = a1;
	// calculate sum of elements with GPU
	cout << "sum(ga1) = " << sum(ga1) << endl;
	// add ga to ga1 with GPU and print result
	ga1 += ga; a1 = ga1; cout << "ga1 = \n"; disp(a1);
}
```

A test suit is in "main.cu", it can also be used as a reference to cuSLISC. To build and run the project, use, "make" and "./main.x" command.

## Scalar/vector/matrix class template
The class templates CUvector\<T>, CUmatrix\<T> and CUmat3d\<T> are similar to NRvector\<T>, NRmatrix\<T> and NRmat3d\<T>. GPU version of "VecDoub", "MatComp" are named "GvecDoub", "GmatComp" etc. In addition, scalar type on GPU are named such as "Gdoub", "Gcomp", etc.

One main difference from SLISC is that GPU vectors/matrices must be resized explicitly.

Note that GPU element access from CPU (such as operator() and operator[]) is very slow, and should mainly be used for debugging purpose.

GPU data element access is realized using "CUref\<T>" class, which has a implicit conversion to "T" and overloaded "operator=()". If there is more than one user defined conversion, explicitly convertion might be need to make from "CUref\<T>" to "T".

## Writing new GPU functions
The cuSLISC classes should never be passed into or used in a kernel function (at least for now). Get the pointer to the first element of a GPU vector/matrix using ".ptr()" member function, then pass it to the kernel function as argument. For example, a simple implementation of operator "+=" is 

```cpp
// Note that "Doub" is double, "Int" is 32-bit int, "Long" is 64-bit int.
__global__ void plus_equals_kernel(Doub *v, Doub *v1, Long N)
{
	Int i, stride, ind;
	ind = blockIdx.x * blockDim.x + threadIdx.x;
	stride = gridDim.x * blockDim.x;
	for (i=ind; i<N; i+=stride)
		v[i] += v1[i];
}

inline void operator+=(GvecDoub &v, GvecDoub_I &v1)
{
	Int N = v.size();
	plus_equals_kernel<<<(N+255)/256, 256>>>(v.ptr(), v1.ptr(), N);
}
```

### Complex type for kernel
Cump (cuslisc::complex\<double>) is the complex type to be used in kernels (although it can also be used in cpu, this is not recommended). Cump is basically thrust::complex\<double>, buth with a default constructor "Cump() = default;", so that it can be declared in file scope as a "__device__" or "__constant__", or declared as "__shared__" inside kernel, or passed by value into kernel, because those usage requires a POD type (trivial type). So keep in mind that default initialized "Cump" will not be "(0,0)".

### MatFile project
cuSLISC is compatible with MatFile

# Developer Notes

## Cump
Before, both cpu and gpu code must use "cuda_complex.h" for complex type. The disadvantage is SLISC project must be modified, and ".cpp" extension is now allowed.

Thus it is best to use "std::complex" for cpu code, and another complex type for gpu code. CUDA provides "cuComplex.h", however, it's grammar is too ugly. I want to use the same grammar, but just a different type name. So, the best solution is to use "cuda_complex.h" in gpu code only, let's name it "Cump" instead of "Comp". Fow users who doesn't know cuda programming, they should not need to know the existence of "Cump", so there should not be types such as "CUbase\<Cump>". It is necessary that "CUref\<Comp>", "CUptr\<Comp>" and "CUbase\<Comp>" are specialized, because they need to have a "Cump*" member.

## Known Bugs
"Comp s{};" will not work inside kernel, need to use "Comp s; s = 0.;"

## Dependency update
Do not update any dependencies directly, always update in their own project then copy here.
