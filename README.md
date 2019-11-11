# cuSLISC
cuSLISC is [SLISC](https://github.com/MacroUniverse/SLISC) for CUDA, fully compatible with SLISC, and the grammar is mostly simmilar. cuSLISC is tested with gcc in Ubuntu, with CUDA capability 6.0.

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
	ga(0,0) += 1.;
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

A test suit is in `main.cu`, it can also be used as a reference to cuSLISC. To build and run the project, use `make` and `./main.x` command.

## Scalar/vector/matrix class template
The class templates `gvector<T>`, `gmatrix<T>` and `gmat3d<T> ` etc. are similar to `vector<T>`, `matrix<T>` and `mat3d<T>`. GPU version of `VecDoub<T>`, `MatComp<T>` are named `GvecDoub<T>`, `GmatComp<T>` etc. In addition, scalar type on GPU are named such as `Gdoub`, `Gcomp`, etc. As with SLISC, single precision support is limited.

As SLISC containersm, GPU vectors/matrices must be resized explicitly.

Note that GPU element access from CPU (such as `operator()` and `operator[]`) is very slow, and should mainly be used for debugging purpose.

GPU data element access is realized using `CUref<T>` class, which has a implicit conversion to `T` and overloaded `operator=()`. If there is more than one user defined conversion, explicitly convertion might be need to make from `CUref<T>` to `T`.

## Writing new GPU functions
The cuSLISC classes should never be passed into or used in a kernel function (at least for now). Get the pointer to the first element of a GPU vector/matrix using `.ptr()` member function, then pass it to the kernel function as argument. For example, a simple implementation of `operator+=` is 

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
Cump (`cuslisc::complex<double>`) is the complex type to be used in kernels (although it can also be used in cpu, this is not recommended). `Cump` is basically `complex<double>`, both with a default constructor `Cump() = default;`, so that it can be declared in file scope as a `__device__` or `__constant__`, or declared as `__shared__` inside kernel, or passed by value into kernel, because those usage requires a POD type (trivial type). So keep in mind that default initialized `Cump` will not be `(0,0)`.

Fow users who don't know cuda programming, they don't need to know the existence of "Cump".

# Developer Notes

## Cump
Before, both cpu and gpu code must use "cuda_complex.h" for complex type. The disadvantage is SLISC project must be modified, and ".cpp" extension is not allowed. Thus it is best to use `std::complex<>` for cpu code, and another complex type `cuslisc::complex<>` for gpu code, aliased `Cump`.

Two options have been tested, one is directly modifying "complex.h" from STL, changing the namespace to "cuslisc", adding "\__global__" and "\__device__" before member functions. This modified file is named "cuda_complex.h".

Another option is to use a CUDA library. CUDA provides "cuComplex.h", however, it's interface is for C (I can probably write a C++ wraper, but I don't want to). There is also the "thrust" library that is basically an STL for CUDA. The `thrust::complex<>` has the same STL interface, but I cannot directly add a default constructor to it (I can if I am a super user). So I just copied the "thrust/complex.h" and some dependent files, changed the namespace, and added a default constructor. However, for more complex implementations like "abs()", "exp()", "cuslisc::complex" are reinterpret_cast-ed into "thrust::complex" and then use "thrust" library to calculate. I was hoping "reinterpret_cast" will not have any runtime overhead.

After testing in "cuCn3D" project, it turns out that the first option is about 20% faster. I'm not sure whether this is because of the "reinterpret_cast" or "thrust::complex" is just not efficient enough. Anyway, I should stick with the first solution for now.

## Known Bugs
none.

## Dependency update
Do not update any dependencies directly, always update in their own project then copy here.
