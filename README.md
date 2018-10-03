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
	// copy GPU matrix back to CPU matrix
	ga.get(a); // will implement "a = ga;" in the future
	// print matrix again
	cout << "a = \n"; disp(a);
	// resize GPU matrix
	ga.resize(2,3);
	// set all elements of GPU matrix
	ga = 3.1;
}
```

A test suit is in "main.cu", it can also be used as a reference to cuSLISC. To build and run the project, use, "make" and "./main.x" command.

## vector/matrix class template
The class templates CUvector<T>, CUmatrix<T> and CUmat3d<T> are similar to NRvector<T>, NRmatrix<T> and NRmat3d<T>. GPU version of "VecDoub", "MatComp" are named "GvecDoub", "GmatComp" etc.

One main difference from SLISC is that GPU vectors/matrices must be resized explicitly.

Note that GPU element access from CPU (such as operator() and operator[]) is very slow, and should mainly be used for debugging purpose.

## Known Bugs
"Comp s{};" will not work inside kernel, need to use "Comp s; s = 0.;"
