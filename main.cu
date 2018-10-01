// test all cuda function

#include "cuslisc.h"
#include "nr3plus.h"
using std::cout; using std::endl; using std::string;
using std::ifstream; using std::to_string;

void test_class()
{
	// default initialize
	{
		GvecDoub gvDoub;
		if (gvDoub.size() != 0) error("failed!");
		if (gvDoub.ptr() != nullptr) error("failed!");
		// GmatDoub aDoub;
		// if (aDoub.size() != 0) error("failed!")
		// if (aDoub.nrows() != 0) error("failed!")
		// if (aDoub.ncols() != 0) error("failed!")
		// if (aDoub.ptr() != nullptr) error("failed!")
		// Gmat3Doub a3Doub;
		// if (a3Doub.size() != 0) error("failed!")
		// if (a3Doub.dim1() != 0) error("failed!")
		// if (a3Doub.dim2() != 0) error("failed!")
		// if (a3Doub.dim3() != 0) error("failed!")
		// if (a3Doub.ptr() != nullptr) error("failed!")
	}

	// size initialize
	{
		GvecDoub gvDoub(3);
		if (gvDoub.size() != 3) error("failed!");
		if (gvDoub.ptr() == nullptr) error("failed!");
		// GmatDoub aDoub(3, 3);
		// if (aDoub.size() != 9) error("failed!")
		// if (aDoub.nrows() != 3) error("failed!")
		// if (aDoub.ncols() != 3) error("failed!")
		// if (aDoub.ptr() != &aDoub[0][0]) error("failed!")
		// Gmat3Doub a3Doub(3, 3, 3);
		// if (a3Doub.size() != 27) error("failed!")
		// if (a3Doub.dim1() != 3) error("failed!")
		// if (a3Doub.dim2() != 3) error("failed!")
		// if (a3Doub.dim3() != 3) error("failed!")
		// if (a3Doub.ptr() != &a3Doub[0][0][0]) error("failed!")
	}

	// memory copy
	{
		VecDoub vDoub, vDoub1;
		linspace(vDoub, 0., 3., 4);
		GvecDoub gvDoub(4);
		gvDoub = vDoub;
		if (gvDoub.size() != 4) error("failed!");
		gvDoub.get(vDoub1);
		if (vDoub1 != vDoub) error("failed!");
	}

	// const initialize
	{
		GvecDoub gvDoub(3, 1.23);
		VecDoub vDoub;
		gvDoub.get(vDoub);
		if (vDoub.size() != 3) error("failed!");
		if (vDoub != 1.23) error("failed!")
	}

	// initialize from cpu vector/matrix
	{
		VecDoub vDoub, vDoub1; linspace(vDoub, 0., 3., 4);
		GvecDoub gvDoub(vDoub);
		gvDoub.get(vDoub1);
		if (gvDoub.size() != 4) error("failed!");
		if (vDoub1 != vDoub) error("failed!");
	}

	// resize
	{
		GvecDoub gvDoub;
		gvDoub.resize(4);
		if (gvDoub.size() != 4) error("failed!");
		if (gvDoub.ptr() == nullptr) error("failed!");
		VecDoub vDoub; linspace(vDoub, 0., 3., 4);
		gvDoub = vDoub;
		VecDoub vDoub1;
		gvDoub.get(vDoub1);
		if (vDoub1 != vDoub) error("failed!");
		gvDoub.resize(0);
		if (gvDoub.size() != 0)  error("failed!");
		if (gvDoub.ptr() != nullptr) error("failed!");
	}

	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		GvecDoub gvDoub(10);
		gvDoub = 3.14;
		VecDoub vDoub;
		gvDoub.get(vDoub);
		if (vDoub != 3.14) error("failed!");

		// copy assignment
		linspace(vDoub, 10., 1., 10);
		gvDoub = vDoub;
		GvecDoub gvDoub1(10);
		gvDoub1 = gvDoub;
		VecDoub vDoub1;
		gvDoub1.get(vDoub1);
		if (vDoub1 != vDoub) error("failed!");
	}

	// test class CUref alone
	{
		GvecDoub gvDoub(3); gvDoub = 0.;
		CUref<Doub> ref(gvDoub.ptr());
		ref = 5.6;
		VecDoub vDoub; gvDoub.get(vDoub);
		if (vDoub[0] != 5.6 || vDoub[1] != 0. || vDoub[2] != 0.) error("failed!");
		if (ref != 5.6) error("failed!");
		const CUref<Doub> ref1(gvDoub.ptr());
		if (ref != 5.6) error("failed!");
		ref += 1.1;
		cout << "ref = " << ref - 6.7 << endl;
		if (abs(ref - 6.7) > 1e-15) error("failed!");
		// TODO: -=, *=, /=
	}

	// // .end()
	// {
	// 	VecDoub vDoub(3); linspace(vDoub, 1.1, 3.3);
	// 	GvecDoub gvDoub(vDoub);
	// 	if (gvDoub.end() != 3.3) error("failed!");
	// 	gvDoub.end() = 4.4;
	// 	gvDoub.get(vDoub);
	// 	if (vDoub[0] != 1.1 || vDoub[1] != 2.2 || vDoub.end() != 4.4) error("failed!");
	// }
	
	// // operator()
	// {
	// 	VecDoub vDoub(3); linspace(vDoub, 1.1, 3.3);
	// 	GvecDoub gvDoub(vDoub);
	// 	if (gvDoub(0) != 1.1 || gvDoub(1) != 2.2 || gvDoub(2) != 3.3) error("failed!");
	// 	gvDoub.get(vDoub);
	// 	if (vDoub[0] != 1.1 || vDoub[1] != 2.2 || vDoub.end() != 4.4) error("failed!");
	// }


	// // operator[]
	// if (gvDoub[0] != 1.1)  error("failed!");
	// if (abs(gvDoub[2] - 3.3) > 1e-15)  error("failed!");
	// if (abs(gvDoub[4] - 5.5) > 1e-15)  error("failed!");
}

int main()
{
	// systematic tests
	cuInit();
	cout << "test_class()" << endl;
	test_class();
	cout << "done testing!" << endl;
	// cuInit();
	// CUvector<Doub> a(4);
	// a = 3.1;
	// cout << "a.size() = " << a.size() << endl;
	// VecDoub v;
	// a.get(v);
	// cout << "a = " << endl;
	// disp(v);

	// VecDoub v1(4); linspace(v1, 0.1, 3.4);
	// cout << "v1 = " << endl;
	// disp(v1);
	// a = v1;
	// a.get(v);
	// cout << "a = " << endl;
	// disp(v);
}
