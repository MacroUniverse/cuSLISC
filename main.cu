// test all cuda function

#include "cusliscplus.h"
using std::cout; using std::endl; using std::string;
using std::ifstream; using std::to_string;

void test_vector()
{
	// default initialize
	{
		GvecDoub gvDoub;
		if (gvDoub.size() != 0) error("failed!");
		if (gvDoub.ptr() != nullptr) error("failed!");
	}

	// size initialize
	{
		GvecDoub gvDoub(3);
		if (gvDoub.size() != 3) error("failed!");
		if (gvDoub.ptr() == nullptr) error("failed!");
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
		GvecDoub gvDoub(3); gvDoub = 1.1;
		CUref<Doub> ref(gvDoub.ptr());
		if (ref.ptr() != gvDoub.ptr()) error("failed!");
		ref = 5.6;
		VecDoub vDoub; gvDoub.get(vDoub);
		if (vDoub[0] != 5.6 || vDoub[1] != 1.1 || vDoub[2] != 1.1) error("failed!");
		if (ref != 5.6) error("failed!");
		const CUref<Doub> ref1(gvDoub.ptr());
		if (ref != 5.6) error("failed!");
		ref += 1.1;
		if (abs(ref - 6.7) > 2e-15) error("failed!");
		ref -= 1.7;
		if (abs(ref - 5.) > 2e-15) error("failed!");
		ref *= 2.;
		if (abs(ref - 10.) > 2e-15) error("failed!");
		ref /= 5.;
		if (abs(ref - 2.) > 2e-15) error("failed!");
	}

	// test class CUptr alone
	{
		CUptr<Doub> ptr0; // default constructor
		if (ptr0.ptr()) error("failed!");
		GvecDoub gvDoub(3); gvDoub = 1.1;
		CUptr<Doub> ptr(gvDoub[0].ptr()); // pointer constructor
		if (ptr.ptr() != gvDoub.ptr()) error("failed!");
		if (*ptr != 1.1) error("failed!"); // dereference
		// operator[]
		if (ptr[0] != 1.1 || ptr[1] != 1.1 || ptr[2] != 1.1) error("failed!"); 
		ptr[1] = 2.2; ptr[2] = 3.3;
		if (ptr[0] != 1.1 || ptr[1] != 2.2 || ptr[2] != 3.3) error("failed!");
		ptr0 = ptr; // copy assignment
		if (ptr0.ptr() != ptr.ptr()) error("failed!");
		ptr0 = ptr[1].ptr(); // T* assignment
		if (*ptr0 != 2.2) error("failed!");
		// pointer arithmatics
		ptr0 = ptr0 + 1;
		if (*ptr0 != 3.3) error("failed!");
		ptr0 = ptr0 - 2;
		if (*ptr0 != 1.1) error("failed!");
		ptr0 += 2;
		if (*ptr0 != 3.3) error("failed!");
		ptr0 -= 1;
		if (*ptr0 != 2.2) error("failed!");
	}

	// .end()
	{
		VecDoub vDoub(3); linspace(vDoub, 1.1, 3.3);
		GvecDoub gvDoub(vDoub);
		if (gvDoub.end() != 3.3) error("failed!");
		gvDoub.end() = 4.4;
		gvDoub.get(vDoub);
		if (vDoub[0] != 1.1 || vDoub[1] != 2.2 || vDoub.end() != 4.4) error("failed!");
	}
	
	// operator()
	{
		VecDoub vDoub(4); vDoub[0] = 1.1; vDoub[1] = 2.2; vDoub[2] = 3.3; vDoub[3] = 4.4;
		GvecDoub gvDoub(vDoub);
		if (gvDoub(0) != 1.1 || gvDoub(1) != 2.2) error("failed!");
		if (gvDoub(2) != 3.3 || gvDoub(3) != 4.4) error("failed!");
		gvDoub(0) *= 4.; gvDoub(1) -= 2.2; gvDoub(2) += 1.1; gvDoub(3) /= 2.2;
		gvDoub.get(vDoub);
		if (abs(vDoub[0] - 4.4) > 2e-15) error("failed!");
		if (abs(vDoub[1]) > 2e-15) error("failed!");
		if (abs(vDoub[2] - 4.4) > 2e-15) error("failed!");
		if (abs(vDoub[3] - 2.) > 2e-15) error("failed!");
	}

	// operator[]
	{
		VecDoub vDoub(4); vDoub[0] = 1.1; vDoub[1] = 2.2; vDoub[2] = 3.3; vDoub[3] = 4.4;
		GvecDoub gvDoub(vDoub);
		if (gvDoub[0] != 1.1 || gvDoub[1] != 2.2) error("failed!");
		if (gvDoub[2] != 3.3 || gvDoub[3] != 4.4) error("failed!");
		gvDoub[0] *= 4.; gvDoub[1] -= 2.2; gvDoub[2] += 1.1; gvDoub[3] /= 2.2;
		gvDoub.get(vDoub);
		if (abs(vDoub[0] - 4.4) > 2e-15) error("failed!");
		if (abs(vDoub[1]) > 2e-15) error("failed!");
		if (abs(vDoub[2] - 4.4) > 2e-15) error("failed!");
		if (abs(vDoub[3] - 2.) > 2e-15) error("failed!");
	}

	// resize
	{
		VecDoub vDoub(4);
		GvecDoub gvDoub;
		if (gvDoub.size() != 0) error("failed!");
		gvDoub.resize(2);
		if (gvDoub.size() != 2) error("failed!");
		gvDoub[1] = 1.3;
		if (gvDoub(1) != 1.3) error("failed!");
		gvDoub.resize(vDoub);
		if (gvDoub.size() != 4) error("failed!");
		gvDoub[3] = 2.4;
		if (gvDoub(3) != 2.4) error("failed!");
		GvecDoub gvDoub1(10);
		gvDoub.resize(gvDoub1);
		if (gvDoub.size() != 10) error("failed!");
		gvDoub(9) = 5.5;
		if (gvDoub[9] != 5.5) error("failed!");
	}
}

void test_matrix()
{
	// default initialize
	{
		GmatDoub gaDoub;
		if (gaDoub.size() != 0) error("failed!");
		if (gaDoub.ptr() != nullptr) error("failed!");
	}

	// size initialize
	{
		GmatDoub gaDoub(2,3);
		if (gaDoub.size() != 6) error("failed!");
		if (gaDoub.nrows() != 2) error("failed!");
		if (gaDoub.ncols() != 3) error("failed!");
		if (gaDoub.ptr() == nullptr) error("failed!");
	}

	// memory copy
	{
		MatDoub aDoub(2,2), aDoub1;
		linspace(aDoub, 0., 3.);
		GmatDoub gaDoub(2,2);
		gaDoub = aDoub;
		if (gaDoub.size() != 4) error("failed!");
		if (gaDoub.nrows() != 2) error("failed!");
		if (gaDoub.ncols() != 2) error("failed!");
		gaDoub.get(aDoub1);
		if (aDoub1 != aDoub) error("failed!");
	}

	// const initialize
	{
		GmatDoub gaDoub(2, 3, 1.23);
		MatDoub aDoub;
		gaDoub.get(aDoub);
		if (aDoub.size() != 6) error("failed!");
		if (aDoub.nrows() != 2 || aDoub.ncols() != 3) error("failed!");
		if (aDoub != 1.23) error("failed!");
	}

	// initialize from cpu vector/matrix
	{
		MatDoub aDoub(2,3), aDoub1; linspace(aDoub, 0., 5.);
		GmatDoub gaDoub(aDoub);
		gaDoub.get(aDoub1);
		if (gaDoub.size() != 6) error("failed!");
		if (gaDoub.nrows() != 2 || gaDoub.ncols() != 3) error("failed!");
		if (aDoub1 != aDoub) error("failed!");
	}

	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		GmatDoub gaDoub(2, 3);
		gaDoub = 3.14;
		MatDoub aDoub;
		gaDoub.get(aDoub);
		if (aDoub != 3.14) error("failed!");

		// copy assignment
		linspace(aDoub, 6., 1., 2, 3);
		gaDoub = aDoub;
		GmatDoub gaDoub1(2, 3);
		gaDoub1 = gaDoub;
		MatDoub aDoub1;
		gaDoub1.get(aDoub1);
		if (aDoub1 != aDoub) error("failed!");
	}

	// .end()
	{
		MatDoub aDoub(3, 3); linspace(aDoub, 1.1, 9.9);
		GmatDoub gaDoub(aDoub);
		if (gaDoub.end() != 9.9) error("failed!");
		gaDoub.end() = 10.10;
		gaDoub.get(aDoub);
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 10.10) error("failed!");
	}

	// operator()
	{
		MatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GmatDoub gaDoub(aDoub);
		if (gaDoub(0) != 1.1 || gaDoub(1) != 2.2) error("failed!");
		if (gaDoub(2) != 3.3 || gaDoub(3) != 4.4) error("failed!");
		gaDoub(0) *= 4.; gaDoub(1) -= 2.2; gaDoub(2) += 1.1; gaDoub(3) /= 2.2;
		gaDoub.get(aDoub);
		if (abs(aDoub(0) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(1)) > 2e-15) error("failed!");
		if (abs(aDoub(2) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(3) - 2.) > 2e-15) error("failed!");
	}

	// operator[]
	{
		MatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GmatDoub gaDoub(aDoub);
		if (gaDoub[0][0] != 1.1 || gaDoub[0][1] != 2.2) error("failed!");
		if (gaDoub[1][0] != 3.3 || gaDoub[1][1] != 4.4) error("failed!");
		gaDoub[0][0] *= 4.; gaDoub[0][1] -= 2.2; gaDoub[1][0] += 1.1; gaDoub[1][1] /= 2.2;
		gaDoub.get(aDoub);
		if (abs(aDoub(0) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(1)) > 2e-15) error("failed!");
		if (abs(aDoub(2) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(3) - 2.) > 2e-15) error("failed!");
	}

	// resize
	{
		// resize(n, m)
		GmatDoub gaDoub;
		gaDoub.resize(2, 3);
		if (gaDoub.size() != 6) error("failed!");
		if (gaDoub.nrows() != 2 || gaDoub.ncols() != 3) error("failed!");
		if (gaDoub.ptr() == nullptr) error("failed!");
		MatDoub aDoub(2, 3); linspace(aDoub, 0., 5.);
		gaDoub = aDoub;
		MatDoub aDoub1;
		gaDoub.get(aDoub1);
		if (aDoub1 != aDoub) error("failed!");
		gaDoub.resize(0, 0);
		if (gaDoub.size() != 0)  error("failed!");
		if (gaDoub.nrows() != 0|| gaDoub.ncols() != 0) error("failed!");
		if (gaDoub.ptr() != nullptr) error("failed!");
		gaDoub.resize(0, 100);
		if (gaDoub.size() != 0)  error("failed!");
		if (gaDoub.nrows() != 0 || gaDoub.ncols() != 100) error("failed!");
		if (gaDoub.ptr() != nullptr) error("failed!");
		// resize(CUmatrix<>)
		gaDoub.resize(2, 3);
		GmatComp gaComp;
		gaComp.resize(gaDoub);
		if (gaComp.size() != 6)  error("failed!");
		if (gaComp.nrows() != 2 || gaComp.ncols() != 3) error("failed!");
		// resize(NRmatrix<>)
		aDoub.resize(4,5);
		gaComp.resize(aDoub);
		if (gaComp.size() != 20)  error("failed!");
		if (gaComp.nrows() != 4 || gaComp.ncols() != 5) error("failed!");
	}
}

void test_mat3d()
{
	// default initialize
	{
		Gmat3Doub gaDoub;
		if (gaDoub.size() != 0) error("failed!");
		if (gaDoub.ptr() != nullptr) error("failed!");
	}

	// size initialize
	{
		Gmat3Doub gaDoub(2,3,4);
		if (gaDoub.size() != 24) error("failed!");
		if (gaDoub.dim1() != 2) error("failed!");
		if (gaDoub.dim2() != 3) error("failed!");
		if (gaDoub.dim3() != 4) error("failed!");
		if (gaDoub.ptr() == nullptr) error("failed!");
	}

	// memory copy
	{
		Mat3Doub aDoub(2,3,4), aDoub1;
		linspace(aDoub, 0., 23.);
		Gmat3Doub gaDoub(2,3,4);
		gaDoub = aDoub;
		if (gaDoub.size() != 24) error("failed!");
		if (gaDoub.dim1() != 2) error("failed!");
		if (gaDoub.dim2() != 3) error("failed!");
		if (gaDoub.dim3() != 4) error("failed!");
		gaDoub.get(aDoub1);
		if (aDoub1 != aDoub) error("failed!");
	}

	// const initialize
	{
		Gmat3Doub gaDoub(2, 3, 4, 1.23);
		Mat3Doub aDoub;
		gaDoub.get(aDoub);
		if (aDoub.size() != 24) error("failed!");
		if (aDoub.dim1() != 2 || aDoub.dim2() != 3 || aDoub.dim3() != 4)
			error("failed!");
		if (aDoub != 1.23) error("failed!");
	}

	// initialize from cpu vector/matrix
	{
		Mat3Doub aDoub(2,3,4), aDoub1; linspace(aDoub, 0., 23.);
		Gmat3Doub gaDoub(aDoub);
		gaDoub.get(aDoub1);
		if (gaDoub.size() != 24) error("failed!");
		if (gaDoub.dim1() != 2 || gaDoub.dim2() != 3 || gaDoub.dim3() != 4)
			error("failed!");
		if (aDoub1 != aDoub) error("failed!");
	}

	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		Gmat3Doub gaDoub(2, 3, 4);
		gaDoub = 3.14;
		Mat3Doub aDoub;
		gaDoub.get(aDoub);
		if (aDoub != 3.14) error("failed!");

		// copy assignment
		aDoub.resize(2,3,4);
		linspace(aDoub, 24., 1.);
		gaDoub = aDoub;
		Gmat3Doub gaDoub1(2, 3, 4);
		gaDoub1 = gaDoub;
		Mat3Doub aDoub1;
		gaDoub1.get(aDoub1);
		if (aDoub1 != aDoub) error("failed!");
	}

	// .end()
	{
		Mat3Doub aDoub(2,2,2); linspace(aDoub, 1.1, 8.8);
		Gmat3Doub gaDoub(aDoub);
		if (gaDoub.end() != 8.8) error("failed!");
		gaDoub.end() = 9.9;
		gaDoub.get(aDoub);
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 9.9)
			error("failed!");
	}

	// operator()
	{
		Mat3Doub aDoub(2,2,2); aDoub(0) = 1.1; aDoub(2) = 2.2; aDoub(3) = 3.3; aDoub(7) = 4.4;
		Gmat3Doub gaDoub(aDoub);
		if (gaDoub(0) != 1.1 || gaDoub(2) != 2.2) error("failed!");
		if (gaDoub(3) != 3.3 || gaDoub(7) != 4.4) error("failed!");
		gaDoub(0) *= 4.; gaDoub(2) -= 2.2; gaDoub(3) += 1.1; gaDoub(7) /= 2.2;
		gaDoub.get(aDoub);
		if (abs(aDoub(0) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(2)) > 2e-15) error("failed!");
		if (abs(aDoub(3) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(7) - 2.) > 2e-15) error("failed!");
	}

	// operator[]
	{
		Mat3Doub aDoub(2,2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(4) = 3.3; aDoub(5) = 4.4;
		Gmat3Doub gaDoub(aDoub);
		if (gaDoub[0][0][0] != 1.1 || gaDoub[0][0][1] != 2.2) error("failed!");
		if (gaDoub[1][0][0] != 3.3 || gaDoub[1][0][1] != 4.4) error("failed!");
		gaDoub[0][0][0] *= 4.; gaDoub[0][0][1] -= 2.2;
		gaDoub[1][0][0] += 1.1; gaDoub[1][0][1] /= 2.2;
		gaDoub.get(aDoub);
		if (abs(aDoub(0) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(1)) > 2e-15) error("failed!");
		if (abs(aDoub(4) - 4.4) > 2e-15) error("failed!");
		if (abs(aDoub(5) - 2.) > 2e-15) error("failed!");
	}

	// resize
	{
		// resize(n, m)
		Gmat3Doub gaDoub;
		gaDoub.resize(2, 3, 4);
		if (gaDoub.size() != 24) error("failed!");
		if (gaDoub.dim1() != 2 || gaDoub.dim2() != 3 || gaDoub.dim3() != 4)
			error("failed!");
		if (gaDoub.ptr() == nullptr) error("failed!");
		Mat3Doub aDoub(2, 3, 4); linspace(aDoub, 0., 23.);
		gaDoub = aDoub;
		Mat3Doub aDoub1;
		gaDoub.get(aDoub1);
		if (aDoub1 != aDoub) error("failed!");
		gaDoub.resize(0, 0, 0);
		if (gaDoub.size() != 0)  error("failed!");
		if (gaDoub.dim1() != 0|| gaDoub.dim2() != 0 || gaDoub.dim3() != 0) error("failed!");
		if (gaDoub.ptr() != nullptr) error("failed!");
		gaDoub.resize(0, 100, 101);
		if (gaDoub.size() != 0)  error("failed!");
		if (gaDoub.dim1() != 0 || gaDoub.dim2() != 100 || gaDoub.dim3() != 101)
			error("failed!");
		if (gaDoub.ptr() != nullptr) error("failed!");
		// resize(CUmatrix<>)
		gaDoub.resize(2, 3, 4);
		Gmat3Comp gaComp;
		gaComp.resize(gaDoub);
		if (gaComp.size() != 24)  error("failed!");
		if (gaComp.dim1() != 2 || gaComp.dim2() != 3 || gaComp.dim3() != 4)
			error("failed!");
		// resize(NRmatrix<>)
		aDoub.resize(4,5,6);
		gaComp.resize(aDoub);
		if (gaComp.size() != 120)  error("failed!");
		if (gaComp.dim1() != 4 || gaComp.dim2() != 5 || gaComp.dim3() != 6)
			error("failed!");
	}
}

// test basic operations
void test_basic()
{
	// plus(v, v1, v2)
	{
		// test sum_kernel()
		GmatComp ga(10,10), ga1(10,10,Comp(1.,1.));
		GmatDoub ga2(10,10,2.);
		plus(ga, ga1, ga2);
		MatComp a; ga.get(a);
		MatComp a1(10,10,Comp(3.,1.));
		if (a != a1) error("failed!");
	}

	// sum(v)
	{
		GmatComp gv(2, 2, Comp(1.,2.));
		cout << "sum(gv) = " << sum(gv) << endl;
	}
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
	cout << "test_vector()" << endl;
	test_vector();
	cout << "test_matrix()" << endl;
	test_matrix();
	cout << "test_mat3d()" << endl;
	test_mat3d();
	cout << "test_basic()" << endl;
	test_basic();
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
