#pragma once
#include "../cuSLISC/cuslisc.h"
#include "../SLISC/arithmetic.h"
#include "../SLISC/disp.h"

void test_gcmat3d()
{
	using namespace slisc;
	// default initialize
	{
		Gcmat3Doub gaDoub;
		if (gaDoub.size() != 0)
			SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr)
			SLS_ERR("failed!");
	}

	// size initialize
	{
		Gcmat3Doub gaDoub(2,3,4);
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2)
			SLS_ERR("failed!");
		if (gaDoub.n2() != 3)
			SLS_ERR("failed!");
		if (gaDoub.n3() != 4)
			SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr)
			SLS_ERR("failed!");
	}

	// memory copy
	{
		Cmat3Doub aDoub(2,3,4), aDoub1(2,3,4);
		linspace(aDoub, 0., 23.);
		Gcmat3Doub gaDoub(2,3,4);
		gaDoub = aDoub;
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2)
			SLS_ERR("failed!");
		if (gaDoub.n2() != 3)
			SLS_ERR("failed!");
		if (gaDoub.n3() != 4)
			SLS_ERR("failed!");
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
	}

	// const initialize
	{
		Gcmat3Doub gaDoub(2, 3, 4, 1.23);
		Cmat3Doub aDoub(2, 3, 4);
		aDoub = gaDoub;
		if (aDoub.size() != 24)
			SLS_ERR("failed!");
		if (aDoub.n1() != 2 || aDoub.n2() != 3 || aDoub.n3() != 4)
			SLS_ERR("failed!");
		if (aDoub != 1.23)
			SLS_ERR("failed!");
	}

	// initialize from cpu vector/matrix
	{
		Cmat3Doub aDoub(2,3,4), aDoub1(2,3,4); linspace(aDoub, 0., 23.);
		Gcmat3Doub gaDoub(aDoub);
		aDoub1 = gaDoub;
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3 || gaDoub.n3() != 4)
			SLS_ERR("failed!");
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
	}

	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		Gcmat3Doub gaDoub(2, 3, 4);
		gaDoub = 3.14;
		Cmat3Doub aDoub(2,4,5);
		aDoub = gaDoub;
		if (aDoub != 3.14)
			SLS_ERR("failed!");

		// copy assignment
		linspace(aDoub, 24., 1.);
		gaDoub = aDoub;
		Gcmat3Doub gaDoub1(2,3,4);
		gaDoub1 = gaDoub;
		Cmat3Doub aDoub1(2,3,4);
		aDoub1 = gaDoub1;
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
	}

	// .end()
	{
		Cmat3Doub aDoub(2,2,2); linspace(aDoub, 1.1, 8.8);
		Gcmat3Doub gaDoub(aDoub);
		if (gaDoub.end() != 8.8)
			SLS_ERR("failed!");
		gaDoub.end() = 9.9;
		aDoub = gaDoub;
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 9.9)
			SLS_ERR("failed!");
	}

	// operator()
	{
		Cmat3Doub aDoub(2,2,2); aDoub(0) = 1.1; aDoub(2) = 2.2; aDoub(3) = 3.3; aDoub(7) = 4.4;
		Gcmat3Doub gaDoub(aDoub);
		if (gaDoub(0) != 1.1 || gaDoub(2) != 2.2)
			SLS_ERR("failed!");
		if (gaDoub(3) != 3.3 || gaDoub(7) != 4.4)
			SLS_ERR("failed!");
		gaDoub(0) *= 4.; gaDoub(2) -= 2.2; gaDoub(3) += 1.1; gaDoub(7) /= 2.2;
		aDoub = gaDoub;
		if (abs(aDoub(0) - 4.4) > 2e-15)
			SLS_ERR("failed!");
		if (abs(aDoub(2)) > 2e-15)
			SLS_ERR("failed!");
		if (abs(aDoub(3) - 4.4) > 2e-15)
			SLS_ERR("failed!");
		if (abs(aDoub(7) - 2.) > 2e-15)
			SLS_ERR("failed!");
	}

	// operator()
	{
		Cmat3Doub aDoub(2,2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(4) = 3.3; aDoub(5) = 4.4;
		Gcmat3Doub gaDoub(aDoub);
		if (gaDoub(0,0,0) != 1.1 || gaDoub(1,0,0) != 2.2)
			SLS_ERR("failed!");
		if (gaDoub(0,0,1) != 3.3 || gaDoub(1,0,1) != 4.4)
			SLS_ERR("failed!");
		gaDoub(0,0,0) *= 4.; gaDoub(1,0,0) -= 2.2;
		gaDoub(0,0,1) += 1.1; gaDoub(1,0,1) /= 2.2;
		aDoub = gaDoub;
		if (abs(aDoub(0) - 4.4) > 2e-15)
			SLS_ERR("failed!");
		if (abs(aDoub(1)) > 2e-15)
			SLS_ERR("failed!");
		if (abs(aDoub(4) - 4.4) > 2e-15)
			SLS_ERR("failed!");
		if (abs(aDoub(5) - 2.) > 2e-15)
			SLS_ERR("failed!");
	}

	// resize
	{
		// resize(n, m)
		Gcmat3Doub gaDoub;
		gaDoub.resize(2, 3, 4);
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3 || gaDoub.n3() != 4)
			SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr)
			SLS_ERR("failed!");
		Cmat3Doub aDoub(2, 3, 4); linspace(aDoub, 0., 23.);
		gaDoub = aDoub;
		Cmat3Doub aDoub1(2,3,4);
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
		gaDoub.resize(0, 0, 0);
		if (gaDoub.size() != 0)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 0|| gaDoub.n2() != 0 || gaDoub.n3() != 0)
			SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr)
			SLS_ERR("failed!");
		gaDoub.resize(0, 100, 101);
		if (gaDoub.size() != 0)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 0 || gaDoub.n2() != 100 || gaDoub.n3() != 101)
			SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr)
			SLS_ERR("failed!");
		// resize(Gmatrix<>)
		gaDoub.resize(2, 3, 4);
		Gcmat3Comp gaComp;
		gaComp.resize(gaDoub);
		if (gaComp.size() != 24)
			SLS_ERR("failed!");
		if (gaComp.n1() != 2 || gaComp.n2() != 3 || gaComp.n3() != 4)
			SLS_ERR("failed!");
		// resize(Matrix<>)
		aDoub.resize(4,5,6);
		gaComp.resize(aDoub);
		if (gaComp.size() != 120)
			SLS_ERR("failed!");
		if (gaComp.n1() != 4 || gaComp.n2() != 5 || gaComp.n3() != 6)
			SLS_ERR("failed!");
	}
}
