#pragma once
#include "../cuSLISC/cuslisc.h"
#include "../SLISC/arithmetic.h"
#include "../SLISC/disp.h"

void test_mat3d()
{
	using namespace slisc;
	// default initialize
	{
		Gmat3Doub gaDoub;
		if (gaDoub.size() != 0)
			SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr)
			SLS_ERR("failed!");
	}

	// size initialize
	{
		Gmat3Doub gaDoub(2,3,4);
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2)
			SLS_ERR("failed!");
		if (gaDoub.n2() != 3)
			SLS_ERR("failed!");
		if (gaDoub.dim3() != 4)
			SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr)
			SLS_ERR("failed!");
	}

	// memory copy
	{
		Mat3Doub aDoub(2,3,4), aDoub1;
		linspace(aDoub, 0., 23.);
		Gmat3Doub gaDoub(2,3,4);
		gaDoub = aDoub;
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2)
			SLS_ERR("failed!");
		if (gaDoub.n2() != 3)
			SLS_ERR("failed!");
		if (gaDoub.dim3() != 4)
			SLS_ERR("failed!");
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
	}

	// const initialize
	{
		Gmat3Doub gaDoub(2, 3, 4, 1.23);
		Mat3Doub aDoub;
		aDoub = gaDoub;
		if (aDoub.size() != 24)
			SLS_ERR("failed!");
		if (aDoub.n1() != 2 || aDoub.n2() != 3 || aDoub.dim3() != 4)
			SLS_ERR("failed!");
		if (aDoub != 1.23)
			SLS_ERR("failed!");
	}

	// initialize from cpu vector/matrix
	{
		Mat3Doub aDoub(2,3,4), aDoub1; linspace(aDoub, 0., 23.);
		Gmat3Doub gaDoub(aDoub);
		aDoub1 = gaDoub;
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3 || gaDoub.dim3() != 4)
			SLS_ERR("failed!");
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
	}

	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		Gmat3Doub gaDoub(2, 3, 4);
		gaDoub = 3.14;
		Mat3Doub aDoub;
		aDoub = gaDoub;
		if (aDoub != 3.14)
			SLS_ERR("failed!");

		// copy assignment
		aDoub.resize(2,3,4);
		linspace(aDoub, 24., 1.);
		gaDoub = aDoub;
		Gmat3Doub gaDoub1(2, 3, 4);
		gaDoub1 = gaDoub;
		Mat3Doub aDoub1;
		aDoub1 = gaDoub1;
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
	}

	// .end()
	{
		Mat3Doub aDoub(2,2,2); linspace(aDoub, 1.1, 8.8);
		Gmat3Doub gaDoub(aDoub);
		if (gaDoub.end() != 8.8)
			SLS_ERR("failed!");
		gaDoub.end() = 9.9;
		aDoub = gaDoub;
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 9.9)
			SLS_ERR("failed!");
	}

	// operator()
	{
		Mat3Doub aDoub(2,2,2); aDoub(0) = 1.1; aDoub(2) = 2.2; aDoub(3) = 3.3; aDoub(7) = 4.4;
		Gmat3Doub gaDoub(aDoub);
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

	// operator[]
	{
		Mat3Doub aDoub(2,2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(4) = 3.3; aDoub(5) = 4.4;
		Gmat3Doub gaDoub(aDoub);
		if (gaDoub[0][0][0] != 1.1 || gaDoub[0][0][1] != 2.2)
			SLS_ERR("failed!");
		if (gaDoub[1][0][0] != 3.3 || gaDoub[1][0][1] != 4.4)
			SLS_ERR("failed!");
		gaDoub[0][0][0] *= 4.; gaDoub[0][0][1] -= 2.2;
		gaDoub[1][0][0] += 1.1; gaDoub[1][0][1] /= 2.2;
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
		Gmat3Doub gaDoub;
		gaDoub.resize(2, 3, 4);
		if (gaDoub.size() != 24)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3 || gaDoub.dim3() != 4)
			SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr)
			SLS_ERR("failed!");
		Mat3Doub aDoub(2, 3, 4); linspace(aDoub, 0., 23.);
		gaDoub = aDoub;
		Mat3Doub aDoub1;
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub)
			SLS_ERR("failed!");
		gaDoub.resize(0, 0, 0);
		if (gaDoub.size() != 0)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 0|| gaDoub.n2() != 0 || gaDoub.dim3() != 0)
			SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr)
			SLS_ERR("failed!");
		gaDoub.resize(0, 100, 101);
		if (gaDoub.size() != 0)
			SLS_ERR("failed!");
		if (gaDoub.n1() != 0 || gaDoub.n2() != 100 || gaDoub.dim3() != 101)
			SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr)
			SLS_ERR("failed!");
		// resize(Gmatrix<>)
		gaDoub.resize(2, 3, 4);
		Gmat3Comp gaComp;
		gaComp.resize(gaDoub);
		if (gaComp.size() != 24)
			SLS_ERR("failed!");
		if (gaComp.n1() != 2 || gaComp.n2() != 3 || gaComp.dim3() != 4)
			SLS_ERR("failed!");
		// resize(Matrix<>)
		aDoub.resize(4,5,6);
		gaComp.resize(aDoub);
		if (gaComp.size() != 120)
			SLS_ERR("failed!");
		if (gaComp.n1() != 4 || gaComp.n2() != 5 || gaComp.dim3() != 6)
			SLS_ERR("failed!");
	}
}
