#pragma once
#include "../cuSLISC/cuslisc.h"
#include "../SLISC/arithmetic.h"
#include "../SLISC/disp.h"

void test_matrix()
{
	using namespace slisc;
	// default initialize
	{
		GmatDoub gaDoub;
		if (gaDoub.size() != 0) SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr) SLS_ERR("failed!");
	}

	// size initialize
	{
		GmatDoub gaDoub(2,3);
		if (gaDoub.size() != 6) SLS_ERR("failed!");
		if (gaDoub.n1() != 2) SLS_ERR("failed!");
		if (gaDoub.n2() != 3) SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr) SLS_ERR("failed!");
	}

	// memory copy
	{
		MatDoub aDoub(2,2), aDoub1(2, 2);
		linspace(aDoub, 0., 3.);
		GmatDoub gaDoub(2,2);
		gaDoub = aDoub;
		if (gaDoub.size() != 4) SLS_ERR("failed!");
		if (gaDoub.n1() != 2) SLS_ERR("failed!");
		if (gaDoub.n2() != 2) SLS_ERR("failed!");
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub) SLS_ERR("failed!");
	}

	// const initialize
	{
		GmatDoub gaDoub(2, 3, 1.23);
		MatDoub aDoub(2, 3);
		aDoub = gaDoub;
		if (aDoub.size() != 6) SLS_ERR("failed!");
		if (aDoub.n1() != 2 || aDoub.n2() != 3) SLS_ERR("failed!");
		if (aDoub != 1.23) SLS_ERR("failed!");
	}

	// initialize from cpu vector/matrix
	{
		MatDoub aDoub(2,3), aDoub1(2,3); linspace(aDoub, 0., 5.);
		GmatDoub gaDoub(aDoub);
		aDoub1 = gaDoub;
		if (gaDoub.size() != 6) SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3) SLS_ERR("failed!");
		if (aDoub1 != aDoub) SLS_ERR("failed!");
	}

	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		GmatDoub gaDoub(2, 3);
		gaDoub = 3.14;
		MatDoub aDoub(2, 3);
		aDoub = gaDoub;
		if (aDoub != 3.14) SLS_ERR("failed!");

		// copy assignment
		linspace(aDoub, 6., 1.);
		gaDoub = aDoub;
		GmatDoub gaDoub1(2, 3);
		gaDoub1 = gaDoub;
		MatDoub aDoub1(2, 3);
		aDoub1 = gaDoub1;
		if (aDoub1 != aDoub) SLS_ERR("failed!");
	}

	// .end()
	{
		MatDoub aDoub(3, 3); linspace(aDoub, 1.1, 9.9);
		GmatDoub gaDoub(aDoub);
		if (gaDoub.end() != 9.9) SLS_ERR("failed!");
		gaDoub.end() = 10.10;
		aDoub = gaDoub;
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 10.10) SLS_ERR("failed!");
	}

	// operator()
	{
		MatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GmatDoub gaDoub(aDoub);
		if (gaDoub(0) != 1.1 || gaDoub(1) != 2.2) SLS_ERR("failed!");
		if (gaDoub(2) != 3.3 || gaDoub(3) != 4.4) SLS_ERR("failed!");
		gaDoub(0) *= 4.; gaDoub(1) -= 2.2; gaDoub(2) += 1.1; gaDoub(3) /= 2.2;
		aDoub = gaDoub;
		if (abs(aDoub(0) - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(1)) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(2) - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(3) - 2.) > 2e-15) SLS_ERR("failed!");
	}

	// operator()
	{
		MatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GmatDoub gaDoub(aDoub);
		if (gaDoub(0,0) != 1.1 || gaDoub(0,1) != 2.2) SLS_ERR("failed!");
		if (gaDoub(1,0) != 3.3 || gaDoub(1,1) != 4.4) SLS_ERR("failed!");
		gaDoub(0,0) *= 4.; gaDoub(0,1) -= 2.2; gaDoub(1,0) += 1.1; gaDoub(1,1) /= 2.2;
		aDoub = gaDoub;
		if (abs(aDoub(0) - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(1)) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(2) - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(3) - 2.) > 2e-15) SLS_ERR("failed!");
	}

	// resize
	{
		// resize(n, m)
		GmatDoub gaDoub;
		gaDoub.resize(2, 3);
		if (gaDoub.size() != 6) SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3) SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr) SLS_ERR("failed!");
		MatDoub aDoub(2, 3); linspace(aDoub, 0., 5.);
		gaDoub = aDoub;
		MatDoub aDoub1(2, 3);
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub) SLS_ERR("failed!");
		gaDoub.resize(0, 0);
		if (gaDoub.size() != 0)  SLS_ERR("failed!");
		if (gaDoub.n1() != 0|| gaDoub.n2() != 0) SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr) SLS_ERR("failed!");
		gaDoub.resize(0, 100);
		if (gaDoub.size() != 0)  SLS_ERR("failed!");
		if (gaDoub.n1() != 0 || gaDoub.n2() != 100) SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr) SLS_ERR("failed!");
		// resize(Gmatrix<>)
		gaDoub.resize(2, 3);
		GmatComp gaComp;
		gaComp.resize(gaDoub);
		if (gaComp.size() != 6)  SLS_ERR("failed!");
		if (gaComp.n1() != 2 || gaComp.n2() != 3) SLS_ERR("failed!");
		// resize(Matrix<>)
		aDoub.resize(4,5);
		gaComp.resize(aDoub);
		if (gaComp.size() != 20)  SLS_ERR("failed!");
		if (gaComp.n1() != 4 || gaComp.n2() != 5) SLS_ERR("failed!");
	}
}
