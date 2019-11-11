#pragma once
#include "../cuSLISC/cuslisc.h"
#include "../SLISC/arithmetic.h"
#include "../SLISC/disp.h"

void test_gcmat()
{
	using namespace slisc;
	// default initialize
	{
		GcmatDoub gaDoub;
		if (gaDoub.size() != 0) SLS_ERR("failed!");
		if (gaDoub.ptr() != nullptr) SLS_ERR("failed!");
	}

	// size initialize
	{
		GcmatDoub gaDoub(2,3);
		if (gaDoub.size() != 6) SLS_ERR("failed!");
		if (gaDoub.n1() != 2) SLS_ERR("failed!");
		if (gaDoub.n2() != 3) SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr) SLS_ERR("failed!");
	}

	// memory copy
	{
		CmatDoub aDoub(2,2), aDoub1(2, 2);
		linspace(aDoub, 0., 3.);
		GcmatDoub gaDoub(2,2);
		gaDoub = aDoub;
		if (gaDoub.size() != 4) SLS_ERR("failed!");
		if (gaDoub.n1() != 2) SLS_ERR("failed!");
		if (gaDoub.n2() != 2) SLS_ERR("failed!");
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub) SLS_ERR("failed!");
	}

	// const initialize
	{
		GcmatDoub gaDoub(2, 3, 1.23);
		CmatDoub aDoub(2, 3);
		aDoub = gaDoub;
		if (aDoub.size() != 6) SLS_ERR("failed!");
		if (aDoub.n1() != 2 || aDoub.n2() != 3) SLS_ERR("failed!");
		if (aDoub != 1.23) SLS_ERR("failed!");
	}

	// initialize from cpu vector/matrix
	{
		CmatDoub aDoub(2,3), aDoub1(2,3); linspace(aDoub, 0., 5.);
		GcmatDoub gaDoub(aDoub);
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
		GcmatDoub gaDoub(2, 3);
		gaDoub = 3.14;
		CmatDoub aDoub(2, 3);
		aDoub = gaDoub;
		if (aDoub != 3.14) SLS_ERR("failed!");

		// copy assignment
		linspace(aDoub, 6., 1.);
		gaDoub = aDoub;
		GcmatDoub gaDoub1(2, 3);
		gaDoub1 = gaDoub;
		CmatDoub aDoub1(2, 3);
		aDoub1 = gaDoub1;
		if (aDoub1 != aDoub) SLS_ERR("failed!");
	}

	// .end()
	{
		CmatDoub aDoub(3, 3); linspace(aDoub, 1.1, 9.9);
		GcmatDoub gaDoub(aDoub);
		if (gaDoub.end() != 9.9) SLS_ERR("failed!");
		gaDoub.end() = 10.10;
		aDoub = gaDoub;
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 10.10) SLS_ERR("failed!");
	}

	// operator()
	{
		CmatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GcmatDoub gaDoub(aDoub);
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
		CmatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GcmatDoub gaDoub(aDoub);
		if (gaDoub(0,0) != 1.1 || gaDoub(1,0) != 2.2) SLS_ERR("failed!");
		if (gaDoub(0,1) != 3.3 || gaDoub(1,1) != 4.4) SLS_ERR("failed!");
		gaDoub(0,0) *= 4.; gaDoub(1,0) -= 2.2; gaDoub(0,1) += 1.1; gaDoub(1,1) /= 2.2;
		aDoub = gaDoub;
		if (abs(aDoub(0) - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(1)) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(2) - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(aDoub(3) - 2.) > 2e-15) SLS_ERR("failed!");
	}

	// resize
	{
		// resize(n, m)
		GcmatDoub gaDoub;
		gaDoub.resize(2, 3);
		if (gaDoub.size() != 6) SLS_ERR("failed!");
		if (gaDoub.n1() != 2 || gaDoub.n2() != 3) SLS_ERR("failed!");
		if (gaDoub.ptr() == nullptr) SLS_ERR("failed!");
		CmatDoub aDoub(2, 3); linspace(aDoub, 0., 5.);
		gaDoub = aDoub;
		CmatDoub aDoub1(2, 3);
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
		// resize(Gcmatrix<>)
		gaDoub.resize(2, 3);
		GcmatComp gaComp;
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
