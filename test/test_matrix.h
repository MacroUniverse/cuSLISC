#pragma once
#include "../cuSLISC/cuslisc.h"

void test_matrix()
{
	using namespace slisc;
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
		aDoub1 = gaDoub;
		if (aDoub1 != aDoub) error("failed!");
	}

	// const initialize
	{
		GmatDoub gaDoub(2, 3, 1.23);
		MatDoub aDoub;
		aDoub = gaDoub;
		if (aDoub.size() != 6) error("failed!");
		if (aDoub.nrows() != 2 || aDoub.ncols() != 3) error("failed!");
		if (aDoub != 1.23) error("failed!");
	}

	// initialize from cpu vector/matrix
	{
		MatDoub aDoub(2,3), aDoub1; linspace(aDoub, 0., 5.);
		GmatDoub gaDoub(aDoub);
		aDoub1 = gaDoub;
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
		aDoub = gaDoub;
		if (aDoub != 3.14) error("failed!");

		// copy assignment
		linspace(aDoub, 6., 1., 2, 3);
		gaDoub = aDoub;
		GmatDoub gaDoub1(2, 3);
		gaDoub1 = gaDoub;
		MatDoub aDoub1;
		aDoub1 = gaDoub1;
		if (aDoub1 != aDoub) error("failed!");
	}

	// .end()
	{
		MatDoub aDoub(3, 3); linspace(aDoub, 1.1, 9.9);
		GmatDoub gaDoub(aDoub);
		if (gaDoub.end() != 9.9) error("failed!");
		gaDoub.end() = 10.10;
		aDoub = gaDoub;
		if (aDoub(0) != 1.1 || aDoub(1) != 2.2 || aDoub.end() != 10.10) error("failed!");
	}

	// operator()
	{
		MatDoub aDoub(2,2); aDoub(0) = 1.1; aDoub(1) = 2.2; aDoub(2) = 3.3; aDoub(3) = 4.4;
		GmatDoub gaDoub(aDoub);
		if (gaDoub(0) != 1.1 || gaDoub(1) != 2.2) error("failed!");
		if (gaDoub(2) != 3.3 || gaDoub(3) != 4.4) error("failed!");
		gaDoub(0) *= 4.; gaDoub(1) -= 2.2; gaDoub(2) += 1.1; gaDoub(3) /= 2.2;
		aDoub = gaDoub;
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
		aDoub = gaDoub;
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
		aDoub1 = gaDoub;
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
