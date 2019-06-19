#pragma once
#include "../cuSLISC/cuslisc.h"
#include "../SLISC/arithmatic.h"
#include "../SLISC/disp.h"

void test_vector()
{
	using namespace slisc;
	using std::cout; using std::endl;

	// default initialize
	{
		GvecDoub gvDoub;
		if (gvDoub.size() != 0) SLS_ERR("failed!");
		if (gvDoub.ptr() != nullptr) SLS_ERR("failed!");
	}

	// size initialize
	{
		GvecDoub gvDoub(3);
		if (gvDoub.size() != 3) SLS_ERR("failed!");
		if (gvDoub.ptr() == nullptr) SLS_ERR("failed!");
	}

	// memory copy
	{
		VecDoub vDoub, vDoub1;
		linspace(vDoub, 0., 3., 4);
		GvecDoub gvDoub(4);
		gvDoub = vDoub;
		if (gvDoub.size() != 4) SLS_ERR("failed!");
		vDoub1 = gvDoub;
		if (vDoub1 != vDoub) SLS_ERR("failed!");

		VecComp vComp, vComp1;
		linspace(vComp, 0., Comp(3.,3.), 4);
		GvecComp gvComp(4);
		gvComp = vComp;
		if (gvComp.size() != 4) SLS_ERR("failed!");
		vComp1 = gvComp;
		if (vComp1 != vComp) SLS_ERR("failed!");
	}

	// const initialize
	{
		GvecDoub gvDoub(3, 1.23);
		VecDoub vDoub;
		vDoub = gvDoub;
		if (vDoub.size() != 3) SLS_ERR("failed!");
		if (vDoub != 1.23) SLS_ERR("failed!")
	}

	// initialize from cpu vector/matrix
	{
		VecDoub vDoub, vDoub1; linspace(vDoub, 0., 3., 4);
		GvecDoub gvDoub(vDoub);
		vDoub1 = gvDoub;
		if (gvDoub.size() != 4) SLS_ERR("failed!");
		if (vDoub1 != vDoub) SLS_ERR("failed!");
	}
	
	// assignment operator
	{
		// = NR*<>
			//already tested
		
		// = scalar
		GvecDoub gvDoub(10);
		gvDoub = 3.14;
		VecDoub vDoub;
		vDoub = gvDoub;
		if (vDoub != 3.14) SLS_ERR("failed!");

		// copy assignment
		linspace(vDoub, 10., 1., 10);
		gvDoub = vDoub;
		GvecDoub gvDoub1(10);
		gvDoub1 = gvDoub;
		VecDoub vDoub1;
		vDoub1 = gvDoub1;
		if (vDoub1 != vDoub) SLS_ERR("failed!");
	}

	// test class Gref
	{
		GvecDoub gvDoub(3); gvDoub = 1.1;
		Gref<Doub> ref(gvDoub.ptr());
		if (ref.ptr() != gvDoub.ptr()) SLS_ERR("failed!");
		ref = 5.6;
		VecDoub vDoub; vDoub = gvDoub;
		if (vDoub[0] != 5.6 || vDoub[1] != 1.1 || vDoub[2] != 1.1) SLS_ERR("failed!");
		if (ref != 5.6) SLS_ERR("failed!");
		ref += 1.1;
		if (abs(ref - 6.7) > 2e-15) SLS_ERR("failed!");
		ref -= 1.7;
		if (abs(ref - 5.) > 2e-15) SLS_ERR("failed!");
		ref *= 2.;
		if (abs(ref - 10.) > 2e-15) SLS_ERR("failed!");
		ref /= 5.;
		if (abs(ref - 2.) > 2e-15) SLS_ERR("failed!");

		GvecComp gvComp(3); gvComp = Comp(1.1,1.1);
		Gref<Comp> ref2(gvComp.ptr());
		if (ref2.ptr() != gvComp.ptr()) SLS_ERR("failed!");
		ref2 = Comp(5.6,5.6);
		VecComp vComp; vComp = gvComp;
		if (vComp[0] != Comp(5.6,5.6) || vComp[1] != Comp(1.1,1.1) || vComp[2] != Comp(1.1,1.1)) SLS_ERR("failed!");
		if (ref2 != Comp(5.6,5.6)) SLS_ERR("failed!");
		ref2 += Comp(1.1,1.1);
		if (abs((Comp)ref2 - Comp(6.7,6.7)) > 2e-15) SLS_ERR("failed!");
		ref2 -= Comp(1.7,1.7);
		if (abs((Comp)ref2 - Comp(5.,5.)) > 2e-15) SLS_ERR("failed!");
		ref2 *= 2.;
		if (abs((Comp)ref2 - Comp(10.,10.)) > 4e-15) SLS_ERR("failed!");
		ref2 /= 5.;
		if (abs((Comp)ref2 - Comp(2.,2.)) > 2e-15) SLS_ERR("failed!");
	}

	// test class Gptr<Doub>
	{
		Gptr<Doub> ptr0; // default constructor
		if (ptr0.ptr()) SLS_ERR("failed!");
		GvecDoub gvDoub(3); gvDoub = 1.1;
		Gptr<Doub> ptr(gvDoub[0].ptr()); // pointer constructor
		if (ptr.ptr() != gvDoub.ptr()) SLS_ERR("failed!");
		if (*ptr != 1.1) SLS_ERR("failed!"); // dereference
		// operator[]
		if (ptr[0] != 1.1 || ptr[1] != 1.1 || ptr[2] != 1.1) SLS_ERR("failed!"); 
		ptr[1] = 2.2; ptr[2] = 3.3;
		if (ptr[0] != 1.1 || ptr[1] != 2.2 || ptr[2] != 3.3) SLS_ERR("failed!");
		ptr0 = ptr; // copy assignment
		if (ptr0.ptr() != ptr.ptr()) SLS_ERR("failed!");
		ptr0 = ptr[1].ptr(); // T* assignment
		if (*ptr0 != 2.2) SLS_ERR("failed!");
		// pointer arithmatics
		ptr0 = ptr0 + 1;
		if (*ptr0 != 3.3) SLS_ERR("failed!");
		ptr0 = ptr0 - 2;
		if (*ptr0 != 1.1) SLS_ERR("failed!");
		ptr0 += 2;
		if (*ptr0 != 3.3) SLS_ERR("failed!");
		ptr0 -= 1;
		if (*ptr0 != 2.2) SLS_ERR("failed!");
	}

	// test class Gptr<Comp>
	{
		Gptr<Comp> ptr0; // default constructor
		if (ptr0.ptr()) SLS_ERR("failed!");
		GvecComp gvComp(3); gvComp = Comp(1.1,1.1);
		Gptr<Comp> ptr(gvComp[0].ptr()); // pointer constructor
		if (ptr.ptr() != gvComp.ptr()) SLS_ERR("failed!");
		if (*ptr != Comp(1.1,1.1)) SLS_ERR("failed!"); // dereference
		// operator[]
		if (ptr[0] != Comp(1.1,1.1) || ptr[1] != Comp(1.1,1.1) || ptr[2] != Comp(1.1,1.1)) SLS_ERR("failed!"); 
		ptr[1] = Comp(2.2,2.2); ptr[2] = Comp(3.3,3.3);
		if (ptr[0] != Comp(1.1,1.1) || ptr[1] != Comp(2.2,2.2) || ptr[2] != Comp(3.3,3.3)) SLS_ERR("failed!");
		ptr0 = ptr; // copy assignment
		if (ptr0.ptr() != ptr.ptr()) SLS_ERR("failed!");
		ptr0 = ptr[1].ptr(); // T* assignment
		if (*ptr0 != Comp(2.2,2.2)) SLS_ERR("failed!");
		// pointer arithmatics
		ptr0 = ptr0 + 1;
		if (*ptr0 != Comp(3.3,3.3)) SLS_ERR("failed!");
		ptr0 = ptr0 - 2;
		if (*ptr0 != Comp(1.1,1.1)) SLS_ERR("failed!");
		ptr0 += 2;
		if (*ptr0 != Comp(3.3,3.3)) SLS_ERR("failed!");
		ptr0 -= 1;
		if (*ptr0 != Comp(2.2,2.2)) SLS_ERR("failed!");
	}

	// test scalar
	{
		Gdoub gs;
		if (!gs.ptr()) SLS_ERR("failed!");
		gs = 3.1415;
		if (gs != 3.1415) SLS_ERR("failed!");
		gs -= 3.1;
		if (abs(gs-0.0415) > 1e-16) SLS_ERR("failed!");
		gs += 3.1;
		if (abs(gs-3.1415) > 1e-16) SLS_ERR("failed!");
		gs *= 2;
		if (abs(gs-6.283) > 1e-16) SLS_ERR("failed!");
		gs /= 2;
		if (abs(gs-3.1415) > 1e-16) SLS_ERR("failed!");
		Gcomp gs1(Comp(1.1, 2.2));
		if ((Comp)gs1 != Comp(1.1,2.2)) SLS_ERR("failed!");
		if (abs((Comp)gs1 + (Comp)gs - Comp(4.2415, 2.2)) > 1e-16)  SLS_ERR("failed!");
	}

	// .end()
	{
		VecDoub vDoub(3); linspace(vDoub, 1.1, 3.3);
		GvecDoub gvDoub(vDoub);
		if (gvDoub.end() != 3.3) SLS_ERR("failed!");
		gvDoub.end() = 4.4;
		vDoub = gvDoub;
		if (vDoub[0] != 1.1 || vDoub[1] != 2.2 || vDoub.end() != 4.4) SLS_ERR("failed!");
	}
	
	// operator()
	{
		VecDoub vDoub(4); vDoub[0] = 1.1; vDoub[1] = 2.2; vDoub[2] = 3.3; vDoub[3] = 4.4;
		GvecDoub gvDoub(vDoub);
		if (gvDoub(0) != 1.1 || gvDoub(1) != 2.2) SLS_ERR("failed!");
		if (gvDoub(2) != 3.3 || gvDoub(3) != 4.4) SLS_ERR("failed!");
		gvDoub(0) *= 4.; gvDoub(1) -= 2.2; gvDoub(2) += 1.1; gvDoub(3) /= 2.2;
		vDoub = gvDoub;
		if (abs(vDoub[0] - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(vDoub[1]) > 2e-15) SLS_ERR("failed!");
		if (abs(vDoub[2] - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(vDoub[3] - 2.) > 2e-15) SLS_ERR("failed!");
	}

	// operator[]
	{
		VecDoub vDoub(4); vDoub[0] = 1.1; vDoub[1] = 2.2; vDoub[2] = 3.3; vDoub[3] = 4.4;
		GvecDoub gvDoub(vDoub);
		if (gvDoub[0] != 1.1 || gvDoub[1] != 2.2) SLS_ERR("failed!");
		if (gvDoub[2] != 3.3 || gvDoub[3] != 4.4) SLS_ERR("failed!");
		gvDoub[0] *= 4.; gvDoub[1] -= 2.2; gvDoub[2] += 1.1; gvDoub[3] /= 2.2;
		vDoub = gvDoub;
		if (abs(vDoub[0] - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(vDoub[1]) > 2e-15) SLS_ERR("failed!");
		if (abs(vDoub[2] - 4.4) > 2e-15) SLS_ERR("failed!");
		if (abs(vDoub[3] - 2.) > 2e-15) SLS_ERR("failed!");
	}

	// resize
	{
		VecDoub vDoub(4);
		GvecDoub gvDoub;
		if (gvDoub.size() != 0) SLS_ERR("failed!");
		gvDoub.resize(2);
		if (gvDoub.size() != 2) SLS_ERR("failed!");
		gvDoub[1] = 1.3;
		if (gvDoub(1) != 1.3) SLS_ERR("failed!");
		gvDoub.resize(vDoub);
		if (gvDoub.size() != 4) SLS_ERR("failed!");
		gvDoub[3] = 2.4;
		if (gvDoub(3) != 2.4) SLS_ERR("failed!");
		GvecDoub gvDoub1(10);
		gvDoub.resize(gvDoub1);
		if (gvDoub.size() != 10) SLS_ERR("failed!");
		gvDoub(9) = 5.5;
		if (gvDoub[9] != 5.5) SLS_ERR("failed!");
	}
}
