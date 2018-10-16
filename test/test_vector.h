#include "../cuSLISC/cuslisc.h"

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
		vDoub1 = gvDoub;
		if (vDoub1 != vDoub) error("failed!");

		VecComp vComp, vComp1;
		linspace(vComp, 0., Comp(3.,3.), 4);
		GvecComp gvComp(4);
		gvComp = vComp;
		if (gvComp.size() != 4) error("failed!");
		vComp1 = gvComp;
		if (vComp1 != vComp) error("failed!");
	}

	// const initialize
	{
		GvecDoub gvDoub(3, 1.23);
		VecDoub vDoub;
		vDoub = gvDoub;
		if (vDoub.size() != 3) error("failed!");
		if (vDoub != 1.23) error("failed!")
	}

	// initialize from cpu vector/matrix
	{
		VecDoub vDoub, vDoub1; linspace(vDoub, 0., 3., 4);
		GvecDoub gvDoub(vDoub);
		vDoub1 = gvDoub;
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
		vDoub = gvDoub;
		if (vDoub != 3.14) error("failed!");

		// copy assignment
		linspace(vDoub, 10., 1., 10);
		gvDoub = vDoub;
		GvecDoub gvDoub1(10);
		gvDoub1 = gvDoub;
		VecDoub vDoub1;
		vDoub1 = gvDoub1;
		if (vDoub1 != vDoub) error("failed!");
	}

	// test class CUref
	{
		GvecDoub gvDoub(3); gvDoub = 1.1;
		CUref<Doub> ref(gvDoub.ptr());
		if (ref.ptr() != gvDoub.ptr()) error("failed!");
		ref = 5.6;
		VecDoub vDoub; vDoub = gvDoub;
		if (vDoub[0] != 5.6 || vDoub[1] != 1.1 || vDoub[2] != 1.1) error("failed!");
		if (ref != 5.6) error("failed!");
		ref += 1.1;
		if (abs(ref - 6.7) > 2e-15) error("failed!");
		ref -= 1.7;
		if (abs(ref - 5.) > 2e-15) error("failed!");
		ref *= 2.;
		if (abs(ref - 10.) > 2e-15) error("failed!");
		ref /= 5.;
		if (abs(ref - 2.) > 2e-15) error("failed!");

		GvecComp gvComp(3); gvComp = Comp(1.1,1.1);
		CUref<Comp> ref2(gvComp.ptr());
		if (ref2.ptr() != gvComp.ptr()) error("failed!");
		ref2 = Comp(5.6,5.6);
		VecComp vComp; vComp = gvComp;
		if (vComp[0] != Comp(5.6,5.6) || vComp[1] != Comp(1.1,1.1) || vComp[2] != Comp(1.1,1.1)) error("failed!");
		if (ref2 != Comp(5.6,5.6)) error("failed!");
		ref2 += Comp(1.1,1.1);
		if (abs((Comp)ref2 - Comp(6.7,6.7)) > 2e-15) error("failed!");
		ref2 -= Comp(1.7,1.7);
		if (abs((Comp)ref2 - Comp(5.,5.)) > 2e-15) error("failed!");
		ref2 *= 2.;
		if (abs((Comp)ref2 - Comp(10.,10.)) > 4e-15) error("failed!");
		ref2 /= 5.;
		if (abs((Comp)ref2 - Comp(2.,2.)) > 2e-15) error("failed!");
	}

	// test class CUptr<Doub>
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

	// test class CUptr<Comp>
	{
		CUptr<Comp> ptr0; // default constructor
		if (ptr0.ptr()) error("failed!");
		GvecComp gvComp(3); gvComp = Comp(1.1,1.1);
		CUptr<Comp> ptr(gvComp[0].ptr()); // pointer constructor
		if (ptr.ptr() != gvComp.ptr()) error("failed!");
		if (*ptr != Comp(1.1,1.1)) error("failed!"); // dereference
		// operator[]
		if (ptr[0] != Comp(1.1,1.1) || ptr[1] != Comp(1.1,1.1) || ptr[2] != Comp(1.1,1.1)) error("failed!"); 
		ptr[1] = Comp(2.2,2.2); ptr[2] = Comp(3.3,3.3);
		if (ptr[0] != Comp(1.1,1.1) || ptr[1] != Comp(2.2,2.2) || ptr[2] != Comp(3.3,3.3)) error("failed!");
		ptr0 = ptr; // copy assignment
		if (ptr0.ptr() != ptr.ptr()) error("failed!");
		ptr0 = ptr[1].ptr(); // T* assignment
		if (*ptr0 != Comp(2.2,2.2)) error("failed!");
		// pointer arithmatics
		ptr0 = ptr0 + 1;
		if (*ptr0 != Comp(3.3,3.3)) error("failed!");
		ptr0 = ptr0 - 2;
		if (*ptr0 != Comp(1.1,1.1)) error("failed!");
		ptr0 += 2;
		if (*ptr0 != Comp(3.3,3.3)) error("failed!");
		ptr0 -= 1;
		if (*ptr0 != Comp(2.2,2.2)) error("failed!");
	}

	// test scalar
	{
		Gdoub gs;
		if (!gs.ptr()) error("failed!");
		gs = 3.1415;
		if (gs != 3.1415) error("failed!");
		gs -= 3.1;
		if (abs(gs-0.0415) > 1e-16) error("failed!");
		gs += 3.1;
		if (abs(gs-3.1415) > 1e-16) error("failed!");
		gs *= 2;
		if (abs(gs-6.283) > 1e-16) error("failed!");
		gs /= 2;
		if (abs(gs-3.1415) > 1e-16) error("failed!");
		Gcomp gs1(Comp(1.1, 2.2));
		if ((Comp)gs1 != Comp(1.1,2.2)) error("failed!");
		if (abs((Comp)gs1 + (Comp)gs - Comp(4.2415, 2.2)) > 1e-16)  error("failed!");
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
		vDoub = gvDoub;
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
		vDoub = gvDoub;
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
