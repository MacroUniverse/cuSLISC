#include "../cuSLISC/arithmatic.h"
#include "../SLISC/disp.h"

// test basic operations
void test_basic()
{
	using namespace slisc;
	// v += v; v -= v; v *= v; v /= v
	{
		GmatComp ga(10,10,Comp(1.,2.));
		GmatDoub ga1(10,10,1.);
		ga += ga1;
		MatComp a; a = ga;
		if (a != Comp(2.,2.)) error("failed!");
		ga -= ga1; a = ga;
		if (a != Comp(1.,2.)) error("failed!");
		ga = Comp(3.14, 3.33); ga1 = 2.;
		ga *= ga1; a = ga;
		if (a != Comp(6.28, 6.66)) error("failed!");
		ga /= ga1; a = ga;
		if (a != Comp(3.14, 3.33)) error("failed!");
	}

	// v +=s; v -= s; v *= s; v /= s;
	{
		GmatComp ga(10,10,Comp(10.,20.));
		ga += 10.; ga -= Comp(0.,10.);
		MatComp a; a = ga;
		if (a != Comp(20.,10.)) error("failed!");
		ga *= 1.5; a = ga;
		if (a != Comp(30.,15.)) error("failed!");
		ga /= Comp(0.,1.5); a = ga;
		if (a != Comp(10.,-20.)) error("failed!");
	}

	// plus(v, v1, s); plus(v, s, v1); plus(v, v1, v2);
	{
		GmatComp ga(10,10);
		GmatDoub ga1(10,10, 2.2);
		MatComp a;
		plus(ga, ga1, -0.2); a = ga;
		if (a != 2.) error("failed!");
		ga = 0.; plus(ga, -0.2, ga1); a = ga;
		if (a != 2.) error("failed!");
		ga1 = 2.;
		GmatComp ga2(10,10,Comp(1.,1.));
		ga = 0.; plus(ga, ga1, ga2); a = ga;
		MatComp a1(10,10,Comp(3.,1.));
		if (a != a1) error("failed!");
	}

	// minus(v);
	{
		GmatComp ga(10,10,Comp(3.14,-6.28));
		MatComp a;
		minus(ga); a = ga;
		if (a != Comp(-3.14, 6.28)) error("failed!");
	}

	// sum(v), norm2(v), norm(v)
	{
		GmatComp gv(10, 10, Comp(1.,2.));
		if (sum(gv) != Comp(100.,200.)) error("failed!");
		if (abs(norm2(gv)-500.)>2e-13) error("failed!");
		GmatDoub gv1(10, 10, 1.2);
		if (abs(norm2(gv1)-144.)>1e-15) error("failed!");
		gv /= norm(gv);
		if (abs(norm2(gv)-1.)>1e-15) error("failed!");
		gv1 /= norm(gv1);
		if (abs(norm2(gv1)-1.)>1e-15) error("failed!");
	}
}
