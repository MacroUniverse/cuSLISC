#pragma once
#include "../cuSLISC/cuslisc.h"

__device__ slisc::Doub glob_dev_Doub;
__device__ slisc::Cump glob_dev_Cump;
__constant__ slisc::Doub glob_const_Doub;
__constant__ slisc::Cump glob_const_Cump;

void test_global()
{
	using namespace slisc;
	// __device__ var
	setsym(glob_dev_Doub, 3.14);
	if (getsym(glob_dev_Doub) != 3.14) error("failed!");
	setsym(glob_dev_Cump, Comp(1.1,2.2));
	if (getsym(glob_dev_Cump) != Comp(1.1,2.2)) error("failed!");
	setsym(glob_dev_Cump, 3.14);
	if (getsym(glob_dev_Cump) != Comp(3.14,0.)) error("failed!");

	// __const__ var
	setsym(glob_const_Doub, 6.28);
	if (getsym(glob_const_Doub) != 6.28) error("failed!");
	setsym(glob_const_Cump, Comp(1.1,2.2));
	if (getsym(glob_const_Cump) != Comp(1.1,2.2)) error("failed!");
	setsym(glob_dev_Cump, 3.14);
	if (getsym(glob_dev_Cump) != Comp(3.14,0.)) error("failed!");
}
