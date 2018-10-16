// manually set max block number and thread number
// TODO : optimize these numbers
// <<<nbl(Nbl*, Nth*, N), Nth*>>> for kernel call

namespace cuslisc {

#ifdef CUSLISC_GTX1080
const Int Nbl_cumemset = 320, Nth_cumemset = 32;
const Int Nbl_plus_equals0 = 320, Nth_plus_equals0 = 32;
const Int Nbl_minus_equals0 = 320, Nth_minus_equals0 = 32;
const Int Nbl_times_equals0 = 320, Nth_times_equals0 = 32;
const Int Nbl_divide_equals0 = 320, Nth_divide_equals0 = 32;
const Int Nbl_plus_equals1 = 320, Nth_plus_equals1 = 32;
const Int Nbl_minus_equals1 = 320, Nth_minus_equals1 = 32;
const Int Nbl_times_equals1 = 320, Nth_times_equals1 = 32;
const Int Nbl_divide_equals1 = 320, Nth_divide_equals1 = 32;
const Int Nbl_plus0 = 320, Nth_plus0 = 32;
const Int Nbl_plus1 = 320, Nth_plus1 = 32;
const Int Nbl_minus0 = 320, Nth_minus0 = 32;
const Int Nbl_minus1 = 320, Nth_minus1 = 32;
const Int Nbl_minus2 = 320, Nth_minus2 = 32;
const Int Nbl_sum = 320, Nth_sum = 32;
const Int Nbl_norm2 = 320, Nth_norm2 = 32;
#endif
#ifdef CUSLISC_P100
const Int Nbl_cumemset = 320, Nth_cumemset = 32;
const Int Nbl_sum = 320, Nth_sum = 32;
#endif
#ifdef CUSLISC_V100
const Int Nbl_cumemset = 320, Nth_cumemset = 32;
const Int Nbl_sum = 320, Nth_sum = 32;
#endif

}
