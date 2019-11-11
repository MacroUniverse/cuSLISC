#pragma once
#include "../SLISC/time.h"
#include "../SLISC/matrix.h"
#include "blocks_threads.h"
#include "complex.h"

namespace slisc {

// Scalar, vector and matrix types
template <typename T> class KerT;
template <typename T> class Gref;
template <typename T> class Gptr;
template <typename T> class Gscalar;
template <typename T> class Gbase;
template <typename T> class Gvector;
template <typename T> class Gmatrix;
template <typename T> class Gmat3d;

typedef const Gscalar<Int> Gint_I;
typedef Gscalar<Int> Gint, Gint_O, Gint_IO;
typedef const Gscalar<Uint> Guint_I;
typedef Gscalar<Uint> Guint, Guint_O, Guint_IO;

typedef const Gscalar<Long> Glong_I;
typedef Gscalar<Long> Glong, Glong_O, Glong_IO;

typedef const Gscalar<Llong> Gllong_I;
typedef Gscalar<Llong> Gllong, Gllong_O, Gllong_IO;
typedef const Gscalar<Ullong> Gullong_I;
typedef Gscalar<Ullong> Gullong, Gullong_O, Gullong_IO;

typedef const Gscalar<Char> Gchar_I;
typedef Gscalar<Char> Gchar, Gchar_O, Gchar_IO;
typedef const Gscalar<Uchar> Guchar_I;
typedef Gscalar<Uchar> Guchar, Guchar_O, Guchar_IO;

typedef const Gscalar<Doub> Gdoub_I;
typedef Gscalar<Doub> Gdoub, Gdoub_O, Gdoub_IO;
typedef const Gscalar<Ldoub> Gldoub_I;
typedef Gscalar<Ldoub> Gldoub, Gldoub_O, Gldoub_IO;

typedef const Gscalar<Comp> Gcomp_I;
typedef Gscalar<Comp> Gcomp, Gcomp_O, Gcomp_IO;

typedef const Gscalar<Bool> Gbool_I;
typedef Gscalar<Bool> Gbool, Gbool_O, Gbool_IO;

typedef const Gvector<Int> GvecInt_I;
typedef Gvector<Int> GvecInt, GvecInt_O, GvecInt_IO;

typedef const Gvector<Uint> GvecUint_I;
typedef Gvector<Uint> GvecUint, GvecUint_O, GvecUint_IO;

typedef const Gvector<Long> GvecLong_I;
typedef Gvector<Long> GvecLong, GvecLong_O, GvecLong_IO;

typedef const Gvector<Llong> GvecLlong_I;
typedef Gvector<Llong> GvecLlong, GvecLlong_O, GvecLlong_IO;

typedef const Gvector<Ullong> GvecUllong_I;
typedef Gvector<Ullong> GvecUllong, GvecUllong_O, GvecUllong_IO;

typedef const Gvector<Char> GvecChar_I;
typedef Gvector<Char> GvecChar, GvecChar_O, GvecChar_IO;

typedef const Gvector<Char*> GvecCharp_I;
typedef Gvector<Char*> GvecCharp, GvecCharp_O, GvecCharp_IO;

typedef const Gvector<Uchar> GvecUchar_I;
typedef Gvector<Uchar> GvecUchar, GvecUchar_O, GvecUchar_IO;

typedef const Gvector<Doub> GvecDoub_I;
typedef Gvector<Doub> GvecDoub, GvecDoub_O, GvecDoub_IO;

typedef const Gvector<Doub*> GvecDoubp_I;
typedef Gvector<Doub*> GvecDoubp, GvecDoubp_O, GvecDoubp_IO;

typedef const Gvector<Comp> GvecComp_I;
typedef Gvector<Comp> GvecComp, GvecComp_O, GvecComp_IO;

typedef const Gvector<Bool> GvecBool_I;
typedef Gvector<Bool> GvecBool, GvecBool_O, GvecBool_IO;

typedef const Gmatrix<Int> GmatInt_I;
typedef Gmatrix<Int> GmatInt, GmatInt_O, GmatInt_IO;

typedef const Gmatrix<Uint> GmatUint_I;
typedef Gmatrix<Uint> GmatUint, GmatUint_O, GmatUint_IO;

typedef const Gmatrix<Llong> GmatLlong_I;
typedef Gmatrix<Llong> GmatLlong, GmatLlong_O, GmatLlong_IO;

typedef const Gmatrix<Ullong> GmatUllong_I;
typedef Gmatrix<Ullong> GmatUllong, GmatUllong_O, GmatUllong_IO;

typedef const Gmatrix<Char> GmatChar_I;
typedef Gmatrix<Char> GmatChar, GmatChar_O, GmatChar_IO;

typedef const Gmatrix<Uchar> GmatUchar_I;
typedef Gmatrix<Uchar> GmatUchar, GmatUchar_O, GmatUchar_IO;

typedef const Gmatrix<Doub> GmatDoub_I;
typedef Gmatrix<Doub> GmatDoub, GmatDoub_O, GmatDoub_IO;

typedef const Gmatrix<Comp> GmatComp_I;
typedef Gmatrix<Comp> GmatComp, GmatComp_O, GmatComp_IO;

typedef const Gmatrix<Bool> GmatBool_I;
typedef Gmatrix<Bool> GmatBool, GmatBool_O, GmatBool_IO;

typedef const Gmat3d<Doub> Gmat3Doub_I;
typedef Gmat3d<Doub> Gmat3Doub, Gmat3Doub_O, Gmat3Doub_IO;

typedef const Gmat3d<Comp> Gmat3Comp_I;
typedef Gmat3d<Comp> Gmat3Comp, Gmat3Comp_O, Gmat3Comp_IO;


} // namespace slisc
