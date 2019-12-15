// Declarations of classes and functions that
// perform a post-hoc registration of the shells
// for the eddy project/.
//
// post_registration.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef PostEddyCF_h
#define PostEddyCF_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"

namespace EDDY {


 
class PostEddyCFImpl;
class PostEddyCF : public MISCMATHS::NonlinCF
{
public:
  PostEddyCF(const NEWIMAGE::volume<float>&  ref,
	     const NEWIMAGE::volume<float>&  ima,
	     const NEWIMAGE::volume<float>&  mask,
	     unsigned int                    nbins);
  PostEddyCF(const NEWIMAGE::volume<float>&  ref,
	     const NEWIMAGE::volume<float>&  ima,
	     const NEWIMAGE::volume<float>&  mask,
	     unsigned int                    nbins,
	     unsigned int                    pe_dir);
  ~PostEddyCF();
  NEWIMAGE::volume<float> GetTransformedIma(const NEWMAT::ColumnVector& p) const;
  double cf(const NEWMAT::ColumnVector& p) const;
  ReturnMatrix grad(const NEWMAT::ColumnVector& p) const;
private:
  int                        _pe_dir;  
  MutualInfoHelper           _fwd_mih;
  MutualInfoHelper           _bwd_mih;
  PostEddyCFImpl             *_pimpl;
};

} 

#endif 

