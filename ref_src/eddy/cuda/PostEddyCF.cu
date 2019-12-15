// Definitions of classes and functions that
// perform a post-hoc registration of the shells
// for the eddy project/.
//
// PostEddyCF.cu
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#pragma diag_suppress = expr_has_no_effect  
#include "newmat.h"
#include "topup/topup_file_io.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "PostEddyCF.h"
#include "CudaVolume.h"
#include "EddyInternalGpuUtils.h"

namespace EDDY {

class PostEddyCFImpl
{
public:
  PostEddyCFImpl(const NEWIMAGE::volume<float>&  ref,
		 const NEWIMAGE::volume<float>&  ima,
		 const NEWIMAGE::volume<float>&  mask) EddyTry : _ref(ref), _ima(ima), _mask(mask) {} EddyCatch
  ~PostEddyCFImpl() EddyTry {} EddyCatch
  double cf(const NEWMAT::ColumnVector&    p,
	    const EDDY::MutualInfoHelper&  fwd_mih,
	    const EDDY::MutualInfoHelper&  bwd_mih,
	    int                            pe_dir) const;
  NEWIMAGE::volume<float> GetTransformedIma(const NEWMAT::ColumnVector& p,
					    int                         pe_dir) const;
private:
  EDDY::CudaVolume _ref;
  EDDY::CudaVolume _ima;
  EDDY::CudaVolume _mask;
};

PostEddyCF::PostEddyCF(const NEWIMAGE::volume<float>&  ref,
		       const NEWIMAGE::volume<float>&  ima,
	               const NEWIMAGE::volume<float>&  mask,
		       unsigned int                    nbins) EddyTry
: _fwd_mih(nbins,ref.robustmin(),ref.robustmax(),ima.robustmin(),ima.robustmax()),
  _bwd_mih(nbins,ima.robustmin(),ima.robustmax(),ref.robustmin(),ref.robustmax()) 
{ 
  _pimpl = new PostEddyCFImpl(ref,ima,mask); 
  _pe_dir = -1; 
} EddyCatch

PostEddyCF::PostEddyCF(const NEWIMAGE::volume<float>&  ref,
		       const NEWIMAGE::volume<float>&  ima,
	               const NEWIMAGE::volume<float>&  mask,
		       unsigned int                    nbins,
		       unsigned int                    pe_dir) EddyTry
: _fwd_mih(nbins,ref.robustmin(),ref.robustmax(),ima.robustmin(),ima.robustmax()),
  _bwd_mih(nbins,ima.robustmin(),ima.robustmax(),ref.robustmin(),ref.robustmax()) 
{ 
  _pimpl = new PostEddyCFImpl(ref,ima,mask);
  if (pe_dir > 1) throw EddyException("EDDY::PostEddyCF::PostEddyCF: pe_dir must be 0 or 1");
  else _pe_dir = static_cast<int>(pe_dir);
} EddyCatch

PostEddyCF::~PostEddyCF() EddyTry { delete _pimpl; } EddyCatch

double PostEddyCF::cf(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->cf(p,_fwd_mih,_bwd_mih,_pe_dir)); } EddyCatch

ReturnMatrix PostEddyCF::grad(const NEWMAT::ColumnVector& p) const EddyTry
{
  NEWMAT::ColumnVector tp = p;
  NEWMAT::ColumnVector gradv(p.Nrows());
  static const double dscale[] = {1e-2, 1e-2, 1e-2, 1e-5, 1e-5, 1e-5}; 
  double base = _pimpl->cf(tp,_fwd_mih,_bwd_mih,_pe_dir);
  for (int i=0; i<p.Nrows(); i++) {
    tp(i+1) += dscale[i];
    gradv(i+1) = (_pimpl->cf(tp,_fwd_mih,_bwd_mih,_pe_dir) - base) / dscale[i];
    tp(i+1) -= dscale[i];
  }
  gradv.Release();
  return(gradv);
} EddyCatch

NEWIMAGE::volume<float> PostEddyCF::GetTransformedIma(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->GetTransformedIma(p,_pe_dir)); } EddyCatch

  

double PostEddyCFImpl::cf(const NEWMAT::ColumnVector&    p,
			  const EDDY::MutualInfoHelper&  fwd_mih,
			  const EDDY::MutualInfoHelper&  bwd_mih,
			  int                            pe_dir) const EddyTry
{
  EDDY::CudaVolume rima = _ima; rima = 0.0;
  EDDY::CudaVolume mask1(_ima,false); mask1 = 1.0;
  EDDY::CudaVolume mask2 = _ima; mask2 = 0.0;
  EDDY::CudaVolume4D skrutt1;
  EDDY::CudaVolume skrutt2;
  NEWMAT::Matrix A;
  if (p.Nrows() == 1) {
    if(pe_dir<0 || pe_dir>1) EddyException("EDDY::PostEddyCFImpl::cf: invalid pe_dir value");
    NEWMAT::ColumnVector tmp(6); tmp = 0.0;
    tmp(pe_dir + 1) = p(1);
    A = TOPUP::MovePar2Matrix(tmp,_ima.GetVolume());
  }
  else if (p.Nrows() == 6) A = TOPUP::MovePar2Matrix(p,_ima.GetVolume());
  else throw EddyException("EDDY::PostEddyCFImpl::cf: size of p must be 1 or 6");
  
  EddyInternalGpuUtils::affine_transform(_ima,A,rima,skrutt1,mask1);
  
  EddyInternalGpuUtils::affine_transform(_mask,A,mask2,skrutt1,skrutt2);
  
  
  mask2 *= mask1;
  
  double rval = - fwd_mih.SoftMI(_ref.GetVolume(),rima.GetVolume(),mask2.GetVolume());
  

  
  rima = 0.0; mask1 = 1.0; mask2 = 0.0;
  EddyInternalGpuUtils::affine_transform(_ref,A.i(),rima,skrutt1,mask1);
  
  EddyInternalGpuUtils::affine_transform(_mask,A.i(),mask2,skrutt1,skrutt2);
  
  mask2 *= mask1;
  
  rval += - bwd_mih.SoftMI(_ima.GetVolume(),rima.GetVolume(),mask2.GetVolume()); 
  
  rval /= 2.0;

  
  

  return(rval);  
} EddyCatch


NEWIMAGE::volume<float> PostEddyCFImpl::GetTransformedIma(const NEWMAT::ColumnVector& p,
                                                          int                         pe_dir) const EddyTry
{
  EDDY::CudaVolume rima = _ima; rima = 0.0;
  EDDY::CudaVolume4D skrutt1;
  EDDY::CudaVolume skrutt2;
  NEWMAT::Matrix A;
  if (p.Nrows() == 1) {
    if(pe_dir<0 || pe_dir>1) EddyException("EDDY::PostEddyCFImpl::GetTransformedIma: invalid pe_dir value");
    NEWMAT::ColumnVector tmp(6); tmp = 0.0;
    tmp(pe_dir + 1) = p(1);
    A = TOPUP::MovePar2Matrix(tmp,_ima.GetVolume());
  }
  else if (p.Nrows() == 6) A = TOPUP::MovePar2Matrix(p,_ima.GetVolume());
  else throw EddyException("EDDY::PostEddyCFImpl::GetTransformedIma: size of p must be 1 or 6");
  EddyInternalGpuUtils::affine_transform(_ima,A,rima,skrutt1,skrutt2);
  return(rima.GetVolume());
} EddyCatch

} 

