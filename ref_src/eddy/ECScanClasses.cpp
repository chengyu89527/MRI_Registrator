
// Declarations of classes that implements a scan
// or a collection of scans within the EC project.
// 
// ECScanClasses.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
#include "newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "warpfns/warpfns.h"
#include "topup/topup_file_io.h"
#include "topup/displacement_vector.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "LSResampler.h"
#include "ECScanClasses.h"
#ifdef COMPILE_GPU
#include "cuda/EddyGpuUtils.h"
#endif

using namespace EDDY;


double ECScan::GetMovementStd(unsigned int mi, std::vector<unsigned int> icsl) const EddyTry
{
  if (mi>5) throw EddyException("ECScan::GetMovementStd: mi out of range");
  if (_mbg.MBFactor() > 1 && icsl.size()) throw EddyException("ECScan::GetMovementStd: non-zero icsl can only be used for single-band data");
  if (!icsl.size()) {
    icsl.resize(_mbg.NGroups());
    for (unsigned int i=0; i<_mbg.NGroups(); i++) icsl[i] = i;
  }
  double stdev = 0.0;
  if (this->IsSliceToVol()) {
    NEWMAT::ColumnVector tmp(icsl.size());
    for (unsigned int i=0; i<icsl.size(); i++) {
      if (icsl[i] >= _mbg.NGroups()) throw EddyException("ECScan::GetMovementStd: icsl out of range");
      tmp(i+1) = _mp.GetGroupWiseParams(icsl[i],_mbg.NGroups())(mi+1);
    }
    tmp -= tmp.Sum()/static_cast<double>(tmp.Nrows());
    stdev = std::sqrt(tmp.SumSquare() / static_cast<double>(tmp.Nrows()-1));
  }
  return(stdev);
} EddyCatch

void ECScan::SetPolation(const PolationPara& pp) EddyTry
{
  if (_ima.getinterpolationmethod() != pp.GetInterp()) _ima.setinterpolationmethod(pp.GetInterp()); 

  if (pp.GetInterp() == NEWIMAGE::spline && _ima.getsplineorder() != 3) _ima.setsplineorder(3); 
  if (_ima.getextrapolationmethod() != pp.GetExtrap()) _ima.setextrapolationmethod(pp.GetExtrap()); 
  if (!pp.GetExtrapValidity()) _ima.setextrapolationvalidity(false,false,false); 
  else {
    if (_acqp.PhaseEncodeVector()(1)) _ima.setextrapolationvalidity(true,false,false); 
    else _ima.setextrapolationvalidity(false,true,false); 
  }
} EddyCatch


bool ECScan::HasEmptyPlane(std::vector<unsigned int>&  pi) const EddyTry
{
  pi.clear();
  bool fe=true, le=true;
  
  for (int k=0; k<_ima.zsize(); k++) {
    for (int j=0; j<_ima.ysize(); j++) {
      if (_ima(0,j,k)) fe=false;
      if (_ima(_ima.xsize()-1,j,k)) le=false;
    }
    if (!fe && !le) break;
  }
  if (fe) pi.push_back(0);
  if (le) pi.push_back(1);
  
  fe = le = true;
  for (int k=0; k<_ima.zsize(); k++) {
    for (int i=0; i<_ima.xsize(); i++) {
      if (_ima(i,0,k)) fe=false;
      if (_ima(i,_ima.ysize()-1,k)) le=false;
    }
    if (!fe && !le) break;
  }
  if (fe) pi.push_back(2);
  if (le) pi.push_back(3);
  return(pi.size()!=0);
} EddyCatch


void ECScan::FillEmptyPlane(const std::vector<unsigned int>&  pi) EddyTry
{
  for (unsigned int d=0; d<pi.size(); d++) {
    if (pi[d]<2) { 
      unsigned int i0=0, i1=1, i2=_ima.xsize()-1; 
      if (pi[d]==1) { i0=_ima.xsize()-1; i1=_ima.xsize()-2; i2=0; }
      float w1=1.0, w2=0.0;
      if (_acqp.PhaseEncodeVector()(1)) { w1=0.5; w2=0.5; } 
      for (int k=0; k<_ima.zsize(); k++) {
	for (int j=0; j<_ima.ysize(); j++) {
	  _ima(i0,j,k) = w1*_ima(i1,j,k) + w2*_ima(i2,j,k);  
	}
      }
    }
    else if (pi[d]<4) { 
      unsigned int j0=0, j1=1, j2=_ima.ysize()-1; 
      if (pi[d]==3) { j0=_ima.ysize()-1; j1=_ima.ysize()-2; j2=0; }
      float w1=1.0, w2=0.0;
      if (_acqp.PhaseEncodeVector()(2)) { w1=0.5; w2=0.5; } 
      for (int k=0; k<_ima.zsize(); k++) {
	for (int i=0; i<_ima.xsize(); i++) {
	  _ima(i,j0,k) = w1*_ima(i,j1,k) + w2*_ima(i,j2,k);  
	}
      }
    }
    else throw EddyException("ECScan::FillEmptyPlane: Invalid plane index");
  }
} EddyCatch

NEWIMAGE::volume<float> ECScan::GetOriginalIma() const EddyTry
{
  NEWIMAGE::volume<float> vol = _ima;
  for (int sl=0; sl<vol.zsize(); sl++) {
    if (_ols[sl]) {
      float *ptr = _ols[sl]; for (int j=0; j<vol.ysize(); j++) for (int i=0; i<vol.xsize(); i++) { vol(i,j,sl) = *ptr; ptr++; }
    }
  }
  return(vol);	
} EddyCatch


NEWIMAGE::volume<float> ECScan::GetUnwarpedOriginalIma(
						       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						       
						       NEWIMAGE::volume<float>&                          omask) const EddyTry
{
  return(transform_to_model_space(this->GetOriginalIma(),susc,omask)); 
} EddyCatch

NEWIMAGE::volume<float> ECScan::GetUnwarpedOriginalIma(
						       std::shared_ptr<const NEWIMAGE::volume<float> >   susc) const EddyTry
{ 
  NEWIMAGE::volume<float> skrutt = this->GetOriginalIma(); skrutt = 0.0;
  return(transform_to_model_space(this->GetOriginalIma(),susc,skrutt)); 
} EddyCatch

EDDY::DiffPara ECScan::GetDiffPara(bool rot) const EddyTry
{
  if (rot) {
    NEWMAT::ColumnVector bvec = _diffp.bVec();
    NEWMAT::Matrix R = InverseMovementMatrix();
    bvec = R.SubMatrix(1,3,1,3)*bvec;
    EDDY::DiffPara rval(bvec,_diffp.bVal());
    return(rval);
  }
  else return(_diffp);
} EddyCatch
 

NEWIMAGE::volume<float> ECScan::GetUnwarpedIma(
					       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
					       
					       NEWIMAGE::volume<float>&                          omask) const EddyTry
{
  return(transform_to_model_space(this->GetIma(),susc,omask)); 
} EddyCatch

NEWIMAGE::volume<float> ECScan::GetUnwarpedIma(
					       std::shared_ptr<const NEWIMAGE::volume<float> >   susc) const EddyTry
{ 
  NEWIMAGE::volume<float> skrutt = this->GetIma(); skrutt = 0.0;
  return(transform_to_model_space(this->GetIma(),susc,skrutt)); 
} EddyCatch

NEWIMAGE::volume<float> ECScan::GetVolumetricUnwarpedIma(
							 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							 
							 NEWIMAGE::volume4D<float>&                        deriv) const EddyTry
{
  NEWIMAGE::volume<float> skrutt = this->GetIma(); skrutt = 0.0;
  return(transform_volumetric_to_model_space(this->GetIma(),susc,skrutt,deriv));
} EddyCatch

NEWMAT::ColumnVector ECScan::GetHz2mmVector() const EddyTry
{
  NEWMAT::ColumnVector hz2mm(3);
  hz2mm(1) = _ima.xdim() * (_acqp.PhaseEncodeVector())(1) * _acqp.ReadOutTime();
  hz2mm(2) = _ima.ydim() * (_acqp.PhaseEncodeVector())(2) * _acqp.ReadOutTime();
  hz2mm(3) = _ima.zdim() * (_acqp.PhaseEncodeVector())(3) * _acqp.ReadOutTime();

  return(hz2mm);
} EddyCatch


double ECScan::GetReg(EDDY::Parameters whichp) const EddyTry
{
  if (whichp==EDDY::EC || !this->IsSliceToVol() || this->GetRegLambda()==0.0) return(0.0);
  else return(this->GetRegLambda() * (_mp.GetParams().t() * _mp.GetHessian(this->GetMBG().NGroups()) * _mp.GetParams()).AsScalar());
} EddyCatch


NEWMAT::ColumnVector ECScan::GetRegGrad(EDDY::Parameters whichp) const EddyTry
{
  NEWMAT::ColumnVector grad(NDerivs(whichp)); grad=0.0;
  if (whichp!=EDDY::EC && this->IsSliceToVol() && this->GetRegLambda()!=0.0) grad.Rows(1,_mp.NDerivs()) = this->GetRegLambda() * _mp.GetHessian(this->GetMBG().NGroups()) * _mp.GetParams();
  return(grad);
} EddyCatch


NEWMAT::Matrix ECScan::GetRegHess(EDDY::Parameters whichp) const EddyTry
{
  NEWMAT::Matrix hess(NDerivs(whichp),NDerivs(whichp)); hess=0.0;
  if (whichp!=EDDY::EC && this->IsSliceToVol() && this->GetRegLambda()!=0.0) hess.SubMatrix(1,_mp.NDerivs(),1,_mp.NDerivs()) = this->GetRegLambda() * _mp.GetHessian(this->GetMBG().NGroups());
  return(hess);
} EddyCatch

double ECScan::GetDerivParam(unsigned int indx, EDDY::Parameters whichp) const EddyTry
{
  if (whichp==EDDY::MOVEMENT) {
    if (indx >= _mp.NDerivs()) throw EddyException("ECScan::GetDerivParam: indx out of range"); 
    return(_mp.GetParam(indx));
  }
  else if (whichp==EDDY::EC) {
    if (indx >= _ecp->NDerivs()) throw EddyException("ECScan::GetDerivParam: indx out of range"); 
    return(_ecp->GetDerivParam(indx));
  }
  else {
    if (indx >= _mp.NDerivs() + _ecp->NDerivs()) throw EddyException("ECScan::GetDerivParam: indx out of range"); 
    if (indx < _mp.NDerivs()) return(_mp.GetParam(indx));
    else { indx -= _mp.NDerivs(); return(_ecp->GetDerivParam(indx)); }
  }
  throw EddyException("ECScan::GetDerivParam: I should not be here"); 
} EddyCatch

double ECScan::GetDerivScale(unsigned int indx, EDDY::Parameters whichp) const EddyTry
{
  if (whichp==EDDY::MOVEMENT) {
    if (indx >= _mp.NDerivs()) throw EddyException("ECScan::GetDerivScale: indx out of range"); 
    return(_mp.GetDerivScale(indx));
  }
  else if (whichp==EDDY::EC) {
    if (indx >= _ecp->NDerivs()) throw EddyException("ECScan::GetDerivScale: indx out of range"); 
    return(_ecp->GetDerivScale(indx));
  }
  else {
    if (indx >= _mp.NDerivs() + _ecp->NDerivs()) throw EddyException("ECScan::GetDerivScale: indx out of range"); 
    if (indx < _mp.NDerivs()) return(_mp.GetDerivScale(indx));
    else { indx -= _mp.NDerivs(); return(_ecp->GetDerivScale(indx)); }
  }
  throw EddyException("ECScan::GetDerivScale: I should not be here"); 
} EddyCatch

EDDY::ImageCoordinates ECScan::SamplingPoints() const EddyTry
{
  EDDY::ImageCoordinates ic(this->GetIma());
  if (this->IsSliceToVol()) {
    EDDY::MultiBandGroups mbg = this->GetMBG();
    NEWMAT::Matrix Ima2World = this->GetIma().sampling_mat();
    NEWMAT::Matrix World2Ima = Ima2World.i();
    std::vector<NEWMAT::Matrix> matrices(mbg.NGroups());
    std::vector<std::vector<unsigned int> > groups(mbg.NGroups());
    for (unsigned int grp=0; grp<mbg.NGroups(); grp++) {
      groups[grp] = mbg.SlicesInGroup(grp);
      matrices[grp] = World2Ima * this->ForwardMovementMatrix(grp) * Ima2World;
    }
    ic.Transform(matrices,groups);
  }
  else ic.Transform(this->GetIma().sampling_mat().i() * this->ForwardMovementMatrix() * this->GetIma().sampling_mat());

  return(ic);
} EddyCatch

void ECScan::SetDerivParam(unsigned int indx, double p, EDDY::Parameters whichp) EddyTry
{ 
  if (whichp==EDDY::MOVEMENT) {
    if (indx >= _mp.NDerivs()) throw EddyException("ECScan::SetDerivParam: indx out of range"); else _mp.SetParam(indx,p); 
  }
  else if (whichp==EDDY::EC) {
    if (indx >= _ecp->NDerivs()) throw EddyException("ECScan::SetDerivParam: indx out of range"); else _ecp->SetDerivParam(indx,p);
  }
  else {
    if (indx >= _mp.NDerivs() + _ecp->NDerivs()) throw EddyException("ECScan::SetDerivParam: indx out of range"); 
    if (indx < _mp.NDerivs()) return(_mp.SetParam(indx,p));
    else { indx -= _mp.NDerivs(); return(_ecp->SetDerivParam(indx,p)); }
  }
  return;
} EddyCatch

void ECScan::SetParams(const NEWMAT::ColumnVector& mpep, EDDY::Parameters whichp) EddyTry
{
  if (whichp==EDDY::MOVEMENT) _mp.SetParams(mpep);
  else if (whichp==EDDY::EC) _ecp->SetParams(mpep);
  else {
    if (mpep.Nrows() != int(_mp.NParam()) + int(_ecp->NParam())) throw EddyException("ECScan::SetParams: mismatched mpep");
    else {
      _mp.SetParams(mpep.Rows(1,_mp.NParam()));
      _ecp->SetParams(mpep.Rows(_mp.NParam()+1,mpep.Nrows()));
    }
  }
  return;  
} EddyCatch


void ECScan::SetAsOutliers(const NEWIMAGE::volume<float>&                   rep,
			   std::shared_ptr<const NEWIMAGE::volume<float> >  susc,
			   const NEWIMAGE::volume<float>&                   inmask, 
			   const std::vector<unsigned int>&                 ol) EddyTry
{
  NEWIMAGE::volume<float> pios;
  NEWIMAGE::volume<float> mask;
  if (ol.size()) { 
    
    pios = EddyUtils::TransformModelToScanSpace(rep,*this,susc);
    
    mask = rep; mask = 0.0;
    NEWIMAGE::volume<float> bios = EddyUtils::transform_model_to_scan_space(inmask,*this,susc,false,mask,NULL,NULL);
    bios.binarise(0.9); 
    mask *= bios;       
  }
  this->SetAsOutliers(pios,mask,ol);
} EddyCatch


void ECScan::SetAsOutliers(const NEWIMAGE::volume<float>&                     rep,
			   const NEWIMAGE::volume<float>&                     mask, 
			   const std::vector<unsigned int>&                   ol) EddyTry
{
  
  for (unsigned int sl=0; sl<_ols.size(); sl++) {
    if (_ols[sl] && !in_list(sl,ol)) {
      float *ptr = _ols[sl];
      for (int j=0; j<_ima.ysize(); j++) {
	for (int i=0; i<_ima.xsize(); i++) {
	  _ima(i,j,sl) = *ptr; ptr++;
	}
      }
      
      delete[] _ols[sl]; _ols[sl] = 0;
    }
  }
  
  for (unsigned int ii=0; ii<ol.size(); ii++) {
    bool nol = (_ols[ol[ii]]) ? false : true;   
    if (nol) {
      _ols[ol[ii]] = new float[_ima.xsize()*_ima.ysize()];
      
    }
    float *ptr = _ols[ol[ii]];
    for (int j=0; j<_ima.ysize(); j++) {
      for (int i=0; i<_ima.xsize(); i++) {
	if (nol) { *ptr = _ima(i,j,ol[ii]); ptr++; }
	if (mask(i,j,ol[ii])) _ima(i,j,ol[ii]) = rep(i,j,ol[ii]);
      }
    }
  }
  return;
} EddyCatch

void ECScan::RecycleOutliers() EddyTry
{
  
  for (unsigned int sl=0; sl<_ols.size(); sl++) {
    if (_ols[sl]) {
      float *ptr = _ols[sl];
      for (int j=0; j<_ima.ysize(); j++) {
	for (int i=0; i<_ima.xsize(); i++) {
	  _ima(i,j,sl) = *ptr; ptr++;
	}
      }
      delete[] _ols[sl]; _ols[sl] = 0;
    }
  }
} EddyCatch

NEWIMAGE::volume<float> ECScan::motion_correct(const NEWIMAGE::volume<float>&  inima,
					       NEWIMAGE::volume<float>         *omask) const EddyTry
{
  if (this->IsSliceToVol()) EddyException("ECScan::motion_correct: Slice to vol not implemented");
  
  NEWMAT::Matrix iR = InverseMovementMatrix();
  NEWIMAGE::volume<float> ovol = inima; ovol = 0.0;
  NEWIMAGE::volume<char> mask(ovol.xsize(),ovol.ysize(),ovol.zsize()); 
  NEWIMAGE::copybasicproperties(inima,mask); mask = 1;
  NEWIMAGE::affine_transform(inima,iR,ovol,mask);
  *omask = EddyUtils::ConvertMaskToFloat(mask);
  EddyUtils::SetTrilinearInterp(*omask);
  return(ovol);
} EddyCatch

NEWIMAGE::volume<float> ECScan::transform_to_model_space(
							 const NEWIMAGE::volume<float>&                  inima,
							 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							 
							 NEWIMAGE::volume<float>&                        omask,
							 
							 bool                                            jacmod) const EddyTry
{
  if (this->IsSliceToVol()) EddyException("ECScan::transform_to_model_space: Slice to vol not implemented");
  
  NEWIMAGE::volume<float> jac;
  NEWIMAGE::volume4D<float> dfield = FieldForScanToModelTransform(susc,omask,jac);
  
  NEWMAT::Matrix iR = InverseMovementMatrix();
  NEWIMAGE::volume<float> ovol = GetIma(); ovol = 0.0;
  NEWIMAGE::volume<char> mask2(ovol.xsize(),ovol.ysize(),ovol.zsize()); 
  NEWIMAGE::copybasicproperties(inima,mask2); mask2 = 1;
  NEWIMAGE::general_transform(inima,iR,dfield,ovol,mask2);
  
  omask *= EddyUtils::ConvertMaskToFloat(mask2);
  EddyUtils::SetTrilinearInterp(omask);
  if (jacmod) return(jac*ovol);
  else return(ovol);
} EddyCatch

NEWIMAGE::volume<float> ECScan::transform_volumetric_to_model_space(
								    const NEWIMAGE::volume<float>&                  inima,
								    std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								    
								    NEWIMAGE::volume<float>&                        omask,
								    NEWIMAGE::volume4D<float>&                      deriv,
								    
								    bool                                            jacmod) const EddyTry
{
  
  NEWIMAGE::volume<float> jac;
  NEWIMAGE::volume4D<float> dfield = FieldForScanToModelTransform(susc,omask,jac);
  
  NEWMAT::Matrix iR = InverseMovementMatrix();
  NEWIMAGE::volume<float> ovol = GetIma(); ovol = 0.0;
  deriv = 0.0;
  NEWIMAGE::volume<char> mask2(ovol.xsize(),ovol.ysize(),ovol.zsize()); 
  NEWIMAGE::copybasicproperties(inima,mask2); mask2 = 1;
  NEWIMAGE::general_transform_3partial(inima,iR,dfield,ovol,deriv,mask2);
  
  omask *= EddyUtils::ConvertMaskToFloat(mask2);
  EddyUtils::SetTrilinearInterp(omask);
  if (jacmod) {
    deriv *= jac;
    return(jac*ovol);
  }
  else return(ovol);
} EddyCatch

NEWIMAGE::volume4D<float> ECScan::total_displacement_to_model_space(
								    const NEWIMAGE::volume<float>&                  inima,
								    std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								    
								    bool                                            movement_only,
								    bool                                            exclude_PE_tr) const EddyTry
{
  
  NEWIMAGE::volume4D<float> dfield = FieldForScanToModelTransform(susc);
  if (movement_only) dfield = 0.0;
  
  NEWMAT::Matrix iR; 
  if (exclude_PE_tr) {
    NEWMAT::ColumnVector pe = this->GetAcqPara().PhaseEncodeVector();
    std::vector<unsigned int> rindx;
    if (pe(1) != 0) rindx.push_back(0);
    if (pe(2) != 0) rindx.push_back(1);
    iR = RestrictedInverseMovementMatrix(rindx);
  }
  else iR = InverseMovementMatrix();
  NEWIMAGE::volume4D<float> ofield = dfield; ofield = 0.0;
  NEWIMAGE::get_displacement_fields(inima,iR,dfield,ofield);
  return(ofield);
} EddyCatch



NEWIMAGE::volume4D<float> ECScan::field_for_scan_to_model_transform(
								    std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								    
								    NEWIMAGE::volume<float>                         *omask,
								    NEWIMAGE::volume<float>                         *jac) const EddyTry
{
  NEWIMAGE::volume<float> eb = this->ECField();
  NEWIMAGE::volume<float> tot = this->GetIma(); tot = 0.0;
  NEWIMAGE::volume<char> mask1(tot.xsize(),tot.ysize(),tot.zsize()); 
  NEWIMAGE::copybasicproperties(tot,mask1); mask1 = 1;  
  if (this->IsSliceToVol()) {
    for (unsigned int tp=0; tp<_mbg.NGroups(); tp++) { 
      NEWMAT::Matrix iR = this->InverseMovementMatrix(tp);
      std::vector<unsigned int> slices = _mbg.SlicesAtTimePoint(tp);
      NEWIMAGE::affine_transform(eb,iR,slices,tot,mask1); 
    }
  }
  else {
    
    NEWMAT::Matrix iR = this->InverseMovementMatrix(); 
    
    NEWIMAGE::affine_transform(eb,iR,tot,mask1);          
  }
  if (omask) {
    *omask = EddyUtils::ConvertMaskToFloat(mask1);
    EddyUtils::SetTrilinearInterp(*omask);
  }
  
  if (susc) tot += *susc;
  
  NEWIMAGE::volume4D<float> dfield = FieldUtils::Hz2VoxelDisplacements(tot,this->GetAcqPara());
  
  if (jac) *jac = FieldUtils::GetJacobian(dfield,this->GetAcqPara());
  
  dfield = FieldUtils::Voxel2MMDisplacements(dfield);

  return(dfield);
} EddyCatch





NEWIMAGE::volume4D<float> ECScan::field_for_model_to_scan_transform(
								    std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								    
								    NEWIMAGE::volume<float>                         *omask,
								    NEWIMAGE::volume<float>                         *jac) const EddyTry
{
  
  NEWIMAGE::volume<float> tot = this->ECField();
  NEWIMAGE::volume<char> mask1(this->GetIma().xsize(),this->GetIma().ysize(),this->GetIma().zsize()); 
  NEWIMAGE::copybasicproperties(this->GetIma(),mask1); mask1 = 1;  
  if (susc) {
    NEWIMAGE::volume<float> tsusc = *susc; tsusc = 0.0;              
    if (this->IsSliceToVol()) {
      for (unsigned int tp=0; tp<_mbg.NGroups(); tp++) { 
	NEWMAT::Matrix R = this->ForwardMovementMatrix(tp);
	std::vector<unsigned int> slices = _mbg.SlicesAtTimePoint(tp);
	NEWIMAGE::affine_transform(*susc,R,slices,tsusc,mask1); 
      }
    }
    else {
      NEWMAT::Matrix R = this->ForwardMovementMatrix();
      NEWIMAGE::affine_transform(*susc,R,tsusc,mask1); 
    }
    tot += tsusc;
  }
  
  NEWIMAGE::volume4D<float> dfield = FieldUtils::Hz2VoxelDisplacements(tot,this->GetAcqPara());
  
  bool own_mask = false;
  if (!omask) { omask = new NEWIMAGE::volume<float>(mask1.xsize(),mask1.ysize(),mask1.zsize()); own_mask=true; }
  *omask = 1.0;  
  NEWIMAGE::volume4D<float> idfield = FieldUtils::Invert3DDisplacementField(dfield,this->GetAcqPara(),EddyUtils::ConvertMaskToFloat(mask1),*omask);
  EddyUtils::SetTrilinearInterp(*omask);
  if (own_mask) delete omask;
  
  if (jac) *jac = FieldUtils::GetJacobian(idfield,this->GetAcqPara());
  
  idfield = FieldUtils::Voxel2MMDisplacements(idfield);

  return(idfield);
} EddyCatch



ECScanManager::ECScanManager(const std::string&               imafname,
			     const std::string&               maskfname,
			     const std::string&               acqpfname,
			     const std::string&               topupfname,
			     const std::string&               fieldfname,
			     const std::string&               field_mat_fname,
			     const std::string&               bvecsfname,
			     const std::string&               bvalsfname,
			     EDDY::ECModel                    ecmodel,
			     EDDY::ECModel                    b0_ecmodel,
			     const std::vector<unsigned int>& indicies,
			     const EDDY::PolationPara&        pp,
			     EDDY::MultiBandGroups            mbg,
			     bool                             fsh) EddyTry
: _has_susc_field(false), _bias_field(nullptr), _pp(pp), _fsh(fsh), _use_b0_4_dwi(true)
{
  
  TOPUP::TopupDatafileReader tdfr(acqpfname);
  
  std::shared_ptr<TOPUP::TopupFileReader> tfrp;
  std::string tmpfname;
  if (topupfname != string("") && fieldfname != string("")) throw EddyException("ECScanManager::ECScanManager: Cannot specify both topupfname and fieldfname");
  else if (topupfname != string("")) tmpfname = topupfname;
  else if (fieldfname != string("")) tmpfname = fieldfname;
  if (tmpfname != string("")) { 
    tfrp = std::shared_ptr<TOPUP::TopupFileReader>(new TOPUP::TopupFileReader(tmpfname));
    if (field_mat_fname != string("")) {
      NEWMAT::Matrix fieldM = MISCMATHS::read_ascii_matrix(field_mat_fname);
      _susc_field = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(tfrp->FieldAsVolume(fieldM)));
    }
    else _susc_field = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(tfrp->FieldAsVolume()));
    EddyUtils::SetSplineInterp(*_susc_field);
    _susc_field->forcesplinecoefcalculation(); 
    _has_susc_field = true;
  }
  
  _susc_d1.resize(6,nullptr);
  _susc_d2.resize(6);
  for (unsigned int i=0; i<_susc_d2.size(); i++) _susc_d2[i].resize(i+1,nullptr);
  
  NEWMAT::Matrix bvecsM = MISCMATHS::read_ascii_matrix(bvecsfname);
  if (bvecsM.Nrows()>bvecsM.Ncols()) bvecsM = bvecsM.t();
  NEWMAT::Matrix bvalsM = MISCMATHS::read_ascii_matrix(bvalsfname);
  if (bvalsM.Nrows()>bvalsM.Ncols()) bvalsM = bvalsM.t();
  
  NEWIMAGE::read_volume(_mask,maskfname);
  EddyUtils::SetTrilinearInterp(_mask);
  
  NEWIMAGE::volume4D<float> all;
  NEWIMAGE::read_volume4D(all,imafname);
  _sf = 100.0 / mean_of_first_b0(all,_mask,bvecsM,bvalsM);
  _fi.resize(all.tsize());
  for (int s=0; s<all.tsize(); s++) {
    EDDY::AcqPara acqp(tdfr.PhaseEncodeVector(indicies[s]),tdfr.ReadOutTime(indicies[s]));
    EDDY::DiffPara dp(bvecsM.Column(s+1),bvalsM(1,s+1));
    EDDY::ScanMovementModel smm(0); 
    if (EddyUtils::IsDiffusionWeighted(dp)) {
      std::shared_ptr<EDDY::ScanECModel> ecp;
      switch (ecmodel) {
      case EDDY::NoEC:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new NoECScanECModel());
	break;
      case EDDY::Linear:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new LinearScanECModel(_has_susc_field));
	break;
      case EDDY::Quadratic:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new QuadraticScanECModel(_has_susc_field));
	break;
      case EDDY::Cubic:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new CubicScanECModel(_has_susc_field));
	break;
      default:
        throw EddyException("ECScanManager::ECScanManager: Invalid EC model");
      }
      if (_has_susc_field && topupfname != string("")) {
	NEWMAT::Matrix fwd_matrix = TOPUP::MovePar2Matrix(tfrp->MovePar(indicies[s]),all[s]);
	NEWMAT::ColumnVector bwd_mp = TOPUP::Matrix2MovePar(fwd_matrix.i(),all[s]);
	smm.SetParams(bwd_mp);
      }
      NEWIMAGE::volume<float> tmp = all[s]*_sf;
      _scans.push_back(ECScan(tmp,acqp,dp,smm,mbg,ecp));
      _scans.back().SetPolation(pp);
      _fi[s].first = 0; _fi[s].second = _scans.size() - 1;
    }
    else {
      std::shared_ptr<EDDY::ScanECModel> ecp;
      switch (b0_ecmodel) {
      case EDDY::NoEC:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new NoECScanECModel());
	break;
      case EDDY::Linear:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new LinearScanECModel(_has_susc_field));
	break;
      case EDDY::Quadratic:
	ecp = std::shared_ptr<EDDY::ScanECModel>(new QuadraticScanECModel(_has_susc_field));
	break;
      default:
        throw EddyException("ECScanManager::ECScanManager: Invalid b0 EC model");
      }
      if (_has_susc_field && topupfname != string("")) {
	NEWMAT::Matrix fwd_matrix = TOPUP::MovePar2Matrix(tfrp->MovePar(indicies[s]),all[s]);
	NEWMAT::ColumnVector bwd_mp = TOPUP::Matrix2MovePar(fwd_matrix.i(),all[s]);
	smm.SetParams(bwd_mp);
      }
      NEWIMAGE::volume<float> tmp = all[s]*_sf;
      _b0scans.push_back(ECScan(tmp,acqp,dp,smm,mbg,ecp));
      _b0scans.back().SetPolation(pp);
      _fi[s].first = 1; _fi[s].second = _b0scans.size() - 1;
    }
  }
  std::vector<double> skrutt;
  _refs = ReferenceScans(GetB0Indicies(),GetShellIndicies(skrutt));
} EddyCatch

unsigned int ECScanManager::NScans(ScanType st) const EddyTry
{
  unsigned int rval = 0;
  switch (st) {
  case ANY: rval = _scans.size()+_b0scans.size(); break;
  case DWI: rval = _scans.size(); break;
  case B0: rval = _b0scans.size(); break;
  default: break;
  } 
  return(rval);
} EddyCatch

bool ECScanManager::IsShelled() const EddyTry
{
  if (_fsh) return(true);
  else {
    std::vector<DiffPara> dpv = this->GetDiffParas(DWI);
    return(EddyUtils::IsShelled(dpv));				      
  }
} EddyCatch

unsigned int ECScanManager::NoOfShells(ScanType st) const EddyTry
{
  if (st==B0) return((_b0scans.size()>0) ? 1 : 0);
  std::vector<DiffPara> dpv = this->GetDiffParas(DWI);
  std::vector<unsigned int> grpi;  
  std::vector<double> grpb;
  if (!EddyUtils::GetGroups(dpv,grpi,grpb) && !_fsh) throw EddyException("ECScanManager::NoOfShells: Data not shelled");
  if (st==ANY) return(grpb.size() + ((_b0scans.size()>0) ? 1 : 0)); else return(grpb.size());
} EddyCatch

std::vector<DiffPara> ECScanManager::GetDiffParas(ScanType st) const EddyTry
{
  std::vector<DiffPara> dpv(this->NScans(st));
  for (unsigned int i=0; i<this->NScans(st); i++) dpv[i] = this->Scan(i,st).GetDiffPara();
  return(dpv);
} EddyCatch

std::vector<unsigned int> ECScanManager::GetB0Indicies() const EddyTry
{
  std::vector<unsigned int> b0_indx;
  for (unsigned int i=0; i<this->NScans(ANY); i++) {
    if (EddyUtils::Isb0(this->Scan(i,ANY).GetDiffPara())) b0_indx.push_back(i);
  }
  return(b0_indx);
} EddyCatch

std::vector<std::vector<unsigned int> > ECScanManager::GetShellIndicies(std::vector<double>& bvals) const EddyTry
{
  std::vector<EDDY::DiffPara> dpv = this->GetDiffParas(ANY);
  std::vector<std::vector<unsigned int> > shindx;
  if (!EddyUtils::GetGroups(dpv,shindx,bvals) && !_fsh) throw EddyException("ECScanManager::GetShellIndicies: Data not shelled");
  if (EddyUtils::Isb0(this->Scan(shindx[0][0],ANY).GetDiffPara())) { 
    shindx.erase(shindx.begin());
    bvals.erase(bvals.begin());
  }
  return(shindx);
} EddyCatch

unsigned int ECScanManager::NLSRPairs(ScanType st) const EddyTry
{
  unsigned int rval = 0;
  if (!CanDoLSRResampling()) throw EddyException("ECScanManager::NLSRPairs: Data not suited for LS-recon");
  else rval = NScans(st)/2;
  return(rval);
} EddyCatch


void ECScanManager::FillEmptyPlanes() EddyTry
{
  for (unsigned int i=0; i<_scans.size(); i++) {
    std::vector<unsigned int> pi;
    if (_scans[i].HasEmptyPlane(pi)) _scans[i].FillEmptyPlane(pi);
  }
  for (unsigned int i=0; i<_b0scans.size(); i++) {
    std::vector<unsigned int> pi;
    if (_b0scans[i].HasEmptyPlane(pi)) _b0scans[i].FillEmptyPlane(pi);
  }
} EddyCatch


std::vector<unsigned int> ECScanManager::GetDwi2GlobalIndexMapping() const EddyTry
{
  std::vector<unsigned int> i2i(_scans.size());
  for (unsigned int i=0; i<_fi.size(); i++) {
    if (!_fi[i].first) i2i[_fi[i].second] = i;
  }
  return(i2i);
} EddyCatch
unsigned int ECScanManager::GetDwi2GlobalIndexMapping(unsigned int dwindx) const EddyTry
{
  if (dwindx>=_scans.size()) throw EddyException("ECScanManager::GetDwi2GlobalIndexMapping: Invalid dwindx");
  else {
    for (unsigned int i=0; i<_fi.size(); i++) {
      if (!_fi[i].first && _fi[i].second == int(dwindx)) return(i);
    }
  }
  throw EddyException("ECScanManager::GetDwi2GlobalIndexMapping: Global mapping not found");
} EddyCatch


std::vector<unsigned int> ECScanManager::Getb02GlobalIndexMapping() const EddyTry
{
  std::vector<unsigned int> i2i(_b0scans.size());
  for (unsigned int i=0; i<_fi.size(); i++) {
    if (_fi[i].first) i2i[_fi[i].second] = i;
  }
  return(i2i);
} EddyCatch
unsigned int ECScanManager::Getb02GlobalIndexMapping(unsigned int b0indx) const EddyTry
{
  if (b0indx>=_b0scans.size()) throw EddyException("ECScanManager::Getb02GlobalIndexMapping: Invalid b0indx");
  else {
    for (unsigned int i=0; i<_fi.size(); i++) {
      if (_fi[i].first && _fi[i].second == int(b0indx)) return(i);
    }
  }
  throw EddyException("ECScanManager::Getb02GlobalIndexMapping: Global mapping not found");
} EddyCatch

unsigned int ECScanManager::GetGlobal2DWIIndexMapping(unsigned int gindx) const EddyTry
{
  if (_fi[gindx].first) throw EddyException("ECScanManager::GetGlobal2DWIIndexMapping: Global index not dwi");
  return(_fi[gindx].second);
} EddyCatch

unsigned int ECScanManager::GetGlobal2b0IndexMapping(unsigned int gindx) const EddyTry
{
  if (!_fi[gindx].first) throw EddyException("ECScanManager::GetGlobal2b0IndexMapping: Global index not b0");
  return(_fi[gindx].second);
} EddyCatch


bool ECScanManager::B0sAreInterspersed() const EddyTry
{
  unsigned int nall = NScans();
  std::vector<unsigned int> b02g = Getb02GlobalIndexMapping();
  if (b02g.size() > 2 && b02g[0] < 0.25*nall && b02g.back() > 0.75*nall) return(true);
  else return(false);
} EddyCatch


bool ECScanManager::B0sAreUsefulForPEAS() const EddyTry
{
  unsigned int nall = NScans();
  std::vector<unsigned int> b02g = Getb02GlobalIndexMapping();
  if (b02g.size() > 3 &&                           
      double(b02g.size())/double(nall) > 0.05 &&   
      !indicies_clustered(b02g,nall)) {            
    return(true);
  }
  else return(false);
} EddyCatch


void ECScanManager::PolateB0MovPar() EddyTry
{
  std::vector<unsigned int> b0g = Getb02GlobalIndexMapping();
  std::vector<unsigned int> dwig = GetDwi2GlobalIndexMapping();
  for (unsigned int i=0; i<dwig.size(); i++) {
    std::pair<int,int> br = bracket(dwig[i],b0g);
    NEWMAT::ColumnVector mp = interp_movpar(dwig[i],br);
    Scan(dwig[i],EDDY::ANY).SetParams(mp,EDDY::MOVEMENT);
  }
} EddyCatch

std::pair<unsigned int,unsigned int> ECScanManager::GetLSRPair(unsigned int i, ScanType st) const EddyTry
{
  std::pair<unsigned int,unsigned int> rval(0,0);
  if (!CanDoLSRResampling()) throw EddyException("ECScanManager::GetLSRPair: Data not suited for LS-recon");
  else if (i >= NLSRPairs(st)) throw EddyException("ECScanManager::GetLSRPair: Index out of range");
  else {
    rval.first = i; rval.second = NLSRPairs(st) + i;
  }  
  return(rval);
} EddyCatch

bool ECScanManager::CanDoLSRResampling() const EddyTry
{
  
  
  NEWMAT::ColumnVector blip1 = Scan(0,ANY).GetAcqPara().PhaseEncodeVector();
  unsigned int n_dir1 = 1;
  for (; n_dir1<NScans(ANY); n_dir1++) {
    if (Scan(n_dir1,ANY).GetAcqPara().PhaseEncodeVector() != blip1) break;
  }
  if (n_dir1 != NScans(ANY)/2) { cout << "n_dir1 = " << n_dir1 << ", NScans(ANY)/2 = " << NScans(ANY)/2 << endl; return(false); }
  
  
  for (unsigned int i=0; i<n_dir1; i++) {
    if (!EddyUtils::AreMatchingPair(Scan(i,ANY),Scan(i+n_dir1,ANY))) { cout << "Scans " << i << " and " << i+n_dir1 << " not a match" << endl; return(false); }
  }
  return(true);
} EddyCatch



void ECScanManager::SetParameters(const NEWMAT::Matrix& pM, ScanType st) EddyTry
{
  if (pM.Nrows() != int(NScans(st))) throw EddyException("ECScanManager::SetParameters: Mismatch between parameter matrix and # of scans");
  for (unsigned int i=0; i<NScans(st); i++) {
    int ncol = Scan(i,st).NParam();
    Scan(i,st).SetParams(pM.SubMatrix(i+1,i+1,1,ncol).t());
  } 
} EddyCatch

const ECScan& ECScanManager::Scan(unsigned int indx, ScanType st) const EddyTry
{
  if (!index_kosher(indx,st))  throw EddyException("ECScanManager::Scan: index out of range"); 
  if (st == DWI) return(_scans[indx]);
  else if (st == B0) return(_b0scans[indx]); 
  else { 
    if (!_fi[indx].first) return(_scans[_fi[indx].second]);
    else return(_b0scans[_fi[indx].second]);
  }   
} EddyCatch

ECScan& ECScanManager::Scan(unsigned int indx, ScanType st) EddyTry
{
  if (!index_kosher(indx,st))  throw EddyException("ECScanManager::Scan: index out of range"); 
  if (st == DWI) return(_scans[indx]);
  else if (st == B0) return(_b0scans[indx]); 
  else { 
    if (!_fi[indx].first) return(_scans[_fi[indx].second]);
    else return(_b0scans[_fi[indx].second]);
  }   
} EddyCatch

NEWIMAGE::volume<float> ECScanManager::GetUnwarpedOrigScan(unsigned int              indx,
							   NEWIMAGE::volume<float>&  omask,
							   ScanType                  st) const EddyTry
{
  if (!index_kosher(indx,st))  throw EddyException("ECScanManager::GetUnwarpedOrigScan: index out of range");
  return(Scan(indx,st).GetUnwarpedOriginalIma(this->GetSuscHzOffResField(indx,st),omask)); 
} EddyCatch

NEWIMAGE::volume<float> ECScanManager::GetUnwarpedScan(unsigned int              indx,
						       NEWIMAGE::volume<float>&  omask,
						       ScanType                  st) const EddyTry
{
  if (!index_kosher(indx,st))  throw EddyException("ECScanManager::GetUnwarpedScan: index out of range");
  return(Scan(indx,st).GetUnwarpedIma(this->GetSuscHzOffResField(indx,st),omask)); 
} EddyCatch


void ECScanManager::AddRotation(const std::vector<float>& rot) EddyTry
{
  float pi = 3.141592653589793;
  for (unsigned int i=0; i<NScans(); i++) {
    NEWMAT::ColumnVector mp = Scan(i).GetParams(EDDY::MOVEMENT);
    for (unsigned int r=0; r<3; r++) mp(4+r) += pi*rot[r]/180.0;
    Scan(i).SetParams(mp,EDDY::MOVEMENT);
  }
} EddyCatch


std::vector<unsigned int> ECScanManager::IntraCerebralSlices(unsigned int nvox) const EddyTry
{
  std::vector<unsigned int> ics;
  for (int k=0; k<_mask.zsize(); k++) {
    unsigned int svox = 0;
    for (int j=0; j<_mask.ysize(); j++) {
      for (int i=0; i<_mask.xsize(); i++) { if (_mask(i,j,k)) svox++; }
      if (svox > nvox) break;
    }
    if (svox > nvox) ics.push_back(static_cast<unsigned int>(k));
  }
  return(ics);
} EddyCatch

std::shared_ptr<const NEWIMAGE::volume<float> > ECScanManager::GetSuscHzOffResField(unsigned int indx, 
										    ScanType     st) const EddyTry
{
  if (this->has_move_by_susc_fields()) {
    NEWMAT::ColumnVector p = this->Scan(indx,st).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
    if (p.Nrows() != static_cast<int>(_susc_d1.size()) || p.Nrows() != static_cast<int>(_susc_d2.size())) throw EddyException("ECScanManager::GetSuscHzOffResField: mismatch between no. of movement parameters and no. of susc derivative fields");
    NEWIMAGE::volume<float> f = *_susc_field;
    
    for (unsigned int i=0; i<_susc_d1.size(); i++) {
      if (_susc_d1[i] != nullptr) f += static_cast<float>(p(i+1)) * *(_susc_d1[i]);
    }
    
    for (unsigned int i=0; i<_susc_d2.size(); i++) {
      for (unsigned int j=0; j<_susc_d2[i].size(); j++) {
	if (_susc_d2[i][j] != nullptr) f += static_cast<float>(p(i+1)*p(j+1)) * *(_susc_d2[i][j]);
      }
    }
    std::shared_ptr<const NEWIMAGE::volume<float> > rval = std::shared_ptr<const NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(f));
    return(rval);
  }
  
  return(_susc_field);
} EddyCatch

void ECScanManager::SetDerivSuscField(unsigned int                   pi, 
				      const NEWIMAGE::volume<float>& dfield) EddyTry
{
  if (pi >= _susc_d1.size()) throw EddyException("ECScanManager::SetDerivSuscField: pi out of range");
  if (_susc_d1[pi] == nullptr) _susc_d1[pi] = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(dfield));
  else *_susc_d1[pi] = dfield;
} EddyCatch

void ECScanManager::Set2ndDerivSuscField(unsigned int                   pi, 
					 unsigned int                   pj,
					 const NEWIMAGE::volume<float>& dfield) EddyTry
{
  if (pi >= _susc_d2.size()) throw EddyException("ECScanManager::Set2ndDerivSuscField: pi out of range");;
  if (pj >= _susc_d2[pi].size()) throw EddyException("ECScanManager::Set2ndDerivSuscField: pj out of range");;
  if (_susc_d2[pi][pj] == nullptr) _susc_d2[pi][pj] = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(dfield));
  else *_susc_d2[pi][pj] = dfield;
} EddyCatch


void ECScanManager::SeparateFieldOffsetFromMovement(EDDY::ScanType    st,           
						    EDDY::OffsetModel m) EddyTry    
{
  if (st==ANY) throw EddyException("ECScanManager::SeparateFieldOffsetFromMovement: Does not make sense to model b0 and DWI together.");
  if (HasFieldOffset(st)) { 
    
    NEWMAT::ColumnVector hz = hz_vector_with_everything(st);
    
    NEWMAT::Matrix X;
    if (m == EDDY::LinearOffset) X = linear_design_matrix(st);
    else if (m == EDDY::QuadraticOffset) X = quadratic_design_matrix(st);
    else throw EddyException("ECScanManager::SeparateFieldOffsetFromMovement: Invalid offset model.");
    
    NEWMAT::Matrix H = X*(X.t()*X).i()*X.t();
    NEWMAT::ColumnVector hz_hat = H*hz;
    
    for (unsigned int i=0; i<NScans(st); i++) {
      Scan(i,st).SetFieldOffset(hz_hat(i+1));
    }
    
    for (unsigned int i=0; i<NScans(st); i++) {
      NEWMAT::ColumnVector mp = Scan(i,st).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
      for (int j=1; j<=3; j++) {
	if (Scan(i,st).GetHz2mmVector()(j)) {
	  mp(j) = (hz(i+1) - hz_hat(i+1))*Scan(i,st).GetHz2mmVector()(j);
	}
      }
      Scan(i,st).SetParams(mp,EDDY::MOVEMENT);
    }
    
    if (st==B0) ApplyB0LocationReference();    
    else ApplyDWILocationReference();
    ApplyLocationReference();
  }
} EddyCatch

void ECScanManager::SetPredictedECParam(ScanType           st,
					SecondLevelECModel slm) EddyTry
{
  if (Scan(0,st).Model() != EDDY::NoEC) { 
    
    NEWMAT::Matrix X;
    if (slm == EDDY::Linear_2nd_lvl_mdl) X = linear_design_matrix(st);
    else if (slm == EDDY::Quadratic_2nd_lvl_mdl) X = quadratic_design_matrix(st);
    else throw EddyException("ECScanManager::SetPredictedECParam: Invalid 2nd level model.");
    NEWMAT::Matrix H = X*(X.t()*X).i()*X.t();
    
    NEWMAT::Matrix ecp(NScans(st),Scan(0,st).NParam(EDDY::EC));
    for (unsigned int i=0; i<NScans(st); i++) {
      ecp.Row(i+1) = Scan(i,st).GetParams(EDDY::EC).t();
    }
    
    NEWMAT::Matrix pecp(ecp.Nrows(),ecp.Ncols());
    for (int i=0; i<ecp.Ncols()-1; i++) {
      pecp.Column(i+1) = H * ecp.Column(i+1);
    }
    
    pecp.Column(ecp.Ncols()) = ecp.Column(ecp.Ncols());
    
    for (unsigned int i=0; i<NScans(st); i++) {
      Scan(i,st).SetParams(pecp.Row(i+1).t(),EDDY::EC);
    }
  }  
} EddyCatch

void ECScanManager::WriteParameterFile(const std::string& fname, ScanType st) const EddyTry
{
  NEWMAT::Matrix  params;
  if (st == B0) params.ReSize(Scan(0,B0).NParam(EDDY::ZERO_ORDER_MOVEMENT)+Scan(0,B0).NParam(EDDY::EC),NScans(st));
  else params.ReSize(Scan(0,DWI).NParam(EDDY::ZERO_ORDER_MOVEMENT)+Scan(0,DWI).NParam(EDDY::EC),NScans(st));
  int b0_np = Scan(0,B0).NParam(EDDY::ZERO_ORDER_MOVEMENT)+Scan(0,B0).NParam(EDDY::EC);
  params = 0.0;
  for (unsigned int i=0; i<NScans(st); i++) {
    if (st==DWI || (st==ANY && IsDWI(i))) params.Column(i+1) = Scan(i,st).GetParams(EDDY::ZERO_ORDER_MOVEMENT) & Scan(i,st).GetParams(EDDY::EC);
    else params.SubMatrix(1,b0_np,i+1,i+1) = Scan(i,st).GetParams(EDDY::ZERO_ORDER_MOVEMENT) & Scan(i,st).GetParams(EDDY::EC);
  }
  MISCMATHS::write_ascii_matrix(fname,params.t());
} EddyCatch

void ECScanManager::WriteMovementOverTimeFile(const std::string& fname, ScanType st) const EddyTry
{
  NEWMAT::Matrix mot;
  unsigned int tppv = this->MultiBand().NGroups(); 
  mot.ReSize(6,NScans(st)*tppv);
  mot = 0.0;
  unsigned int mot_cntr = 1;
  for (unsigned int i=0; i<NScans(st); i++) {
    for (unsigned int j=0; j<tppv; j++) {
      mot.Column(mot_cntr++) = TOPUP::Matrix2MovePar(Scan(i,st).ForwardMovementMatrix(j),Scan(i,st).GetIma());
    }
  }
  MISCMATHS::write_ascii_matrix(fname,mot.t());
} EddyCatch

void ECScanManager::WriteECFields(const std::string& fname, ScanType st) const EddyTry
{
  NEWIMAGE::volume4D<float> ovol(Scan(0).GetIma().xsize(),Scan(0).GetIma().ysize(),Scan(0).GetIma().zsize(),NScans(st));
  NEWIMAGE::copybasicproperties(Scan(0).GetIma(),ovol);
  for (unsigned int i=0; i<NScans(st); i++) ovol[i] = GetScanHzECOffResField(i,st);
  NEWIMAGE::write_volume(ovol,fname);
} EddyCatch

void ECScanManager::WriteRotatedBVecs(const std::string& fname, ScanType st) const EddyTry
{
  NEWMAT::Matrix bvecs(3,NScans(st));
  for (unsigned int i=0; i<NScans(st); i++) bvecs.Column(i+1) = Scan(i,st).GetDiffPara(true).bVec();
  MISCMATHS::write_ascii_matrix(fname,bvecs);
} EddyCatch

void ECScanManager::WriteMovementRMS(const std::string& fname, ScanType st) const EddyTry
{
  NEWMAT::Matrix rms(NScans(st),2);
  NEWIMAGE::volume<float> mask = Mask();
  NEWIMAGE::volume4D<float> mov_field;
  NEWIMAGE::volume4D<float> prev_mov_field;
  for (unsigned int s=0; s<NScans(st); s++) {
    if (s) prev_mov_field = mov_field;
    mov_field = Scan(s,st).MovementDisplacementToModelSpace(GetSuscHzOffResField(s,st));
    NEWIMAGE::volume4D<float> sqr_mov_field = mov_field * mov_field;     
    NEWIMAGE::volume<float> sqr_norm = NEWIMAGE::sumvol(sqr_mov_field);  
    double ms = sqr_norm.mean(mask);
    rms(s+1,1) = std::sqrt(ms);
    if (s) {
      NEWIMAGE::volume4D<float> delta_field  = mov_field - prev_mov_field;
      delta_field *= delta_field; 
      sqr_norm = NEWIMAGE::sumvol(delta_field); 
      ms = sqr_norm.mean(mask);
      rms(s+1,2) = std::sqrt(ms);
    }
    else rms(s+1,2) = 0.0;
  }
  MISCMATHS::write_ascii_matrix(rms,fname);
} EddyCatch

void ECScanManager::WriteRestrictedMovementRMS(const std::string& fname, ScanType st) const EddyTry
{
  NEWMAT::Matrix rms(NScans(st),2);
  NEWIMAGE::volume<float> mask = Mask();
  NEWIMAGE::volume4D<float> mov_field;
  NEWIMAGE::volume4D<float> prev_mov_field;
  for (unsigned int s=0; s<NScans(st); s++) {
    if (s) prev_mov_field = mov_field;
    mov_field = Scan(s,st).RestrictedMovementDisplacementToModelSpace(GetSuscHzOffResField(s,st));
    NEWIMAGE::volume4D<float> sqr_mov_field = mov_field * mov_field; 
    NEWIMAGE::volume<float> sqr_norm = NEWIMAGE::sumvol(sqr_mov_field); 
    double ms = sqr_norm.mean(mask);
    rms(s+1,1) = std::sqrt(ms);
    if (s) {
      NEWIMAGE::volume4D<float> delta_field  = mov_field - prev_mov_field;
      delta_field *= delta_field; 
      sqr_norm = NEWIMAGE::sumvol(delta_field); 
      ms = sqr_norm.mean(mask);
      rms(s+1,2) = std::sqrt(ms);
    }
    else rms(s+1,2) = 0.0;
  }
  MISCMATHS::write_ascii_matrix(rms,fname);
} EddyCatch



void ECScanManager::WriteDisplacementFields(const std::string& basefname, ScanType st) const EddyTry
{
  for (unsigned int s=0; s<NScans(st); s++) {
    NEWIMAGE::volume4D<float> dfield = Scan(s,st).TotalDisplacementToModelSpace(GetSuscHzOffResField(s,st));
    char fname[256]; sprintf(fname,"%s.%03d",basefname.c_str(),s+1);
    NEWIMAGE::write_volume(dfield,fname);
  }
} EddyCatch

void ECScanManager::WriteOutlierFreeData(const std::string& fname, ScanType st) const EddyTry
{
  NEWIMAGE::volume4D<float> ovol(Scan(0,st).GetIma().xsize(),Scan(0,st).GetIma().ysize(),Scan(0,st).GetIma().zsize(),NScans(st));
  NEWIMAGE::copybasicproperties(Scan(0,st).GetIma(),ovol);
  for (unsigned int i=0; i<NScans(st); i++) {
    NEWIMAGE::volume<float> tmp = Scan(i,st).GetIma() / ScaleFactor();
    {
      ovol[i] = tmp;
    }
  }
  NEWIMAGE::write_volume(ovol,fname);
} EddyCatch


NEWMAT::ColumnVector ECScanManager::hz_vector_with_everything(ScanType st) const EddyTry
{
  NEWMAT::ColumnVector hz(NScans(st));
  for (unsigned int i=0; i<NScans(st); i++) {
    NEWMAT::Matrix X = Scan(i,st).GetHz2mmVector();
    NEWMAT::ColumnVector ymm = Scan(i,st).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
    hz(i+1) = Scan(i,st).GetFieldOffset() + (MISCMATHS::pinv(X)*ymm.Rows(1,3)).AsScalar();
  }
  return(hz);
} EddyCatch

NEWMAT::Matrix ECScanManager::linear_design_matrix(ScanType st) const EddyTry
{
  NEWMAT::Matrix X(NScans(st),3);
  for (unsigned int i=0; i<NScans(st); i++) {
    X.Row(i+1) = (Scan(i,st).GetDiffPara().bVal()/1000.0) * Scan(i,st).GetDiffPara().bVec().t();
  }
  return(X);
} EddyCatch

NEWMAT::Matrix ECScanManager::quadratic_design_matrix(ScanType st) const EddyTry
{
  NEWMAT::Matrix X = linear_design_matrix(st);
  NEWMAT::Matrix qX(NScans(st),6);
  for (unsigned int i=0; i<NScans(st); i++) {
    NEWMAT::RowVector r(6);
    double bvsq = MISCMATHS::Sqr(Scan(i,st).GetDiffPara().bVal()/1000.0);
    NEWMAT::ColumnVector c = Scan(i,st).GetDiffPara().bVec();
    r << c(1)*c(1) << c(2)*c(2) << c(3)*c(3) << c(1)*c(2) << c(1)*c(3) << c(2)*c(3);
    r *= bvsq;
    qX.Row(i+1) = r;
  }
  X |= qX;
  return(X);
} EddyCatch

NEWMAT::Matrix ECScanManager::get_b0_movement_vector(ScanType st) const EddyTry
{
  NEWMAT::Matrix skrutt;
  throw EddyException("ECScanManager::get_b0_movement_vector: Not yet implemented");
  return(skrutt);
} EddyCatch

void ECScanManager::set_reference(unsigned int ref,  
				  ScanType     st) EddyTry
{
  if (ref >= NScans(st)) throw EddyException("ECScanManager::set_reference: ref index out of bounds");
  NEWMAT::Matrix Mr = Scan(ref,st).ForwardMovementMatrix();
  for (unsigned int i=0; i<NScans(st); i++) {
    NEWMAT::Matrix M = Scan(i,st).ForwardMovementMatrix();
    NEWMAT::ColumnVector new_mp = TOPUP::Matrix2MovePar(M*Mr.i(),Scan(i,st).GetIma());
    Scan(i,st).SetParams(new_mp,MOVEMENT);
  }
} EddyCatch

void ECScanManager::set_slice_to_vol_reference(unsigned int ref,         
					       ScanType      st,
					       int           si) EddyTry 
{
  if (st==ANY) throw EddyException("ECScanManager::set_slice_to_vol_reference: cannot set shape reference for type ANY");
  std::vector<unsigned int>  indx;
  if (st==B0) indx = GetB0Indicies();
  else {
    std::vector<double> skrutt;
    std::vector<std::vector<unsigned int> >  shindx = GetShellIndicies(skrutt);
    if (si>=static_cast<int>(shindx.size())) throw EddyException("ECScanManager::set_slice_to_vol_reference: shell index out of bounds");
    indx = shindx[si];
  }
  unsigned int rindx=0;
  for (rindx=0; rindx<indx.size(); rindx++) if (indx[rindx]==ref) break;
  if (rindx==indx.size()) throw EddyException("ECScanManager::set_slice_to_vol_reference: ref index out of bounds");

  unsigned int tppv = this->MultiBand().NGroups(); 
  std::vector<NEWMAT::Matrix> Mrs(tppv);
  for (unsigned int i=0; i<tppv; i++) Mrs[i] = Scan(ref,ANY).ForwardMovementMatrix(i);
  for (unsigned int i=0; i<indx.size(); i++) {
    NEWMAT::Matrix new_mp(6,tppv);
    for (unsigned int j=0; j<tppv; j++) {
      NEWMAT::Matrix M = Scan(indx[i],ANY).ForwardMovementMatrix(j);
      new_mp.Column(j+1) = TOPUP::Matrix2MovePar(M*Mrs[j].i(),Scan(indx[i],ANY).GetIma());
    }
    Scan(indx[i],ANY).GetMovementModel().SetGroupWiseParameters(new_mp);
  }
} EddyCatch

double ECScanManager::mean_of_first_b0(const NEWIMAGE::volume4D<float>&   vols,
				       const NEWIMAGE::volume<float>&     mask,
				       const NEWMAT::Matrix&              bvecs,
				       const NEWMAT::Matrix&              bvals) const EddyTry
{
  double rval = 0.0;
  for (int s=0; s<vols.tsize(); s++) {
    EDDY::DiffPara  dp(bvecs.Column(s+1),bvals(1,s+1));
    if (EddyUtils::Isb0(dp)) {
      rval = vols[s].mean(mask);
      break;
    }
  }
  if (!rval) throw EddyException("ECScanManager::mean_of_first_b0: Zero mean");

  return(rval);
} EddyCatch

void ECScanManager::write_jac_registered_images(const std::string& fname,
						const std::string& maskfname,
						bool               mask_output, 
						ScanType           st) const EddyTry
{
  NEWIMAGE::volume4D<float> ovol(Scan(0,st).GetIma().xsize(),Scan(0,st).GetIma().ysize(),Scan(0,st).GetIma().zsize(),NScans(st));
  NEWIMAGE::copybasicproperties(Scan(0,st).GetIma(),ovol);
  NEWIMAGE::volume<float> omask = Scan(0,st).GetIma(); omask = 1.0;
  NEWIMAGE::volume<float> tmpmask = omask;
  NEWIMAGE::volume4D<float> mask_4D;
  if (maskfname.size()) {
    mask_4D.reinitialize(Scan(0,st).GetIma().xsize(),Scan(0,st).GetIma().ysize(),Scan(0,st).GetIma().zsize(),NScans(st));
    NEWIMAGE::copybasicproperties(Scan(0,st).GetIma(),mask_4D);
  }
  #ifdef COMPILE_GPU
  EddyGpuUtils::InitGpu();
  for (unsigned int s=0; s<NScans(st); s++) {
    ovol[s] = EddyGpuUtils::GetUnwarpedScan(Scan(s,st),GetSuscHzOffResField(s,st),false,this->GetPolation(),&tmpmask) / ScaleFactor();
    omask *= tmpmask;
    if (maskfname.size()) mask_4D[s] = tmpmask;
  }
  #else
  
  for (unsigned int i=0; i<NScans(st); i++) {
    NEWIMAGE::volume<float> tmp = GetUnwarpedScan(i,tmpmask,st) / ScaleFactor();
    
    {
      ovol[i] = tmp;
      omask *= tmpmask;
      if (maskfname.size()) mask_4D[i] = tmpmask;
    }
  }
  #endif
  if (mask_output) ovol *= omask;
  NEWIMAGE::write_volume(ovol,fname);
  if (maskfname.size()) NEWIMAGE::write_volume(mask_4D,maskfname);
} EddyCatch

void ECScanManager::write_jac_registered_images(const std::string&               fname, 
						const std::string&               maskfname,
						bool                             mask_output, 
						const NEWIMAGE::volume4D<float>& pred, 
						ScanType                         st) const EddyTry
{
  NEWIMAGE::volume4D<float> ovol(Scan(0,st).GetIma().xsize(),Scan(0,st).GetIma().ysize(),Scan(0,st).GetIma().zsize(),NScans(st));
  NEWIMAGE::copybasicproperties(Scan(0,st).GetIma(),ovol);
  NEWIMAGE::volume<float> omask = Scan(0,st).GetIma(); omask = 1.0;
  NEWIMAGE::volume<float> tmpmask = omask;
  NEWIMAGE::volume4D<float> mask_4D;
  if (pred.tsize() != int(NScans(st))) throw EddyException("ECScanManager::write_jac_registered_images: Size mismatch between pred and NScans");
  if (!this->IsSliceToVol()) cout << "ECScanManager::write_jac_registered_images: Warning, predictions have been supplied for a volume-to-volume registration" << endl;
  if (maskfname.size()) {
    mask_4D.reinitialize(Scan(0,st).GetIma().xsize(),Scan(0,st).GetIma().ysize(),Scan(0,st).GetIma().zsize(),NScans(st));
    NEWIMAGE::copybasicproperties(Scan(0,st).GetIma(),mask_4D);
  }
  #ifdef COMPILE_GPU
  EddyGpuUtils::InitGpu();
  if (this->IsSliceToVol()) { 
    for (unsigned int s=0; s<NScans(st); s++) {
      ovol[s] = EddyGpuUtils::GetUnwarpedScan(Scan(s,st),GetSuscHzOffResField(s,st),pred[s],false,this->GetPolation(),&tmpmask) / ScaleFactor();
      omask *= tmpmask;
      if (maskfname.size()) mask_4D[s] = tmpmask;
    }
  }
  else {
    for (unsigned int s=0; s<NScans(st); s++) {
      ovol[s] = EddyGpuUtils::GetUnwarpedScan(Scan(s,st),GetSuscHzOffResField(s,st),false,this->GetPolation(),&tmpmask) / ScaleFactor();
      omask *= tmpmask;
      if (maskfname.size()) mask_4D[s] = tmpmask;
    }
  }
  #else
  
  for (unsigned int i=0; i<NScans(st); i++) {
    NEWIMAGE::volume<float> tmp = GetUnwarpedScan(i,tmpmask,st) / ScaleFactor();
    
    {
      ovol[i] = tmp;
      omask *= tmpmask;
      if (maskfname.size()) mask_4D[i] = tmpmask;
    }
  }
  #endif
  if (mask_output) ovol *= omask;
  NEWIMAGE::write_volume(ovol,fname);
  if (maskfname.size()) NEWIMAGE::write_volume(mask_4D,maskfname);
} EddyCatch

void ECScanManager::write_lsr_registered_images(const std::string& fname, double lambda, ScanType st) const EddyTry
{
  NEWIMAGE::volume4D<float> ovol(Scan(0,st).GetIma().xsize(),Scan(0,st).GetIma().ysize(),Scan(0,st).GetIma().zsize(),NLSRPairs(st));
  NEWIMAGE::copybasicproperties(Scan(0,st).GetIma(),ovol);
  NEWIMAGE::volume<float> omask = Scan(0,st).GetIma(); omask = 1.0; 
  #ifndef COMPILE_GPU
  # pragma omp parallel for shared(ovol)
  #endif
  for (unsigned int i=0; i<NLSRPairs(st); i++) {
    std::pair<unsigned int,unsigned int> par = GetLSRPair(i,st);
    LSResampler lsr(Scan(par.first,st),Scan(par.second,st),GetSuscHzOffResField(),lambda); 
    #ifndef COMPILE_GPU
    # pragma omp critical
    #endif
    {
      ovol[i] = lsr.GetResampledVolume() / ScaleFactor();
      omask *= lsr.GetMask();
    }
  }
  ovol *= omask;
  NEWIMAGE::write_volume(ovol,fname);    
} EddyCatch

bool ECScanManager::has_pe_in_direction(unsigned int dir, ScanType st) const EddyTry
{
  if (dir != 1 && dir != 2) throw EddyException("ECScanManager::has_pe_in_direction: index out of range");
  for (unsigned int i=0; i<NScans(st); i++) {
    if (Scan(i,st).GetAcqPara().PhaseEncodeVector()(dir)) return(true);
  }
  return(false);
} EddyCatch

std::pair<int,int> ECScanManager::bracket(unsigned int                      i, 
                                          const std::vector<unsigned int>&  ii) const EddyTry
{
  std::pair<int,int> rval;
  unsigned int j;
  for (j=0; j<ii.size(); j++) if (ii[j] > i) break;
  if (!j) { rval.first=-1; rval.second=ii[0]; }
  else if (j==ii.size()) { rval.first=ii.back(); rval.second=-1; }
  else { rval.first=ii[j-1]; rval.second=ii[j]; }
  return(rval);
} EddyCatch

NEWMAT::ColumnVector ECScanManager::interp_movpar(unsigned int               i, 
                                                  const std::pair<int,int>&  br) const EddyTry
{
  NEWMAT::ColumnVector rval(6);
  if (br.first < 0) rval = Scan(br.second,EDDY::ANY).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
  else if (br.second < 0) rval = Scan(br.first,EDDY::ANY).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
  else {
    double a = double(i - br.first) / double(br.second - br.first);
    rval = (1.0 - a) * Scan(br.first,EDDY::ANY).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
    rval += a * Scan(br.second,EDDY::ANY).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
  }
  return(rval);
} EddyCatch

NEWMAT::Matrix ECScanManager::read_rb_matrix(const std::string& fname) const EddyTry
{
  NEWMAT::Matrix M;
  try {
    M = MISCMATHS::read_ascii_matrix(fname);
    if (M.Nrows() != 4 || M.Ncols() != 4) throw EddyException("ECScanManager::read_rb_matrix: matrix must be 4x4");    
    NEWMAT::Matrix ul = M.SubMatrix(1,3,1,3);
    float det = ul.Determinant();
    if (std::abs(det-1.0) > 1e-6) throw EddyException("ECScanManager::read_rb_matrix: matrix must be a rigid body transformation");    
  }
  catch (...) { throw EddyException("ECScanManager::read_rb_matrix: cannot read file"); }
  return(M);
} EddyCatch

bool ECScanManager::indicies_clustered(const std::vector<unsigned int>& indicies,
                                           unsigned int                     N) const EddyTry
{
  unsigned int even = static_cast<unsigned int>(double(N)/double(indicies.size()) + 0.5);
  for (unsigned int i=1; i<indicies.size(); i++) {
  if (double(indicies[i]-indicies[i-1])/double(even) > 1.5) return(true); 
  }
  return(false);
} EddyCatch

bool ECScanManager::has_move_by_susc_fields() const EddyTry
{
  for (unsigned int i=0; i<_susc_d1.size(); i++) { if (_susc_d1[i] != nullptr) return(true); }
  for (unsigned int i=0; i<_susc_d2.size(); i++) {
    for (unsigned int j=0; j<_susc_d2[i].size(); j++) { if (_susc_d2[i][j] != nullptr) return(true); }
  }
  return(false);
} EddyCatch

