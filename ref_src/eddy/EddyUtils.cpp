
// Declarations of classes that implements useful
// utility functions for the eddy current project.
// They are collections of statically declared
// functions that have been collected into classes 
// to make it explicit where they come from. There
// will never be any instances of theses classes.
// 
// EddyUtils.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#include <cstdlib>
#include <string>

#include <vector>
#include <cfloat>
#include <cmath>
#include "newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "libprob.h"
#include "warpfns/warpfns.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"

using namespace EDDY;










bool EddyUtils::get_groups(
			   const std::vector<DiffPara>&             dpv,
			   
			   std::vector<std::vector<unsigned int> >& grps,
			   std::vector<unsigned int>&               grpi,
			   std::vector<double>&                     grpb) EddyTry
{
  std::vector<unsigned int>     grp_templates;
  
  grp_templates.push_back(0);
  for (unsigned int i=1; i<dpv.size(); i++) {
    unsigned int j;
    for (j=0; j<grp_templates.size(); j++) { if (EddyUtils::AreInSameShell(dpv[grp_templates[j]],dpv[i])) break; }
    if (j == grp_templates.size()) grp_templates.push_back(i);
  }
  
  grpb.resize(grp_templates.size());
  std::vector<unsigned int>   grp_n(grp_templates.size(),1);
  for (unsigned int j=0; j<grp_templates.size(); j++) {
    grpb[j] = dpv[grp_templates[j]].bVal();
    for (unsigned int i=0; i<dpv.size(); i++) { 
      if (EddyUtils::AreInSameShell(dpv[grp_templates[j]],dpv[i]) && i!=grp_templates[j]) {
	grpb[j] += dpv[i].bVal();
	grp_n[j]++;
      }
    }
    grpb[j] /= grp_n[j];
  }
  
  std::sort(grpb.begin(),grpb.end());
  
  grpi.resize(dpv.size()); grps.resize(grpb.size());
  for (unsigned int j=0; j<grpb.size(); j++) {
    grp_n[j] = 0;
    for (unsigned int i=0; i<dpv.size(); i++) {
      if (std::abs(dpv[i].bVal()-grpb[j]) <= EddyUtils::b_range) { grpi[i] = j; grps[j].push_back(i); grp_n[j]++; }
    }
  }
  
  bool is_shelled;
  if (EddyUtils::Isb0(EDDY::DiffPara(grpb[0]))) { 
    is_shelled = grpb.size() < 7; 
    unsigned int scans_per_shell = static_cast<unsigned int>((double(dpv.size() - grp_n[0]) / double(grpb.size() - 1)) + 0.5);
    is_shelled &= bool(*std::max_element(grp_n.begin()+1,grp_n.end()) < 2 * scans_per_shell); 
    is_shelled &= bool(3 * *std::min_element(grp_n.begin()+1,grp_n.end()) > scans_per_shell); 
  }
  else { 
    is_shelled = grpb.size() < 6; 
    unsigned int scans_per_shell = static_cast<unsigned int>((double(dpv.size()) / double(grpb.size())) + 0.5);
    is_shelled &= bool(*std::max_element(grp_n.begin(),grp_n.end()) < 2 * scans_per_shell); 
    is_shelled &= bool(3 * *std::min_element(grp_n.begin(),grp_n.end()) > scans_per_shell); 
  }
  if (!is_shelled) return(false);
  
  unsigned int nscan = grps[0].size();
  for (unsigned int i=1; i<grps.size(); i++) nscan += grps[i].size();
  if (nscan != dpv.size()) throw EddyException("EddyUtils::get_groups: Inconsistent b-values detected");
  return(true);
} EddyCatch

bool EddyUtils::IsShelled(const std::vector<DiffPara>& dpv) EddyTry
{
  std::vector<std::vector<unsigned int> > grps;
  std::vector<unsigned int>               grpi;
  std::vector<double>                     grpb;
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

bool EddyUtils::IsMultiShell(const std::vector<DiffPara>& dpv) EddyTry
{
  std::vector<std::vector<unsigned int> > grps;
  std::vector<unsigned int>               grpi;
  std::vector<double>                     grpb;
  bool is_shelled = get_groups(dpv,grps,grpi,grpb);
  return(is_shelled && grpb.size() > 1);
} EddyCatch

bool EddyUtils::GetGroups(
			   const std::vector<DiffPara>&             dpv,
			   
			   std::vector<unsigned int>&               grpi,
			   std::vector<double>&                     grpb) EddyTry
{
  std::vector<std::vector<unsigned int> > grps;
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

bool EddyUtils::GetGroups(
			   const std::vector<DiffPara>&             dpv,
			   
			   std::vector<std::vector<unsigned int> >& grps,
			   std::vector<double>&                     grpb) EddyTry
{
  std::vector<unsigned int> grpi;
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

bool EddyUtils::GetGroups(
			   const std::vector<DiffPara>&             dpv,
			   
			   std::vector<std::vector<unsigned int> >& grps,
			   std::vector<unsigned int>&               grpi,
			   std::vector<double>&                     grpb) EddyTry
{
  return(get_groups(dpv,grps,grpi,grpb));
} EddyCatch

std::vector<unsigned int> EddyUtils::GetIndiciesOfDWIs(const std::vector<DiffPara>& dpars) EddyTry
{
  std::vector<unsigned int> indicies;
  for (unsigned int i=0; i<dpars.size(); i++) { if (EddyUtils::IsDiffusionWeighted(dpars[i])) indicies.push_back(i); }
  return(indicies);
} EddyCatch

std::vector<DiffPara> EddyUtils::GetDWIDiffParas(const std::vector<DiffPara>&   dpars) EddyTry
{
  std::vector<unsigned int> indx = EddyUtils::GetIndiciesOfDWIs(dpars);
  std::vector<DiffPara> dwi_dpars;
  for (unsigned int i=0; i<indx.size(); i++) dwi_dpars.push_back(dpars[indx[i]]);
  return(dwi_dpars);
} EddyCatch

bool EddyUtils::AreMatchingPair(const ECScan& s1, const ECScan& s2) EddyTry
{
  double dp = NEWMAT::DotProduct(s1.GetAcqPara().PhaseEncodeVector(),s2.GetAcqPara().PhaseEncodeVector());
  if (std::abs(dp + 1.0) > 1e-6) return(false);
  if (!EddyUtils::AreInSameShell(s1.GetDiffPara(),s2.GetDiffPara())) return(false);
  if (IsDiffusionWeighted(s1.GetDiffPara()) && !HaveSameDirection(s1.GetDiffPara(),s2.GetDiffPara())) return(false);
  return(true);
} EddyCatch

std::vector<NEWMAT::Matrix> EddyUtils::GetSliceWiseForwardMovementMatrices(const EDDY::ECScan&           scan) EddyTry
{
  std::vector<NEWMAT::Matrix> R(scan.GetIma().zsize());
  for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) {
    NEWMAT::Matrix tR = scan.ForwardMovementMatrix(tp);
    std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
    for (unsigned int i=0; i<slices.size(); i++) R[slices[i]] = tR;
  }
  return(R);
} EddyCatch

std::vector<NEWMAT::Matrix> EddyUtils::GetSliceWiseInverseMovementMatrices(const EDDY::ECScan&           scan) EddyTry
{
  std::vector<NEWMAT::Matrix> R(scan.GetIma().zsize());
  for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) {
    NEWMAT::Matrix tR = scan.InverseMovementMatrix(tp); 
    std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
    for (unsigned int i=0; i<slices.size(); i++) R[slices[i]] = tR;
  }
  return(R);
} EddyCatch

int EddyUtils::read_DWI_volume4D(NEWIMAGE::volume4D<float>&     dwivols,
				 const std::string&             fname,
				 const std::vector<DiffPara>&   dpars) EddyTry
{
  std::vector<unsigned int> indx = EddyUtils::GetIndiciesOfDWIs(dpars);
  NEWIMAGE::volume<float> tmp;
  read_volumeROI(tmp,fname,0,0,0,0,-1,-1,-1,0);
  dwivols.reinitialize(tmp.xsize(),tmp.ysize(),tmp.zsize(),indx.size());
  for (unsigned int i=0; i<indx.size(); i++) {
    read_volumeROI(tmp,fname,0,0,0,indx[i],-1,-1,-1,indx[i]);
    dwivols[i] = tmp;
  }
  return(1);
} EddyCatch

NEWIMAGE::volume<float> EddyUtils::ConvertMaskToFloat(const NEWIMAGE::volume<char>& charmask) EddyTry
{
  NEWIMAGE::volume<float> floatmask(charmask.xsize(),charmask.ysize(),charmask.zsize());
  NEWIMAGE::copybasicproperties(charmask,floatmask);
  for (int k=0; k<charmask.zsize(); k++) {
    for (int j=0; j<charmask.ysize(); j++) {
      for (int i=0; i<charmask.xsize(); i++) {
	floatmask(i,j,k) = static_cast<float>(charmask(i,j,k));
      }
    }
  }
  return(floatmask);
} EddyCatch


NEWIMAGE::volume<float> EddyUtils::Smooth(const NEWIMAGE::volume<float>& ima, float fwhm, const NEWIMAGE::volume<float>& mask) EddyTry
{
  if (mask.getextrapolationmethod() != NEWIMAGE::zeropad) throw EddyException("EddyUtils::Smooth: mask must use zeropad for extrapolation");
  float sx = (fwhm/std::sqrt(8.0*std::log(2.0)))/ima.xdim();
  float sy = (fwhm/std::sqrt(8.0*std::log(2.0)))/ima.ydim();
  float sz = (fwhm/std::sqrt(8.0*std::log(2.0)))/ima.zdim();
  int nx=((int) (sx-0.001))*2 + 3;
  int ny=((int) (sy-0.001))*2 + 3;
  int nz=((int) (sz-0.001))*2 + 3;
  NEWMAT::ColumnVector krnlx = NEWIMAGE::gaussian_kernel1D(sx,nx);
  NEWMAT::ColumnVector krnly = NEWIMAGE::gaussian_kernel1D(sy,ny);
  NEWMAT::ColumnVector krnlz = NEWIMAGE::gaussian_kernel1D(sz,nz);
  NEWIMAGE::volume4D<float> ovol = ima; 
  for (int i=0; i<ima.tsize(); i++) {
    ovol[i] = NEWIMAGE::convolve_separable(ima[i],krnlx,krnly,krnlz,mask)*mask;
  }
  return(ovol);  
} EddyCatch




NEWIMAGE::volume<float> EddyUtils::MakeNoiseIma(const NEWIMAGE::volume<float>& ima, float mu, float stdev) EddyTry
{
  NEWIMAGE::volume<float>  nima = ima;
  double rnd;
  for (int k=0; k<nima.zsize(); k++) {
    for (int j=0; j<nima.ysize(); j++) {
      for (int i=0; i<nima.xsize(); i++) {
	drand(&rnd);
	nima(i,j,k) = mu + stdev*static_cast<float>(ndtri(rnd-1));
      }
    }
  }
  return(nima);
} EddyCatch

DiffStats EddyUtils::GetSliceWiseStats(
				       const NEWIMAGE::volume<float>&                  pred,          
				       std::shared_ptr<const NEWIMAGE::volume<float> > susc,          
				       const NEWIMAGE::volume<float>&                  pmask,         
				       const NEWIMAGE::volume<float>&                  bmask,         
				       const EDDY::ECScan&                             scan) EddyTry  
{
  
  NEWIMAGE::volume<float> pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
  
  NEWIMAGE::volume<float> mask = pred; mask = 0.0;
  NEWIMAGE::volume<float> bios = EddyUtils::transform_model_to_scan_space(pmask*bmask,scan,susc,false,mask,NULL,NULL);
  bios.binarise(0.99); 
  mask *= bios; 
  
  DiffStats stats(scan.GetOriginalIma()-pios,mask);
  return(stats);
} EddyCatch

double EddyUtils::param_update(
			       const NEWIMAGE::volume<float>&                    pred,      
			       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
			       const NEWIMAGE::volume<float>&                    pmask,     
			       Parameters                                        whichp,    
			       bool                                              cbs,       
			       float                                             fwhm,      
			       
			       unsigned int                                      scindx,    
			       unsigned int                                      iter,      
			       unsigned int                                      level,     
			       
			       EDDY::ECScan&                                     scan,      
			       
			       NEWMAT::ColumnVector                              *rupdate)  
EddyTry
{
  
  NEWIMAGE::volume<float> mask = pred; mask.setextrapolationmethod(NEWIMAGE::zeropad); mask = 0.0;
  NEWIMAGE::volume<float> jac = pred; jac = 1.0;
  
  NEWIMAGE::volume<float> pios = EddyUtils::transform_model_to_scan_space(pred,scan,susc,true,mask,&jac,NULL);
  
  NEWIMAGE::volume<float> skrutt = pred; skrutt = 0.0;
  NEWIMAGE::volume<float> mios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,skrutt,NULL,NULL);
  mios.binarise(0.99); 
  mask *= mios; 
  
  NEWIMAGE::volume4D<float> derivs = EddyUtils::get_partial_derivatives_in_scan_space(pred,scan,susc,whichp);
  if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); derivs = EddyUtils::Smooth(derivs,fwhm,mask); }
  
  NEWMAT::Matrix XtX = EddyUtils::make_XtX(derivs,mask);
  
  NEWIMAGE::volume<float> dima = pios-scan.GetIma();
  if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); dima = EddyUtils::Smooth(dima,fwhm,mask); }
  
  NEWMAT::ColumnVector Xty = EddyUtils::make_Xty(derivs,dima,mask);
  
  double mss = (dima*mask).sumsquares() / mask.sum();
  
  NEWMAT::IdentityMatrix eye(XtX.Nrows());
  double lambda = 1.0;
  
  NEWMAT::ColumnVector update = -(XtX+lambda*eye).i()*Xty;
  NEWIMAGE::volume<float> sims;
  if (level) sims = scan.GetUnwarpedIma(susc);
  
  for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
    scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)+update(i+1),whichp);
  }
  if (!level) { 
    if (cbs) {
      pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
      
      mask = 0.0;
      mios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,mask,NULL,NULL);
      mios.binarise(0.99); 
      mask *= mios; 
      dima = pios-scan.GetIma();
      if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); dima = EddyUtils::Smooth(dima,fwhm,mask); }
      double mss_au = ((dima*mask).sumsquares()) / mask.sum();
      if (mss_au > mss) { 
	for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
	  scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)-update(i+1),whichp);
	}
      }
    }
    if (rupdate) *rupdate = update;

    return(mss);

  
  }
  else { 
    NEWIMAGE::volume<float> new_pios;
    NEWIMAGE::volume<float> new_mios;
    NEWIMAGE::volume<float> new_mask;
    NEWIMAGE::volume<float> new_dima;
    if (cbs) {
      new_pios = EddyUtils::TransformModelToScanSpace(pred,scan,susc);
      
      new_mask = new_pios; new_mask = 0.0;
      new_mios = EddyUtils::transform_model_to_scan_space(pmask,scan,susc,false,new_mask,NULL,NULL);
      new_mios.binarise(0.99); 
      new_mask *= new_mios; 
      new_dima = new_pios-scan.GetIma();
      if (fwhm) { mask.setextrapolationmethod(NEWIMAGE::zeropad); new_dima = EddyUtils::Smooth(new_dima,fwhm,mask); }
      double mss_au = new_dima.sumsquares() / new_mask.sum();
      if (mss_au > mss) { 
	for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
	  scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)-update(i+1),whichp);
	}
      }
    }
    if (rupdate) *rupdate = update;

    char fname[256];
    if (level>0) {
      sprintf(fname,"EDDY_DEBUG_masked_dima_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(dima*mask,fname);
      sprintf(fname,"EDDY_DEBUG_reverse_dima_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(pred-sims,fname);
    }
    if (level>1) {
      sprintf(fname,"EDDY_DEBUG_mask_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(mask,fname);
      sprintf(fname,"EDDY_DEBUG_pios_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(pios,fname);
      sprintf(fname,"EDDY_DEBUG_pred_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(pred,fname);
      sprintf(fname,"EDDY_DEBUG_dima_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(dima,fname);      
      sprintf(fname,"EDDY_DEBUG_jac_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(jac,fname);      
      sprintf(fname,"EDDY_DEBUG_orig_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(scan.GetIma(),fname);
      if (cbs) {
	sprintf(fname,"EDDY_DEBUG_new_masked_dima_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(new_dima*new_mask,fname);
	sprintf(fname,"EDDY_DEBUG_new_reverse_dima_%02d_%04d",iter,scindx);
	sims = scan.GetUnwarpedIma(susc); NEWIMAGE::write_volume(pred-sims,fname);
	sprintf(fname,"EDDY_DEBUG_new_mask_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(new_mask,fname);
	sprintf(fname,"EDDY_DEBUG_new_pios_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(new_pios,fname);
	sprintf(fname,"EDDY_DEBUG_new_dima_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(new_dima,fname);
      }
    }
    if (level>2) {
      sprintf(fname,"EDDY_DEBUG_mios_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(mios,fname);
      sprintf(fname,"EDDY_DEBUG_pmask_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(pmask,fname);
      sprintf(fname,"EDDY_DEBUG_derivs_%02d_%04d",iter,scindx); NEWIMAGE::write_volume(derivs,fname);
      sprintf(fname,"EDDY_DEBUG_XtX_%02d_%04d.txt",iter,scindx); MISCMATHS::write_ascii_matrix(fname,XtX);
      sprintf(fname,"EDDY_DEBUG_Xty_%02d_%04d.txt",iter,scindx); MISCMATHS::write_ascii_matrix(fname,Xty);
      sprintf(fname,"EDDY_DEBUG_update_%02d_%04d.txt",iter,scindx); MISCMATHS::write_ascii_matrix(fname,update);
    }
  } 

  return(mss);
} EddyCatch



NEWIMAGE::volume<float> EddyUtils::transform_model_to_scan_space(
								 const NEWIMAGE::volume<float>&                    pred,
								 const EDDY::ECScan&                               scan,
								 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
								 bool                                              jacmod,
								 
								 NEWIMAGE::volume<float>&                          omask,
								 NEWIMAGE::volume<float>                           *jac,
								 NEWIMAGE::volume4D<float>                         *grad)
{
  
  if (jacmod && !jac) throw EddyException("EddyUtils::transform_model_to_scan_space: jacmod can only be used with valid jac");
  NEWIMAGE::volume4D<float> dfield;
  if (jacmod || jac) dfield = scan.FieldForModelToScanTransform(susc,omask,*jac);
  else dfield = scan.FieldForModelToScanTransform(susc,omask);
  NEWMAT::Matrix eye(4,4); eye=0; eye(1,1)=1.0; eye(2,2)=1.0; eye(3,3)=1.0; eye(4,4)=1.0;
  NEWIMAGE::volume<float> ovol = pred; ovol = 0.0;
  NEWIMAGE::volume<char> mask3(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::copybasicproperties(pred,mask3); mask3 = 0;
  std::vector<int> ddir(3); ddir[0] = 0; ddir[1] = 1; ddir[2] = 2;
  if (scan.IsSliceToVol()) {
    if (grad) {
      grad->reinitialize(pred.xsize(),pred.ysize(),pred.zsize(),3); 
      NEWIMAGE::copybasicproperties(pred,*grad);
    }
    for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) { 
      NEWMAT::Matrix R = scan.ForwardMovementMatrix(tp);
      std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
      if (grad) NEWIMAGE::raw_general_transform(pred,eye,dfield,ddir,ddir,slices,&eye,&R,ovol,*grad,&mask3);
      else NEWIMAGE::apply_warp(pred,eye,dfield,eye,R,slices,ovol,mask3);
    }
  }
  else {
    std::vector<unsigned int> all_slices;
    
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    
    if (grad) {
      grad->reinitialize(pred.xsize(),pred.ysize(),pred.zsize(),3); 
      NEWIMAGE::copybasicproperties(pred,*grad);
      NEWIMAGE::raw_general_transform(pred,eye,dfield,ddir,ddir,all_slices,&eye,&R,ovol,*grad,&mask3);
    }
    else NEWIMAGE::apply_warp(pred,eye,dfield,eye,R,ovol,mask3);
  }
  omask *= EddyUtils::ConvertMaskToFloat(mask3); 
  EddyUtils::SetTrilinearInterp(omask);
  if (jacmod) ovol *= *jac;                      
  return(ovol);
}



EDDY::ImageCoordinates EddyUtils::transform_coordinates_from_model_to_scan_space(
										 const NEWIMAGE::volume<float>&                    pred,
										 const EDDY::ECScan&                               scan,
										 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
										 
										 NEWIMAGE::volume<float>                           *omask,
										 NEWIMAGE::volume<float>                           *jac)
{
  
  NEWIMAGE::volume4D<float> dfield;
  if (omask && jac) dfield = scan.FieldForModelToScanTransform(susc,*omask,*jac);
  else if (omask) dfield = scan.FieldForModelToScanTransform(susc,*omask);
  else if (jac) dfield = scan.FieldForModelToScanTransformWithJac(susc,*jac);
  else dfield = scan.FieldForModelToScanTransform(susc);

  ImageCoordinates coord(pred.xsize(),pred.ysize(),pred.zsize()); 
  if (scan.IsSliceToVol()) {
    for (unsigned int tp=0; tp<scan.GetMBG().NGroups(); tp++) { 
      NEWMAT::Matrix R = scan.ForwardMovementMatrix(tp);
      std::vector<unsigned int> slices = scan.GetMBG().SlicesAtTimePoint(tp);
      
      if (omask) { 
	NEWIMAGE::volume<float> mask2(pred.xsize(),pred.ysize(),pred.zsize());
	NEWIMAGE::copybasicproperties(pred,mask2); mask2 = 0;
	EddyUtils::transform_coordinates(pred,dfield,R,slices,coord,&mask2);
	*omask *= mask2;
	EddyUtils::SetTrilinearInterp(*omask);
      }
      else EddyUtils::transform_coordinates(pred,dfield,R,slices,coord,NULL);
    }
  }
  else {
  
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    std::vector<unsigned int> all_slices;
    
    if (omask) { 
      NEWIMAGE::volume<float> mask2(pred.xsize(),pred.ysize(),pred.zsize());
      NEWIMAGE::copybasicproperties(pred,mask2); mask2 = 0;
      EddyUtils::transform_coordinates(pred,dfield,R,all_slices,coord,&mask2);
      *omask *= mask2;
      EddyUtils::SetTrilinearInterp(*omask);
    }
    else EddyUtils::transform_coordinates(pred,dfield,R,all_slices,coord,NULL);
  }

  return(coord);
}


NEWIMAGE::volume4D<float> EddyUtils::get_partial_derivatives_in_scan_space(
									   const NEWIMAGE::volume<float>&                    pred,      
									   const EDDY::ECScan&                               scan,      
									   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
									   EDDY::Parameters                                  whichp)
{
  NEWIMAGE::volume<float> basejac;
  NEWIMAGE::volume4D<float> grad;
  NEWIMAGE::volume<float> skrutt(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> base = transform_model_to_scan_space(pred,scan,susc,true,skrutt,&basejac,&grad);
  ImageCoordinates basecoord = transform_coordinates_from_model_to_scan_space(pred,scan,susc,NULL,NULL);
  NEWIMAGE::volume4D<float> derivs(base.xsize(),base.ysize(),base.zsize(),scan.NDerivs(whichp));
  NEWIMAGE::volume<float> jac = pred;
  ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    ImageCoordinates diff = transform_coordinates_from_model_to_scan_space(pred,sc,susc,NULL,&jac) - basecoord;
    derivs[i] = (diff*grad) / sc.GetDerivScale(i,whichp);
    derivs[i] += base * (jac-basejac) / sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp); 
  }
  return(derivs);
}

NEWIMAGE::volume4D<float> EddyUtils::get_direct_partial_derivatives_in_scan_space(
										  const NEWIMAGE::volume<float>&                    pred,     
										  const EDDY::ECScan&                               scan,     
										  std::shared_ptr<const NEWIMAGE::volume<float> >   susc,     
										  EDDY::Parameters                                  whichp)
{
  NEWIMAGE::volume<float> jac(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> skrutt(pred.xsize(),pred.ysize(),pred.zsize());
  NEWIMAGE::volume<float> base = transform_model_to_scan_space(pred,scan,susc,true,skrutt,&jac,NULL);
  NEWIMAGE::volume4D<float> derivs(base.xsize(),base.ysize(),base.zsize(),scan.NDerivs(whichp));
  ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    NEWIMAGE::volume<float> perturbed = transform_model_to_scan_space(pred,sc,susc,true,skrutt,&jac,NULL);
    derivs[i] = (perturbed-base) / sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp); 
  }
  return(derivs);
}




NEWIMAGE::volume<float> EddyUtils::DirectTransformScanToModelSpace(
								   const EDDY::ECScan&                             scan,
								   std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								   
								   NEWIMAGE::volume<float>&                        omask)
{
  NEWIMAGE::volume<float> ima = scan.GetIma();
  NEWIMAGE::volume<float> eb = scan.ECField();
  NEWIMAGE::volume4D<float> dfield = FieldUtils::Hz2VoxelDisplacements(eb,scan.GetAcqPara());
  dfield = FieldUtils::Voxel2MMDisplacements(dfield);
  NEWMAT::Matrix eye(4,4); eye=0; eye(1,1)=1.0; eye(2,2)=1.0; eye(3,3)=1.0; eye(4,4)=1.0;
  NEWIMAGE::volume<float> ovol = ima; ovol = 0.0;
  NEWIMAGE::volume<char> mask(ima.xsize(),ima.ysize(),ima.zsize());
  NEWIMAGE::apply_warp(ima,eye,dfield,eye,eye,ovol,mask);

  NEWMAT::Matrix iR = scan.InverseMovementMatrix();
  NEWIMAGE::volume<float> tmp = ovol; ovol = 0;
  NEWIMAGE::affine_transform(tmp,iR,ovol,mask);
  
  return(ovol);
}




NEWMAT::Matrix EddyUtils::make_XtX(const NEWIMAGE::volume4D<float>& vols,
				   const NEWIMAGE::volume<float>&   mask)
{
  NEWMAT::Matrix XtX(vols.tsize(),vols.tsize());
  XtX = 0.0;
  for (int r=1; r<=vols.tsize(); r++) {
    for (int c=r; c<=vols.tsize(); c++) {
      for (NEWIMAGE::volume<float>::fast_const_iterator rit=vols.fbegin(r-1), ritend=vols.fend(r-1), cit=vols.fbegin(c-1), mit=mask.fbegin(); rit!=ritend; ++rit, ++cit, ++mit) {
	if (*mit) XtX(r,c) += (*rit)*(*cit);
      }
    }
  }
  for (int r=2; r<=vols.tsize(); r++) {
    for (int c=1; c<r; c++) XtX(r,c) = XtX(c,r);
  }
  return(XtX);
}

NEWMAT::ColumnVector EddyUtils::make_Xty(const NEWIMAGE::volume4D<float>& Xvols,
					 const NEWIMAGE::volume<float>&   Yvol,
					 const NEWIMAGE::volume<float>&   mask)
{
  NEWMAT::ColumnVector Xty(Xvols.tsize());
  Xty = 0.0;
  for (int r=1; r<=Xvols.tsize(); r++) {
    for (NEWIMAGE::volume<float>::fast_const_iterator Xit=Xvols.fbegin(r-1), Xend=Xvols.fend(r-1), Yit=Yvol.fbegin(), mit=mask.fbegin(); Xit!=Xend; ++Xit, ++Yit, ++mit) {
      if (*mit) Xty(r) += (*Xit)*(*Yit);
    }
  }
  return(Xty);
}



void EddyUtils::transform_coordinates(
				      const NEWIMAGE::volume<float>&    f,
				      const NEWIMAGE::volume4D<float>&  d,
				      const NEWMAT::Matrix&             M,
				      std::vector<unsigned int>         slices,
				      
				      ImageCoordinates&                 c,
                                      
				      NEWIMAGE::volume<float>           *omask)
{
  NEWMAT::Matrix iA = d[0].sampling_mat();

  float A11=iA(1,1), A12=iA(1,2), A13=iA(1,3), A14=iA(1,4);
  float A21=iA(2,1), A22=iA(2,2), A23=iA(2,3), A24=iA(2,4);
  float A31=iA(3,1), A32=iA(3,2), A33=iA(3,3), A34=iA(3,4); 

  
  
  

  NEWMAT::Matrix iM = f.sampling_mat().i() * M.i();

  float M11=iM(1,1), M12=iM(1,2), M13=iM(1,3), M14=iM(1,4);
  float M21=iM(2,1), M22=iM(2,2), M23=iM(2,3), M24=iM(2,4);
  float M31=iM(3,1), M32=iM(3,2), M33=iM(3,3), M34=iM(3,4); 

  
  if (slices.size() == 0) { slices.resize(c.NZ()); for (unsigned int z=0; z<c.NZ(); z++) slices[z] = z; }
  else if (slices.size() > c.NZ()) throw EddyException("EddyUtils::transform_coordinates: slices vector too long");
  else { for (unsigned int z=0; z<slices.size(); z++) if (slices[z] >= c.NZ()) throw EddyException("EddyUtils::transform_coordinates: slices vector has invalid entry");}

  for (unsigned int k=0; k<slices.size(); k++) {
    unsigned int z = slices[k];
    unsigned int index = k * c.NY() * c.NX();
    float xtmp1 = A13*z + A14;
    float ytmp1 = A23*z + A24;
    float ztmp1 = A33*z + A34;
    for (unsigned int y=0; y<c.NY(); y++) {
      float xtmp2 = xtmp1 + A12*y;
      float ytmp2 = ytmp1 + A22*y;
      float ztmp2 = ztmp1 + A32*y;
      for (unsigned int x=0; x<c.NX(); x++) {
	float o1 = xtmp2 + A11*x + d(x,y,z,0);
	float o2 = ytmp2 + A21*x + d(x,y,z,1);
	float o3 = ztmp2 + A31*x + d(x,y,z,2);
	if (omask) (*omask)(x,y,z) = 1;  
	c.x(index) = M11*o1 + M12*o2 + M13*o3 + M14;
	c.y(index) = M21*o1 + M22*o2 + M23*o3 + M24;
	c.z(index) = M31*o1 + M32*o2 + M33*o3 + M34;
	if (omask) (*omask)(x,y,z) *= (f.valid(c.x(index),c.y(index),c.z(index))) ? 1 : 0; 
        index++;
      }
    }
  }
  return;
}










NEWIMAGE::volume4D<float> FieldUtils::Hz2VoxelDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                            const AcqPara&                 acqp)
{
  NEWIMAGE::volume4D<float> dfield(hzfield.xsize(),hzfield.ysize(),hzfield.zsize(),3);
  NEWIMAGE::copybasicproperties(hzfield,dfield);
  for (int i=0; i<3; i++) dfield[i] = float((acqp.PhaseEncodeVector())(i+1) * acqp.ReadOutTime()) * hzfield;
  return(dfield);
}

NEWIMAGE::volume4D<float> FieldUtils::Hz2MMDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                         const AcqPara&                 acqp)
{
  NEWIMAGE::volume4D<float> dfield(hzfield.xsize(),hzfield.ysize(),hzfield.zsize(),3);
  NEWIMAGE::copybasicproperties(hzfield,dfield);
  dfield[0] = float(hzfield.xdim()*(acqp.PhaseEncodeVector())(1) * acqp.ReadOutTime()) * hzfield;
  dfield[1] = float(hzfield.ydim()*(acqp.PhaseEncodeVector())(2) * acqp.ReadOutTime()) * hzfield;
  dfield[2] = float(hzfield.zdim()*(acqp.PhaseEncodeVector())(3) * acqp.ReadOutTime()) * hzfield;
  return(dfield);
}







NEWIMAGE::volume<float> FieldUtils::Invert1DDisplacementField(
							      const NEWIMAGE::volume<float>& dfield,
							      const AcqPara&                 acqp,
							      const NEWIMAGE::volume<float>& inmask,
							      
							      NEWIMAGE::volume<float>&       omask)
{
  NEWIMAGE::volume<float> fc = dfield;   
  NEWIMAGE::volume<float> imc = inmask;  
  
  unsigned int d=0;
  for (; d<3; d++) if ((acqp.PhaseEncodeVector())(d+1)) break;
  if (d==1) {
    fc.swapdimensions(2,1,3);
    imc.swapdimensions(2,1,3);
    omask.swapdimensions(2,1,3);
  }
  else if (d==2) {
    fc.swapdimensions(3,2,1);
    imc.swapdimensions(3,2,1);
    omask.swapdimensions(3,2,1);
  }
  NEWIMAGE::volume<float> idf = fc;    
  
  for (int k=0; k<idf.zsize(); k++) {
    for (int j=0; j<idf.ysize(); j++) {
      int oi=0;
      for (int i=0; i<idf.xsize(); i++) {
	int ii=oi;
	for (; ii<idf.xsize() && fc(ii,j,k)+ii<i; ii++) ; 
	if (ii>0 && ii<idf.xsize()) { 
	  idf(i,j,k) = ii - i - 1.0 + float(i+1-ii-fc(ii-1,j,k))/float(fc(ii,j,k)+1.0-fc(ii-1,j,k));
          if (imc(ii-1,j,k)) omask(i,j,k) = 1.0;
	  else omask(i,j,k) = 0.0;
	}
	else {
	  idf(i,j,k) = FLT_MAX;    
	  omask(i,j,k) = 0.0;
	}
	oi = std::max(0,ii-1);
      }
      
      int ii=0;
      for (ii=0; ii<idf.xsize()-1 && idf(ii,j,k)==FLT_MAX; ii++) ; 
      for (; ii>0; ii--) idf(ii-1,j,k) = idf(ii,j,k); 
      
      for (ii=idf.xsize()-1; ii>0 && idf(ii,j,k)==FLT_MAX; ii--) ; 
      for (; ii<idf.xsize()-1; ii++) idf(ii+1,j,k) = idf(ii,j,k); 
    }
  }
  
  if (d==1) {
    idf.swapdimensions(2,1,3);
    omask.swapdimensions(2,1,3);
  }
  else if (d==2) {
    idf.swapdimensions(3,2,1);
    omask.swapdimensions(3,2,1);
  }

  return(idf);
}








NEWIMAGE::volume4D<float> FieldUtils::Invert3DDisplacementField(
								const NEWIMAGE::volume4D<float>& dfield,
								const AcqPara&                   acqp,
								const NEWIMAGE::volume<float>&   inmask,
								
								NEWIMAGE::volume<float>&         omask)
{
  NEWIMAGE::volume4D<float> idfield = dfield;
  idfield = 0.0;
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EddyException("FieldUtils::InvertDisplacementField: Phase encode vector must have exactly one non-zero component");
  unsigned int i=0;
  for (; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) break;
  idfield[i] = Invert1DDisplacementField(dfield[i],acqp,inmask,omask);

  return(idfield);
}








NEWIMAGE::volume<float> FieldUtils::GetJacobian(const NEWIMAGE::volume4D<float>& dfield,
                                                const AcqPara&                   acqp)
{
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EddyException("FieldUtils::GetJacobian: Phase encode vector must have exactly one non-zero component");
  unsigned int i=0;
  for (; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) break;

  NEWIMAGE::volume<float> jacfield = GetJacobianFrom1DField(dfield[i],i);

  return(jacfield);  
}







NEWIMAGE::volume<float> FieldUtils::GetJacobianFrom1DField(const NEWIMAGE::volume<float>& dfield,
                                                           unsigned int                   dir)
{
  
  std::vector<unsigned int>                        dim(3,0);
  dim[0] = dfield.xsize(); dim[1] = dfield.ysize(); dim[2] = dfield.zsize();
  std::vector<SPLINTERPOLATOR::ExtrapolationType>  ep(3,SPLINTERPOLATOR::Mirror);
  SPLINTERPOLATOR::Splinterpolator<float> spc(dfield.fbegin(),dim,ep,3,false);
  
  NEWIMAGE::volume<float> jacf = dfield;
  for (int k=0; k<dfield.zsize(); k++) {
    for (int j=0; j<dfield.ysize(); j++) {
      for (int i=0; i<dfield.xsize(); i++) {
        jacf(i,j,k) = 1.0 + spc.DerivXYZ(i,j,k,dir);
      }
    }
  }
  return(jacf);
}

 
void s2vQuant::common_construction()
{
  if (!_sm.Scan(0,ANY).IsSliceToVol()) throw EddyException("s2vQuant::common_construction: Data is not slice-to-vol");;

  std::vector<unsigned int> icsl;
  if (_sm.MultiBand().MBFactor() == 1) icsl = _sm.IntraCerebralSlices(500); 
  _tr.ReSize(3,_sm.NScans(ANY));
  _rot.ReSize(3,_sm.NScans(ANY));
  for (unsigned int i=0; i<_sm.NScans(ANY); i++) {
    for (unsigned int j=0; j<3; j++) {
      _tr(j+1,i+1) = _sm.Scan(i,ANY).GetMovementStd(j,icsl);
      _rot(j+1,i+1) = 180.0 * _sm.Scan(i,ANY).GetMovementStd(3+j,icsl) / 3.141592653589793;
    }
  }
}

 
std::vector<unsigned int> s2vQuant::FindStillVolumes(ScanType                         st,
						     const std::vector<unsigned int>& mbsp) const
{
  std::vector<unsigned int> rval;
  for (unsigned int i=0; i<_sm.NScans(st); i++) {
    unsigned int j = i;
    if (st==B0) j = _sm.Getb02GlobalIndexMapping(i);
    else if (st==DWI) j = _sm.GetDwi2GlobalIndexMapping(i);
    bool is_still = true;
    for (unsigned int pi=0; pi<mbsp.size(); pi++) {
      if (mbsp[pi] < 3) {
	NEWMAT::ColumnVector tmp = _tr.Column(j+1);
	if (tmp(mbsp[pi]+1) > _trth) is_still = false;
      }
      else if (mbsp[pi] > 2 && mbsp[pi] < 6) {
	NEWMAT::ColumnVector tmp = _rot.Column(j+1);
	if (tmp(mbsp[pi]+1-3) > _rotth) is_still = false;
      }
      else throw EddyException("s2vQuant::FindStillVolumes: mbsp out of range");
    }
    if (is_still) rval.push_back(i);
  }
  return(rval);
}

