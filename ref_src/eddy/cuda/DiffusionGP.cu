/*! \file DiffusionGP.cu
    \brief Contains definitions for class for making Gaussian process based predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Feb., 2013.
*/
// Definitions of class to make Gaussian-Process
// based predictions about diffusion data.
//
// DiffusionGP.cu
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
#include "newimage/newimageall.h"
#pragma pop
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "DiffusionGP.h"
#include "CudaVolume.h"

using namespace EDDY;

 
void DiffusionGP::predict_image_gpu(
				    unsigned int             indx,
				    bool                     exclude,
				    const NEWMAT::RowVector& pvec,
				    
				    NEWIMAGE::volume<float>& pi) const EddyTry
{
  if (!NEWIMAGE::samesize(pi,*_sptrs[0])) {
    pi.reinitialize(_sptrs[0]->xsize(),_sptrs[0]->ysize(),_sptrs[0]->zsize());
    NEWIMAGE::copybasicproperties(*_sptrs[0],pi);
  }
  EDDY::CudaVolume pcv(pi,false);
  for (unsigned int s=0; s<_sptrs.size(); s++) {
    
    
    if (exclude) {
      if (s < indx) pcv.MultiplyAndAddToMe(EDDY::CudaVolume(*(_sptrs[s])),pvec(s+1));
      
      else if (s > indx) pcv.MultiplyAndAddToMe(EDDY::CudaVolume(*(_sptrs[s])),pvec(s));
    }
    else pcv.MultiplyAndAddToMe(EDDY::CudaVolume(*(_sptrs[s])),pvec(s+1));
  }
  pcv += EDDY::CudaVolume(*_mptrs[which_mean(indx)]);
  pcv.GetVolume(pi);
  return;
} EddyCatch

void DiffusionGP::predict_images_gpu(
				     const std::vector<unsigned int>&       indicies,
				     bool                                   exclude,
				     const std::vector<NEWMAT::RowVector>&  pvecs,
				     
				     std::vector<NEWIMAGE::volume<float> >& pi) const EddyTry
{
  if (indicies.size() != pvecs.size() || indicies.size() != pi.size()) {
    throw EDDY::EddyException("DiffusionGP::predict_images_gpu: mismatch among indicies, pvecs and pi");
  }
  
  std::vector<EDDY::CudaVolume> pcvs(indicies.size());
  for (unsigned int i=0; i<indicies.size(); i++) {
    if (!NEWIMAGE::samesize(pi[i],*_sptrs[0])) {
      pi[i].reinitialize(_sptrs[0]->xsize(),_sptrs[0]->ysize(),_sptrs[0]->zsize());
      NEWIMAGE::copybasicproperties(*_sptrs[0],pi[i]);      
    }
    pcvs[i].SetHdr(pi[i]);
  }
  
  std::vector<EDDY::CudaVolume> means(_mptrs.size());
  for (unsigned int m=0; m<means.size(); m++) means[m] = *(_mptrs[m]);
  
  for (unsigned int s=0; s<_sptrs.size(); s++) { 
    EDDY::CudaVolume cv = *(_sptrs[s]);
    for (unsigned int i=0; i<indicies.size(); i++) { 
      if (exclude) {
	if (s < indicies[i]) pcvs[i].MultiplyAndAddToMe(cv,pvecs[i](s+1));
	
	else if (s > indicies[i]) pcvs[i].MultiplyAndAddToMe(cv,pvecs[i](s));
      }
      else pcvs[i].MultiplyAndAddToMe(cv,pvecs[i](s+1));
    }
  }
  
  for (unsigned int i=0; i<indicies.size(); i++) {
    pcvs[i] += means[which_mean(indicies[i])];
    pcvs[i].GetVolume(pi[i]);
  }
  return;
} EddyCatch

