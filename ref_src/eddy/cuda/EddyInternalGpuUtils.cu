/////////////////////////////////////////////////////////////////////
///
/// \file EddyInternalGpuUtils.cu
/// \brief Definitions of static class with collection of GPU routines used in the eddy project
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///


#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <sys/time.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#pragma diag_suppress = expr_has_no_effect  
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyUtils.h"
#include "CudaVolume.h"
#include "EddyKernels.h"
#include "EddyFunctors.h"
#include "EddyInternalGpuUtils.h"
#include "GpuPredictorChunk.h"
#include "StackResampler.h"

using namespace EDDY;

void EddyInternalGpuUtils::load_prediction_maker(
						 const EddyCommandLineOptions&        clo,
						 ScanType                             st,
						 const ECScanManager&                 sm,
						 unsigned int                         iter,
						 float                                fwhm,
						 bool                                 use_orig,
						 const EDDY::PolationPara&            pp,
						 
						 std::shared_ptr<DWIPredictionMaker>  pmp,
						 NEWIMAGE::volume<float>&             mask) EddyTry
{
  if (sm.NScans(st)) {
    EDDY::CudaVolume omask(sm.Scan(0,st).GetIma(),false);
    omask.SetInterp(NEWIMAGE::trilinear); omask = 1.0; 
    EDDY::CudaVolume tmpmask(omask,false);    
    EDDY::CudaVolume uwscan;
    EDDY::CudaVolume empty;

    if (clo.Verbose()) cout << "Loading prediction maker";
    if (clo.VeryVerbose()) cout << endl << "Scan: ";
    for (int s=0; s<int(sm.NScans(st)); s++) {
      if (clo.VeryVerbose()) { cout << " " << s; cout.flush(); }
      EDDY::CudaVolume susc;
      if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(s,st));      
      EddyInternalGpuUtils::get_unwarped_scan(sm.Scan(s,st),susc,empty,true,use_orig,pp,uwscan,tmpmask);
      pmp->SetScan(uwscan.GetVolume(),sm.Scan(s,st).GetDiffPara(clo.RotateBVecsDuringEstimation()),s);
      omask *= tmpmask;
    }
    mask = omask.GetVolume();

    if (clo.Verbose()) cout << endl << "Evaluating prediction maker model" << endl;
    pmp->EvaluateModel(sm.Mask()*mask,fwhm,clo.VeryVerbose());
  }
} EddyCatch



void EddyInternalGpuUtils::get_motion_corrected_scan(
						     const EDDY::ECScan&     scan,
						     bool                    use_orig,
						     
						     EDDY::CudaVolume&       oima,
						     
						     EDDY::CudaVolume&       omask) EddyTry
{
  if (!oima.Size()) oima.SetHdr(scan.GetIma());
  else if (oima != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_motion_corrected_scan: scan<->oima mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_motion_corrected_scan: scan<->omask mismatch");
  EDDY::CudaVolume ima;
  EDDY::CudaVolume4D skrutt;
  if (use_orig) ima = scan.GetOriginalIma();
  else ima = scan.GetIma();
  if (scan.IsSliceToVol()) {
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    EddyInternalGpuUtils::affine_transform(ima,iR,oima,skrutt,omask);
  }
  else {
    
    NEWMAT::Matrix iR = scan.InverseMovementMatrix();
    if (omask.Size()) omask = 1.0;
    EddyInternalGpuUtils::affine_transform(ima,iR,oima,skrutt,omask);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::get_unwarped_scan(
					     const EDDY::ECScan&        scan,
					     const EDDY::CudaVolume&    susc,
					     const EDDY::CudaVolume&    pred,
					     bool                       jacmod,
					     bool                       use_orig,
					     const EDDY::PolationPara&  pp,
					     
					     EDDY::CudaVolume&          oima,
					     
					     EDDY::CudaVolume&          omask) EddyTry
{
  if (!oima.Size()) oima.SetHdr(scan.GetIma());
  else if (oima != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: scan<->oima mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: scan<->omask mismatch");
  if (pred.Size() && !scan.IsSliceToVol()) throw EDDY::EddyException("EddyInternalGpuUtils::get_unwarped_scan: pred for volumetric does not make sense");
  EDDY::CudaVolume ima;
  if (use_orig) ima = scan.GetOriginalIma();
  else ima = scan.GetIma();
  EDDY::CudaVolume4D dfield(ima,3,false);
  EDDY::CudaVolume4D skrutt;
  EDDY::CudaVolume jac(ima,false);
  EDDY::CudaVolume mask2;
  if (omask.Size()) { mask2.SetHdr(jac); mask2 = 1.0; }
  EddyInternalGpuUtils::field_for_scan_to_model_transform(scan,susc,dfield,omask,jac);
  if (scan.IsSliceToVol()) {
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    EddyInternalGpuUtils::general_slice_to_vol_transform(ima,iR,dfield,jac,pred,jacmod,pp.GetS2VInterp(),oima,mask2);
  }
  else {
    NEWMAT::Matrix iR = scan.InverseMovementMatrix();
    NEWMAT::IdentityMatrix I(4);
    EddyInternalGpuUtils::general_transform(ima,iR,dfield,I,oima,skrutt,mask2);
    if (jacmod) oima *= jac;
  }
  if (omask.Size()) {
    omask *= mask2;
    omask.SetInterp(NEWIMAGE::trilinear);
  }
} EddyCatch

void EddyInternalGpuUtils::get_volumetric_unwarped_scan(
							const EDDY::ECScan&        scan,
							const EDDY::CudaVolume&    susc,
							bool                       jacmod,
							bool                       use_orig,
							const EDDY::PolationPara&  pp,
							
							EDDY::CudaVolume&          oima,
							
							EDDY::CudaVolume&          omask,
							EDDY::CudaVolume4D&        deriv) EddyTry
{
  if (!oima.Size()) oima.SetHdr(scan.GetIma());
  else if (oima != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->oima mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->omask mismatch");
  if (deriv.Size() && deriv != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::get_volumetric_unwarped_scan: scan<->deriv mismatch");
  EDDY::CudaVolume ima;
  if (use_orig) ima = scan.GetOriginalIma();
  else ima = scan.GetIma();
  EDDY::CudaVolume4D dfield(ima,3,false);
  EDDY::CudaVolume jac(ima,false);
  EDDY::CudaVolume mask2;
  if (omask.Size()) { mask2.SetHdr(jac); mask2 = 1.0; }
  EddyInternalGpuUtils::field_for_scan_to_model_volumetric_transform(scan,susc,dfield,omask,jac);
  NEWMAT::Matrix iR = scan.InverseMovementMatrix();
  NEWMAT::IdentityMatrix I(4);
  EddyInternalGpuUtils::general_transform(ima,iR,dfield,I,oima,deriv,mask2);
  if (jacmod) oima *= jac;
  if (omask.Size()) {
    omask *= mask2;
    omask.SetInterp(NEWIMAGE::trilinear);
  }
} EddyCatch

void EddyInternalGpuUtils::detect_outliers(
					   const EddyCommandLineOptions&             clo,
					   ScanType                                  st,
					   const std::shared_ptr<DWIPredictionMaker> pmp,
					   const NEWIMAGE::volume<float>&            pmask,
					   const ECScanManager&                      sm,
					   
					   ReplacementManager&                       rm,
					   DiffStatsVector&                          dsv) EddyTry
{
  if (dsv.NScan() != sm.NScans(st)) throw EDDY::EddyException("EddyInternalGpuUtils::detect_outliers: dsv<->sm mismatch");
  if (clo.Verbose()) cout << "Checking for outliers" << endl;
  
  for (GpuPredictorChunk c(sm.NScans(st),pmask); c<sm.NScans(st); c++) {
    std::vector<unsigned int> si = c.Indicies();
    EDDY::CudaVolume   pios(pmask,false);
    EDDY::CudaVolume   mios(pmask,false);
    EDDY::CudaVolume   mask(pmask,false);
    EDDY::CudaVolume   skrutt(pmask,false);
    EDDY::CudaVolume4D skrutt4D;
    if (clo.VeryVerbose()) cout << "Making predictions for scans: " << c << endl;
    std::vector<NEWIMAGE::volume<float> > cpred = pmp->Predict(si);
    if (clo.VeryVerbose()) { cout << "Checking scan: "; cout.flush(); }
    for (unsigned int i=0; i<si.size(); i++) {
      if (clo.VeryVerbose()) { cout << si[i] << " "; cout.flush(); }
      EDDY::CudaVolume gpred = cpred[i];
      
      EDDY::CudaVolume susc;
      if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(si[i],st));
      EddyInternalGpuUtils::transform_model_to_scan_space(gpred,sm.Scan(si[i],st),susc,true,pios,mask,skrutt,skrutt4D);
      
      CudaVolume bmask = sm.Mask();
      bmask *= pmask; bmask.SetInterp(NEWIMAGE::trilinear);
      EddyInternalGpuUtils::transform_model_to_scan_space(bmask,sm.Scan(si[i],st),susc,false,mios,mask,skrutt,skrutt4D);
      mios.Binarise(0.99); 
      mask *= mios;        
      
      DiffStats stats(sm.Scan(si[i],st).GetOriginalIma()-pios.GetVolume(),mask.GetVolume());
      dsv[si[i]] = stats;
    }
  }
  if (clo.VeryVerbose()) cout << endl;

  
  rm.Update(dsv);
  return;
} EddyCatch

void EddyInternalGpuUtils::replace_outliers(
					    const EddyCommandLineOptions&             clo,
					    ScanType                                  st,
					    const std::shared_ptr<DWIPredictionMaker> pmp,
					    const NEWIMAGE::volume<float>&            pmask,
					    const ReplacementManager&                 rm,
					    bool                                      add_noise,
					    
					    ECScanManager&                            sm) EddyTry
{
  
  if (clo.VeryVerbose()) cout << "Replacing outliers with predictions" << endl;
  for (unsigned int s=0; s<sm.NScans(st); s++) {
    std::vector<unsigned int> ol = rm.OutliersInScan(s);
    if (ol.size()) { 
      if (clo.VeryVerbose()) cout << "Scan " << s << " has " << ol.size() << " outlier slices" << endl;
      EDDY::CudaVolume pred = pmp->Predict(s,true); 
      EDDY::CudaVolume pios(pred,false);
      EDDY::CudaVolume mios(pred,false);
      EDDY::CudaVolume mask(pred,false);
      EDDY::CudaVolume jac(pred,false);
      EDDY::CudaVolume   skrutt;;
      EDDY::CudaVolume4D skrutt4D;
      EDDY::CudaVolume susc;
      if (sm.HasSuscHzOffResField()) susc = *(sm.GetSuscHzOffResField(s,st));
      
      EddyInternalGpuUtils::transform_model_to_scan_space(pred,sm.Scan(s,st),susc,true,pios,mask,jac,skrutt4D);
      
      EDDY::CudaVolume pmask_cuda = pmask; pmask_cuda.SetInterp(NEWIMAGE::trilinear);
      EddyInternalGpuUtils::transform_model_to_scan_space(pmask_cuda,sm.Scan(s,st),susc,false,mios,mask,skrutt,skrutt4D);
      mios.Binarise(0.9); 
      mask *= mios;        
      if (add_noise) {
        double vp = pmp->PredictionVariance(s,true);
	double ve = pmp->ErrorVariance(s);
	double stdev = std::sqrt(vp+ve) - std::sqrt(vp);
	EDDY::CudaVolume nvol(pios,false);
	nvol.MakeNormRand(0.0,stdev);
	pios += nvol;
      }
      sm.Scan(s,st).SetAsOutliers(pios.GetVolume(),mask.GetVolume(),ol);
    }
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::field_for_scan_to_model_transform(
							     const EDDY::ECScan&            scan,
							     const EDDY::CudaVolume&        susc,
							     
							     EDDY::CudaVolume4D&            dfield,
							     
							     EDDY::CudaVolume&              omask,
							     EDDY::CudaVolume&              jac) EddyTry
{
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->susc mismatch");
  if (dfield.Size() && dfield != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->dfield mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->omask mismatch");
  if (jac.Size() && jac != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->jac mismatch");
  
  EDDY::CudaVolume  ec;
  EddyInternalGpuUtils::get_ec_field(scan,ec);
  
  if (omask.Size()) { omask = 1.0; omask.SetInterp(NEWIMAGE::trilinear); }
  EDDY::CudaVolume  tot(ec,false); tot = 0.0;
  if (scan.IsSliceToVol()) {
    
    
    
    std::vector<NEWMAT::Matrix> iR = EddyUtils::GetSliceWiseInverseMovementMatrices(scan);
    NEWMAT::Matrix M1 = ec.Ima2WorldMatrix();
    std::vector<NEWMAT::Matrix> AA(iR.size());
    for (unsigned int i=0; i<iR.size(); i++) AA[i] = iR[i].i();
    NEWMAT::Matrix M2 = ec.World2ImaMatrix();  
    EDDY::CudaImageCoordinates coord(ec.Size(0),ec.Size(1),ec.Size(2),false);
    EDDY::CudaVolume zcoordV(ec,false);
    EDDY::CudaVolume4D skrutt(ec,3,false);
    skrutt=0.0;
    coord.GetSliceToVolXYZCoord(M1,AA,skrutt,M2,zcoordV);
    
    ec.Sample(coord,tot);
    
    if (susc.Size()) {
      EDDY::CudaImageCoordinates zcoord(ec.Size(0),ec.Size(1),ec.Size(2),false);
      zcoord.GetSliceToVolZCoord(M1,AA,skrutt,M2);
      EDDY::CudaVolume tmp(ec,false);
      susc.Sample(zcoord,tmp);
      tot += tmp;
    }
  }
  else {
    
    NEWMAT::Matrix    iR = scan.InverseMovementMatrix();
    EDDY::CudaVolume4D skrutt;
    
    EddyInternalGpuUtils::affine_transform(ec,iR,tot,skrutt,omask);
    
    if (susc.Size()) tot += susc;
  }
  
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  
  if (jac.Size()) FieldGpuUtils::GetJacobian(dfield,scan.GetAcqPara(),jac);
  
  FieldGpuUtils::Voxel2MMDisplacements(dfield);
} EddyCatch

void EddyInternalGpuUtils::field_for_scan_to_model_volumetric_transform(
									const EDDY::ECScan&            scan,
									const EDDY::CudaVolume&        susc,
									
									EDDY::CudaVolume4D&            dfield,
									
									EDDY::CudaVolume&              omask,
									EDDY::CudaVolume&              jac) EddyTry
{
  if (susc.Size() && susc != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->susc mismatch");
  if (dfield.Size() && dfield != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->dfield mismatch");
  if (omask.Size() && omask != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->omask mismatch");
  if (jac.Size() && jac != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_scan_to_model_transform: scan<->jac mismatch");
  
  EDDY::CudaVolume  ec;
  EddyInternalGpuUtils::get_ec_field(scan,ec);
  
  if (omask.Size()) { omask = 1.0; omask.SetInterp(NEWIMAGE::trilinear); }
  EDDY::CudaVolume  tot(ec,false); tot = 0.0;
  
  NEWMAT::Matrix    iR = scan.InverseMovementMatrix();
  EDDY::CudaVolume4D skrutt;
  
  EddyInternalGpuUtils::affine_transform(ec,iR,tot,skrutt,omask);
  
  if (susc.Size()) tot += susc;
  
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  
  if (jac.Size()) FieldGpuUtils::GetJacobian(dfield,scan.GetAcqPara(),jac);
  
  FieldGpuUtils::Voxel2MMDisplacements(dfield);
} EddyCatch

double EddyInternalGpuUtils::param_update(
					  const NEWIMAGE::volume<float>&                  pred,     
					  std::shared_ptr<const NEWIMAGE::volume<float> > susc,     
					  const NEWIMAGE::volume<float>&                  pmask,    
					  EDDY::Parameters                                whichp,   
					  bool                                            cbs,      
					  float                                           fwhm,     
					  const EDDY::PolationPara&                       pp,       
					  
					  unsigned int                                    scindx,
					  unsigned int                                    iter,
					  unsigned int                                    level,
					  
					  EDDY::ECScan&                                   scan,     
					  
					  NEWMAT::ColumnVector                            *rupdate) EddyTry 
{
  
  EDDY::CudaVolume pred_gpu(pred);
  
  EDDY::CudaVolume susc_gpu;
  if (susc != nullptr) susc_gpu = *susc;
  
  EDDY::CudaVolume pios(pred,false);
  EDDY::CudaVolume mask(pred,false); mask = 1.0;
  EDDY::CudaVolume jac(pred,false);
  EDDY::CudaVolume jacmask(pred,false);
  EDDY::CudaVolume   skrutt;
  EDDY::CudaVolume4D skrutt4D;
  EddyInternalGpuUtils::transform_model_to_scan_space(pred_gpu,scan,susc_gpu,true,pios,mask,jac,skrutt4D);
  
  EDDY::CudaVolume pmask_gpu(pmask); pmask_gpu.SetInterp(NEWIMAGE::trilinear);
  EDDY::CudaVolume mios(pmask,false);
  EDDY::CudaVolume skruttmask(pred,false);
  EddyInternalGpuUtils::transform_model_to_scan_space(pmask_gpu,scan,susc_gpu,false,mios,skruttmask,skrutt,skrutt4D);
  
  mios.Binarise(0.99); mask *= mios;
  
  EDDY::CudaVolume4D derivs(pred,scan.NDerivs(),false);
  EddyInternalGpuUtils::get_direct_partial_derivatives_in_scan_space(pred_gpu,scan,susc_gpu,whichp,derivs);
  if (fwhm) derivs.Smooth(fwhm,mask);
  
  NEWMAT::Matrix XtX = EddyInternalGpuUtils::make_XtX(derivs,mask);
  
  EDDY::CudaVolume dima = pios-EDDY::CudaVolume(scan.GetIma());
  if (fwhm) dima.Smooth(fwhm,mask);
  
  NEWMAT::ColumnVector Xty = EddyInternalGpuUtils::make_Xty(derivs,dima,mask);
  
  NEWMAT::ColumnVector lHb = scan.GetRegGrad(whichp);
  NEWMAT::Matrix H = scan.GetRegHess(whichp);
  
  double mss = dima.SumOfSquares(mask) / mask.Sum() + scan.GetReg(whichp);
  
  double lambda = 1.0/mask.Sum();
  NEWMAT::IdentityMatrix eye(XtX.Nrows());
  
  NEWMAT::ColumnVector update = -(XtX/mask.Sum() + H + lambda*eye).i()*(Xty/mask.Sum() + lHb);
  EDDY::CudaVolume sims; 
  if (level) EddyInternalGpuUtils::get_unwarped_scan(scan,susc_gpu,skrutt,true,false,pp,sims,skrutt);
  
  for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
    scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)+update(i+1),whichp);
  }
  if (!level) { 
    if (cbs) {
      EddyInternalGpuUtils::transform_model_to_scan_space(pred_gpu,scan,susc_gpu,true,pios,mask,jac,skrutt4D);
      
      mask = 0.0;
      EddyInternalGpuUtils::transform_model_to_scan_space(pmask_gpu,scan,susc_gpu,false,mios,mask,skrutt,skrutt4D);
      mios.Binarise(0.99); 
      mask *= mios; 
      dima = pios-EDDY::CudaVolume(scan.GetIma());
      if (fwhm) dima.Smooth(fwhm,mask);
      double mss_au = dima.SumOfSquares(mask) / mask.Sum() + scan.GetReg(whichp);
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
    EDDY::CudaVolume  new_pios;
    EDDY::CudaVolume  new_mios;
    EDDY::CudaVolume  new_mask;
    EDDY::CudaVolume  new_dima;
    EDDY::CudaVolume  new_jac(pred,false);
    if (cbs) {
      EddyInternalGpuUtils::transform_model_to_scan_space(pred_gpu,scan,susc_gpu,true,new_pios,new_mask,new_jac,skrutt4D);
      
      new_mask = 0.0;
      EddyInternalGpuUtils::transform_model_to_scan_space(pmask_gpu,scan,susc_gpu,false,new_mios,new_mask,skrutt,skrutt4D);
      new_mios.Binarise(0.99); 
      new_mask *= new_mios; 
      new_dima = new_pios-EDDY::CudaVolume(scan.GetIma());
      if (fwhm) new_dima.Smooth(fwhm,mask);
      double mss_au = new_dima.SumOfSquares(new_mask) / new_mask.Sum();
      if (mss_au > mss) { 
	for (unsigned int i=0; i<scan.NDerivs(whichp); i++) {
	  scan.SetDerivParam(i,scan.GetDerivParam(i,whichp)-update(i+1),whichp);
	}
      }
    }
    if (rupdate) *rupdate = update;
    
    char fname[256], bname[256];
    EDDY::CudaVolume scratch;
    if (scan.IsSliceToVol()) strcpy(bname,"EDDY_DEBUG_S2V_GPU");
    else strcpy(bname,"EDDY_DEBUG_GPU");
    if (level>0) {
      sprintf(fname,"%s_masked_dima_%02d_%04d",bname,iter,scindx);
      scratch = dima * mask; scratch.Write(fname);
    }
    if (level>1) {
      sprintf(fname,"%s_mask_%02d_%04d",bname,iter,scindx); mask.Write(fname);
      sprintf(fname,"%s_pios_%02d_%04d",bname,iter,scindx); pios.Write(fname);
      sprintf(fname,"%s_pred_%02d_%04d",bname,iter,scindx); pred_gpu.Write(fname);
      sprintf(fname,"%s_dima_%02d_%04d",bname,iter,scindx); dima.Write(fname);
      sprintf(fname,"%s_jac_%02d_%04d",bname,iter,scindx); jac.Write(fname);
      sprintf(fname,"%s_orig_%02d_%04d",bname,iter,scindx);
      scratch = scan.GetIma(); scratch.Write(fname);
      if (cbs) {
	sprintf(fname,"%s_new_masked_dima_%02d_%04d",bname,iter,scindx);
	scratch = new_dima * new_mask; scratch.Write(fname);
	sprintf(fname,"%s_new_reverse_dima_%02d_%04d",bname,iter,scindx);
	EddyInternalGpuUtils::get_unwarped_scan(scan,susc_gpu,skrutt,true,false,pp,scratch,skrutt);
        scratch = pred_gpu - scratch; scratch.Write(fname);
	sprintf(fname,"%s_new_mask_%02d_%04d",bname,iter,scindx); new_mask.Write(fname);
	sprintf(fname,"%s_new_pios_%02d_%04d",bname,iter,scindx); new_pios.Write(fname);
	sprintf(fname,"%s_new_dima_%02d_%04d",bname,iter,scindx); new_dima.Write(fname);
	sprintf(fname,"%s_new_jac_%02d_%04d",bname,iter,scindx); new_jac.Write(fname);
      }
    }
    if (level>2) {
      sprintf(fname,"%s_mios_%02d_%04d",bname,iter,scindx); mios.Write(fname);
      sprintf(fname,"%s_pmask_%02d_%04d",bname,iter,scindx); pmask_gpu.Write(fname);
      sprintf(fname,"%s_derivs_%02d_%04d",bname,iter,scindx); derivs.Write(fname);
      sprintf(fname,"%s_XtX_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,XtX);
      sprintf(fname,"%s_Xty_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,Xty);
      sprintf(fname,"%s_update_%02d_%04d.txt",bname,iter,scindx); MISCMATHS::write_ascii_matrix(fname,update);
    }
  } 
  
  return(mss);
} EddyCatch



void EddyInternalGpuUtils::transform_model_to_scan_space(
							 const EDDY::CudaVolume&       pred,
							 const EDDY::ECScan&           scan,
							 const EDDY::CudaVolume&       susc,
							 bool                          jacmod,
							 
							 EDDY::CudaVolume&             oima,
							 EDDY::CudaVolume&             omask,
							 
							 EDDY::CudaVolume&             jac,
							 EDDY::CudaVolume4D&           grad) EddyTry
{
  
  if (oima != pred) oima.SetHdr(pred);
  if (omask != pred) omask.SetHdr(pred);
  if (jac.Size() && jac!=pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_model_to_scan_space: jac size mismatch");
  if (grad.Size() && grad!=pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_model_to_scan_space: grad size mismatch");
  if (jacmod && !jac.Size()) throw EDDY::EddyException("EddyInternalGpuUtils::transform_model_to_scan_space: jacmod can only be used with valid jac");
  EDDY::CudaVolume4D dfield(susc,3,false);
  EDDY::CudaVolume mask2(omask,false);
  NEWMAT::IdentityMatrix I(4);
  if (scan.IsSliceToVol()) {
    
    EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac);
    
    std::vector<NEWMAT::Matrix> R = EddyUtils::GetSliceWiseForwardMovementMatrices(scan);
    std::vector<NEWMAT::Matrix> II(R.size()); for (unsigned int i=0; i<R.size(); i++) II[i] = I;
    
    EddyInternalGpuUtils::general_transform(pred,II,dfield,R,oima,grad,mask2);
  }
  else {
    
    EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac);
    
    NEWMAT::Matrix R = scan.ForwardMovementMatrix();
    
    EddyInternalGpuUtils::general_transform(pred,I,dfield,R,oima,grad,mask2);
  }
  omask *= mask2;
  omask.SetInterp(NEWIMAGE::trilinear);
  if (jacmod) oima *= jac;

  return;
} EddyCatch

void EddyInternalGpuUtils::field_for_model_to_scan_transform(
							     const EDDY::ECScan&           scan,
							     const EDDY::CudaVolume&       susc,
							     
							     EDDY::CudaVolume4D&           idfield,
							     EDDY::CudaVolume&             omask,
							     
							     EDDY::CudaVolume&             jac) EddyTry
{
  
  if (idfield != scan.GetIma()) idfield.SetHdr(scan.GetIma(),3);
  if (omask.Size() && omask != scan.GetIma()) omask.SetHdr(scan.GetIma());
  if (jac.Size() && jac != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::field_for_model_to_scan_transform: jac size mismatch");

  EDDY::CudaVolume tot(scan.GetIma(),false);              
  EDDY::CudaVolume mask(scan.GetIma(),false); mask = 1.0; 
  EddyInternalGpuUtils::get_ec_field(scan,tot);
  if (susc.Size()) {
    EDDY::CudaVolume tsusc(susc,false);
    if (scan.IsSliceToVol()) {
      std::vector<NEWMAT::Matrix> R = EddyUtils::GetSliceWiseForwardMovementMatrices(scan);
      EDDY::CudaVolume4D skrutt;
      EddyInternalGpuUtils::affine_transform(susc,R,tsusc,skrutt,mask);
    }
    else {
      NEWMAT::Matrix R = scan.ForwardMovementMatrix();
      EDDY::CudaVolume4D skrutt;
      EddyInternalGpuUtils::affine_transform(susc,R,tsusc,skrutt,mask);
    }
    tot += tsusc;
  }
  
  EDDY::CudaVolume4D dfield(tot,3,false);
  FieldGpuUtils::Hz2VoxelDisplacements(tot,scan.GetAcqPara(),dfield);
  
  FieldGpuUtils::InvertDisplacementField(dfield,scan.GetAcqPara(),mask,idfield,omask);
  
  if (jac.Size()) FieldGpuUtils::GetJacobian(idfield,scan.GetAcqPara(),jac);
  
  FieldGpuUtils::Voxel2MMDisplacements(idfield);
} EddyCatch

EDDY::CudaImageCoordinates EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space(
												const EDDY::CudaVolume&     pred,
												const EDDY::ECScan&         scan,
												const EDDY::CudaVolume&     susc,
												
												EDDY::CudaImageCoordinates& coord,
												
												EDDY::CudaVolume&           omask,
												EDDY::CudaVolume&           jac) EddyTry
{
  if (pred != scan.GetIma()) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->scan mismatch");
  if (susc.Size() && pred != susc) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->susc mismatch");
  if (omask.Size() && omask != pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->omask mismatch");
  if (jac.Size() && jac != pred) throw EDDY::EddyException("EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space: pred<->jac mismatch");
  
  EDDY::CudaVolume4D dfield;
  EddyInternalGpuUtils::field_for_model_to_scan_transform(scan,susc,dfield,omask,jac);
  
  NEWMAT::Matrix R = scan.ForwardMovementMatrix();
  
  NEWMAT::Matrix A = pred.Ima2WorldMatrix();
  NEWMAT::Matrix M = pred.World2ImaMatrix() * R.i();
  
  if (omask.Size()) { 
    EDDY::CudaVolume mask2(omask,false);
    coord.Transform(A,dfield,M);
    pred.ValidMask(coord,mask2);
    omask *= mask2;
    omask.SetInterp(NEWIMAGE::trilinear);
  }
  else coord.Transform(A,dfield,M);

  return(coord);
} EddyCatch

void EddyInternalGpuUtils::get_partial_derivatives_in_scan_space(const EDDY::CudaVolume& pred,
								 const EDDY::ECScan&     scan,
								 const EDDY::CudaVolume& susc,
								 EDDY::Parameters        whichp,
								 EDDY::CudaVolume4D&     derivs) EddyTry
{
  EDDY::CudaVolume base(pred,false);
  EDDY::CudaVolume mask(pred,false);
  EDDY::CudaVolume basejac(pred,false);
  EDDY::CudaVolume4D grad(pred,3,false);
  EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,base,mask,basejac,grad);
  EDDY::CudaImageCoordinates basecoord(pred.Size(0),pred.Size(1),pred.Size(2));
  EDDY::CudaVolume skrutt;
  EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space(pred,scan,susc,basecoord,skrutt,skrutt);
  if (derivs != pred || derivs.Size(3) != scan.NDerivs(whichp)) derivs.SetHdr(pred,scan.NDerivs(whichp));
  EDDY::CudaVolume jac(pred,false);
  EDDY::ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    EDDY::CudaImageCoordinates diffcoord(pred.Size(0),pred.Size(1),pred.Size(2));
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    EddyInternalGpuUtils::transform_coordinates_from_model_to_scan_space(pred,sc,susc,diffcoord,skrutt,jac);
    diffcoord -= basecoord;
    EddyInternalGpuUtils::make_deriv_from_components(diffcoord,grad,base,jac,basejac,sc.GetDerivScale(i,whichp),derivs,i);
    sc.SetDerivParam(i,p,whichp);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::get_direct_partial_derivatives_in_scan_space(const EDDY::CudaVolume& pred,
									const EDDY::ECScan&     scan,
									const EDDY::CudaVolume& susc,
									EDDY::Parameters        whichp,
									EDDY::CudaVolume4D&     derivs) EddyTry
{
  EDDY::CudaVolume base(pred,false);
  EDDY::CudaVolume mask(pred,false);
  EDDY::CudaVolume jac(pred,false);
  EDDY::CudaVolume4D grad;
  EddyInternalGpuUtils::transform_model_to_scan_space(pred,scan,susc,true,base,mask,jac,grad);
  if (derivs != pred || derivs.Size(3) != scan.NDerivs(whichp)) derivs.SetHdr(pred,scan.NDerivs(whichp));
  EDDY::CudaVolume perturbed(pred,false);
  EDDY::ECScan sc = scan;
  for (unsigned int i=0; i<sc.NDerivs(whichp); i++) {
    double p = sc.GetDerivParam(i,whichp);
    sc.SetDerivParam(i,p+sc.GetDerivScale(i,whichp),whichp);
    EddyInternalGpuUtils::transform_model_to_scan_space(pred,sc,susc,true,perturbed,mask,jac,grad);
    derivs[i] = (perturbed-base)/sc.GetDerivScale(i,whichp);
    sc.SetDerivParam(i,p,whichp);
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::make_deriv_from_components(const EDDY::CudaImageCoordinates&  coord,
						      const EDDY::CudaVolume4D&          grad,
						      const EDDY::CudaVolume&            base,
						      const EDDY::CudaVolume&            jac,
						      const EDDY::CudaVolume&            basejac,
						      float                              dstep,
						      EDDY::CudaVolume4D&                deriv,
						      unsigned int                       indx) EddyTry
{
  int tpb = EddyInternalGpuUtils::threads_per_block_make_deriv;
  int nthreads = base.Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::make_deriv<<<nblocks,tpb>>>(base.Size(0),base.Size(1),base.Size(2),coord.XPtr(),coord.YPtr(),coord.ZPtr(),
					   grad.GetPtr(0),grad.GetPtr(1),grad.GetPtr(2),base.GetPtr(),jac.GetPtr(),
					   basejac.GetPtr(),dstep,deriv.GetPtr(indx),nthreads);
  EddyKernels::CudaSync("EddyKernels::make_deriv");
  
  return;
} EddyCatch

NEWMAT::Matrix EddyInternalGpuUtils::make_XtX(const EDDY::CudaVolume4D&  X,
					      const EDDY::CudaVolume&    mask) EddyTry
{
  NEWMAT::Matrix XtX(X.Size(3),X.Size(3));
  for (int i=0; i<X.Size(3); i++) {
    thrust::device_vector<float> masked(X.Size());
    try {
      thrust::copy(mask.Begin(),mask.End(),masked.begin());
      thrust::transform(X.Begin(i),X.End(i),masked.begin(),masked.begin(),thrust::multiplies<float>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_XtX after copy and transform with index: " << i << ", and message: " << e.what() << std::endl;
      throw;
    }    
    for (int j=i; j<X.Size(3); j++) {
      try {
	XtX(j+1,i+1) = thrust::inner_product(masked.begin(),masked.end(),X.Begin(j),0.0f);
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_XtX after inner_product with i = " << i << ", j = " << j << ", and message: " << e.what() << std::endl;
	throw;
      }    
      if (j!=i) XtX(i+1,j+1) = XtX(j+1,i+1);
    }
  }
  return(XtX);
} EddyCatch

NEWMAT::ColumnVector EddyInternalGpuUtils::make_Xty(const EDDY::CudaVolume4D&  X,
						    const EDDY::CudaVolume&    y,
						    const EDDY::CudaVolume&    mask) EddyTry
{
  NEWMAT::ColumnVector Xty(X.Size(3));
  thrust::device_vector<float> masked(X.Size());
  try {
    thrust::copy(mask.Begin(),mask.End(),masked.begin());
    thrust::transform(y.Begin(),y.End(),masked.begin(),masked.begin(),thrust::multiplies<float>());
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_Xty after copy and transform with message: " << e.what() << std::endl;
    throw;
  }    
  for (int i=0; i<X.Size(3); i++) {
    try {
      Xty(i+1) = thrust::inner_product(masked.begin(),masked.end(),X.Begin(i),0.0f);
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in EddyInternalGpuUtils::make_Xty after inner_product with i = " << i << ", and with message: " << e.what() << std::endl;
      throw;
    }    
  }  
  return(Xty);
} EddyCatch


void FieldGpuUtils::Hz2VoxelDisplacements(const EDDY::CudaVolume&  hzfield,
					  const EDDY::AcqPara&     acqp,
					  EDDY::CudaVolume4D&      dfield) EddyTry
{
  if (dfield != hzfield || dfield.Size(3) != 3) dfield.SetHdr(hzfield,3);
  for (unsigned int i=0; i<3; i++) {
    if (acqp.PhaseEncodeVector()(i+1)) {
      thrust::transform(hzfield.Begin(),hzfield.End(),dfield.Begin(i),EDDY::MulByScalar<float>((acqp.PhaseEncodeVector())(i+1) * acqp.ReadOutTime()));
    }
    else thrust::fill(dfield.Begin(i),dfield.End(i),0.0);
  }
} EddyCatch

void FieldGpuUtils::Voxel2MMDisplacements(EDDY::CudaVolume4D&      dfield) EddyTry
{
  if (dfield.Size(3) != 3) throw EDDY::EddyException("FieldGpuUtils::Voxel2MMDisplacements: dfield.Size(3) must be 3");
  for (unsigned int i=0; i<dfield.Size(3); i++) {
    thrust::transform(dfield.Begin(i),dfield.End(i),dfield.Begin(i),EDDY::MulByScalar<float>(dfield.Vxs(i)));
  }
} EddyCatch

void FieldGpuUtils::InvertDisplacementField(
					    const EDDY::CudaVolume4D&  dfield,
					    const EDDY::AcqPara&       acqp,
					    const EDDY::CudaVolume&    inmask,
					    
					    EDDY::CudaVolume4D&        idfield,
					    EDDY::CudaVolume&          omask) EddyTry
					    
{
  if (inmask != dfield) throw EDDY::EddyException("FieldGpuUtils::InvertDisplacementField: dfield<->inmask mismatch");
  if (acqp.PhaseEncodeVector()(1) && acqp.PhaseEncodeVector()(2)) throw EDDY::EddyException("FieldGpuUtils::InvertDisplacementField: Phase encode vector must have exactly one non-zero component");
  if (acqp.PhaseEncodeVector()(3)) throw EDDY::EddyException("FieldGpuUtils::InvertDisplacementField: Phase encode in z not allowed.");
  if (idfield != dfield) idfield.SetHdr(dfield); idfield = 0.0;
  if (omask != inmask) omask.SetHdr(inmask); omask = 0.0;
  int tpb = FieldGpuUtils::threads_per_block_invert_field;
  if (acqp.PhaseEncodeVector()(1)) {
    int nthreads = dfield.Size(1)*dfield.Size(2);
    int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
    EddyKernels::invert_displacement_field<<<nblocks,tpb>>>(dfield.GetPtr(0),inmask.GetPtr(),dfield.Size(0),dfield.Size(1),
							    dfield.Size(2),0,idfield.GetPtr(0),omask.GetPtr(),nthreads);
    EddyKernels::CudaSync("EddyKernels::invert_displacement_field x");
  }
  else if (acqp.PhaseEncodeVector()(2)) {
    int nthreads = dfield.Size(0)*dfield.Size(2);
    int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
    EddyKernels::invert_displacement_field<<<nblocks,tpb>>>(dfield.GetPtr(1),inmask.GetPtr(),dfield.Size(0),dfield.Size(1),
							    dfield.Size(2),1,idfield.GetPtr(1),omask.GetPtr(),nthreads);
    EddyKernels::CudaSync("EddyKernels::invert_displacement_field y");
  }
} EddyCatch

void FieldGpuUtils::GetJacobian(
				const EDDY::CudaVolume4D&  dfield,
				const EDDY::AcqPara&       acqp,
				
				EDDY::CudaVolume&          jac) EddyTry
{
  if (jac != dfield) jac.SetHdr(dfield);
  unsigned int cnt=0;
  for (unsigned int i=0; i<3; i++) if ((acqp.PhaseEncodeVector())(i+1)) cnt++;
  if (cnt != 1) throw EDDY::EddyException("FieldGpuUtils::GetJacobian: Phase encode vector must have exactly one non-zero component");
  unsigned int dir=0;
  for (dir=0; dir<3; dir++) if ((acqp.PhaseEncodeVector())(dir+1)) break;
  
  EDDY::CudaImageCoordinates coord(jac.Size(0),jac.Size(1),jac.Size(2),true);
  EDDY::CudaVolume tmpfield(jac,false);
  tmpfield.SetInterp(NEWIMAGE::spline);
  thrust::copy(dfield.Begin(dir),dfield.End(dir),tmpfield.Begin());
  EDDY::CudaVolume skrutt(jac,false);
  EDDY::CudaVolume4D grad(jac,3,false);
  tmpfield.Sample(coord,skrutt,grad);
  jac = 1.0;
  thrust::transform(jac.Begin(),jac.End(),grad.Begin(dir),jac.Begin(),thrust::plus<float>());

  return;
} EddyCatch

void EddyInternalGpuUtils::general_transform(
					     const EDDY::CudaVolume&    inima,
					     const NEWMAT::Matrix&      A,
					     const EDDY::CudaVolume4D&  dfield,
					     const NEWMAT::Matrix&      M,
					     
					     EDDY::CudaVolume&          oima,
					     
					     EDDY::CudaVolume4D&        deriv,
					     EDDY::CudaVolume&          omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: ima<->field mismatch");
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size omask");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size deriv");
  
  NEWMAT::Matrix AA = A.i() * inima.Ima2WorldMatrix();  
  NEWMAT::Matrix MM = oima.World2ImaMatrix() * M.i();
  
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(AA,dfield,MM);
  
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  
  if (omask.Size()) inima.ValidMask(coord,omask);  
  
  return;
} EddyCatch

void EddyInternalGpuUtils::general_transform(
					     const EDDY::CudaVolume&             inima,
					     const std::vector<NEWMAT::Matrix>&  A,
					     const EDDY::CudaVolume4D&           dfield,
					     const std::vector<NEWMAT::Matrix>&  M,
					     
					     EDDY::CudaVolume&                   oima,
					     
					     EDDY::CudaVolume4D&                 deriv,
					     EDDY::CudaVolume&                   omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: ima<->field mismatch");
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size omask");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: wrong size deriv");
  if (A.size() != inima.Size(2) || M.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::general_transform: mismatched A or M vector");
  
  std::vector<NEWMAT::Matrix> AA(A.size());
  for (unsigned int i=0; i<A.size(); i++) AA[i] = A[i].i() * inima.Ima2WorldMatrix();  
  std::vector<NEWMAT::Matrix> MM(M.size());
  for (unsigned int i=0; i<M.size(); i++) MM[i] = oima.World2ImaMatrix() * M[i].i();
  
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(AA,dfield,MM);
  
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  
  if (omask.Size()) inima.ValidMask(coord,omask);  
  
  return;
} EddyCatch

void EddyInternalGpuUtils::general_slice_to_vol_transform(
							  const EDDY::CudaVolume&             inima,
							  const std::vector<NEWMAT::Matrix>&  A,
							  const EDDY::CudaVolume4D&           dfield,
							  const EDDY::CudaVolume&             jac,
							  const EDDY::CudaVolume&             pred,
							  bool                                jacmod,
							  NEWIMAGE::interpolation             interp,
							  
							  EDDY::CudaVolume&                   oima,
							  
							  EDDY::CudaVolume&                   omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: wrong size omask");
  if (A.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: mismatched A vector");
  
  if (dfield.Size()) { 
    if (dfield != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: ima<->field mismatch");
    if (jacmod && jac != inima) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: ima<->jac mismatch");
  }
  else { 
    if (jacmod) throw EDDY::EddyException("EddyInternalGpuUtils::general_slice_to_vol_transform: Invalid combination of jacmod and affine transform");
  }
  
  NEWMAT::Matrix M1 = inima.Ima2WorldMatrix();
  std::vector<NEWMAT::Matrix> AA(A.size());
  for (unsigned int i=0; i<A.size(); i++) AA[i] = A[i].i();
  NEWMAT::Matrix M2 = oima.World2ImaMatrix();  

  
  

  
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  EDDY::CudaVolume zcoordV(inima,false);
  
  coord.GetSliceToVolXYZCoord(M1,AA,dfield,M2,zcoordV);
  
  EDDY::CudaVolume resampled2D(inima,false);
  inima.Sample(coord,resampled2D);
  if (jacmod) resampled2D *= jac;
  EDDY::CudaVolume mask(inima,false);
  inima.ValidMask(coord,mask);
  
  if (pred.Size()) {
    StackResampler sr(resampled2D,zcoordV,pred,mask,0.005);
    oima = sr.GetResampledIma();
    if (omask.Size()) omask = sr.GetMask();
  }
  else {
    StackResampler sr(resampled2D,zcoordV,mask,interp,0.005);
    oima = sr.GetResampledIma();
    if (omask.Size()) omask = sr.GetMask();
  }
  return;
} EddyCatch

void EddyInternalGpuUtils::affine_transform(const EDDY::CudaVolume&    inima,
					    const NEWMAT::Matrix&      R,
					    EDDY::CudaVolume&          oima,
					    EDDY::CudaVolume4D&        deriv,
					    EDDY::CudaVolume&          omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->omask mismatch");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->deriv mismatch");
  
  NEWMAT::Matrix A = oima.World2ImaMatrix() * R.i() * inima.Ima2WorldMatrix();
  
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(A);
  
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  
  if (omask.Size()) inima.ValidMask(coord,omask);  

  return;
} EddyCatch
						       
void EddyInternalGpuUtils::affine_transform(const EDDY::CudaVolume&             inima,
					    const std::vector<NEWMAT::Matrix>&  R,
					    EDDY::CudaVolume&                   oima,
					    EDDY::CudaVolume4D&                 deriv,
					    EDDY::CudaVolume&                   omask) EddyTry
{
  if (oima != inima) oima.SetHdr(inima);
  if (omask.Size() && omask != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->omask mismatch");
  if (deriv.Size() && deriv != inima) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: inima<->deriv mismatch");
  if (R.size() != inima.Size(2)) throw EDDY::EddyException("EddyInternalGpuUtils::affine_transform: mismatched R vector");
  
  std::vector<NEWMAT::Matrix> A(R.size());
  for (unsigned int i=0; i<R.size(); i++) A[i] = oima.World2ImaMatrix() * R[i].i() * inima.Ima2WorldMatrix();
  
  EDDY::CudaImageCoordinates coord(inima.Size(0),inima.Size(1),inima.Size(2),false);
  coord.Transform(A);
  
  if (deriv.Size()) inima.Sample(coord,oima,deriv);
  else inima.Sample(coord,oima);
  
  if (omask.Size()) inima.ValidMask(coord,omask);  

  return;
} EddyCatch
						       
void EddyInternalGpuUtils::get_ec_field(
					const EDDY::ECScan&       scan,
					
					EDDY::CudaVolume&         ecfield) EddyTry
{
  if (ecfield != scan.GetIma()) ecfield.SetHdr(scan.GetIma()); 
  
  NEWMAT::ColumnVector epp = scan.GetParams(EDDY::EC);
  thrust::host_vector<float> epp_host(scan.NParam(EDDY::EC));
  for (unsigned int i=0; i<epp_host.size(); i++) epp_host[i] = epp(i+1);
  thrust::device_vector<float> epp_dev = epp_host; 

  int tpb = EddyInternalGpuUtils::threads_per_block_ec_field; 
  int nthreads = ecfield.Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;

  if (scan.Model() == EDDY::NoEC) {
    ecfield = 0.0;
  }
  if (scan.Model() == EDDY::Linear) {
    EddyKernels::linear_ec_field<<<nblocks,tpb>>>(ecfield.GetPtr(),ecfield.Size(0),ecfield.Size(1),ecfield.Size(2),
						  ecfield.Vxs(0),ecfield.Vxs(1),ecfield.Vxs(2),
						  thrust::raw_pointer_cast(epp_dev.data()),epp_dev.size(),nthreads);
    EddyKernels::CudaSync("EddyKernels::linear_ec_field");
  }
  else if (scan.Model() == EDDY::Quadratic) {
    EddyKernels::quadratic_ec_field<<<nblocks,tpb>>>(ecfield.GetPtr(),ecfield.Size(0),ecfield.Size(1),ecfield.Size(2),
						     ecfield.Vxs(0),ecfield.Vxs(1),ecfield.Vxs(2),
						     thrust::raw_pointer_cast(epp_dev.data()),epp_dev.size(),nthreads);
    EddyKernels::CudaSync("EddyKernels::quadratic_ec_field");
  }
  else if (scan.Model() == EDDY::Cubic) {
    EddyKernels::cubic_ec_field<<<nblocks,tpb>>>(ecfield.GetPtr(),ecfield.Size(0),ecfield.Size(1),ecfield.Size(2),
						 ecfield.Vxs(0),ecfield.Vxs(1),ecfield.Vxs(2),
						 thrust::raw_pointer_cast(epp_dev.data()),epp_dev.size(),nthreads);
    EddyKernels::CudaSync("EddyKernels::cubic_ec_field");
  }
  return; 
} EddyCatch

