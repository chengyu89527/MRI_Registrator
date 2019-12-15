// Declarations of classes that implements useful
// utility functions for the eddy current project.
// They are collections of statically declared
// functions that have been collected into classes 
// to make it explicit where they come from. There
// will never be any instances of theses classes.
// 
// EddyUtils.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef EddyUtils_h
#define EddyUtils_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"

namespace EDDY {










class EddyUtils
{
private:
  
  static const int b_range = 100;

  
  static NEWIMAGE::volume4D<float> get_partial_derivatives_in_scan_space(
									 const NEWIMAGE::volume<float>&                    pred,      
									 const EDDY::ECScan&                               scan,      
									 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
                                                                         EDDY::Parameters                                  whichp);

  static NEWIMAGE::volume4D<float> get_direct_partial_derivatives_in_scan_space(
										const NEWIMAGE::volume<float>&                    pred,     
										const EDDY::ECScan&                               scan,     
										std::shared_ptr<const NEWIMAGE::volume<float> >   susc,     
										EDDY::Parameters                                  whichp);

  static double param_update(
			     const NEWIMAGE::volume<float>&                      pred,      
			     std::shared_ptr<const NEWIMAGE::volume<float> >     susc,      
			     const NEWIMAGE::volume<float>&                      pmask,     
			     Parameters                                          whichp,    
			     bool                                                cbs,       
			     float                                               fwhm,      
			     
			     unsigned int                                        scindx,    
			     unsigned int                                        iter,      
			     unsigned int                                        level,     
			     
			     EDDY::ECScan&                                       scan,      
			     
			     NEWMAT::ColumnVector                                *rupdate); 

  static EDDY::ImageCoordinates transform_coordinates_from_model_to_scan_space(
									       const NEWIMAGE::volume<float>&                    pred,
									       const EDDY::ECScan&                               scan,
									       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
									       
									       NEWIMAGE::volume<float>                           *omask,
									       NEWIMAGE::volume<float>                           *jac);
  
  
  static void transform_coordinates(
				    const NEWIMAGE::volume<float>&    f,
				    const NEWIMAGE::volume4D<float>&  d,
				    const NEWMAT::Matrix&             M,
				    std::vector<unsigned int>         slices,
				    
				    ImageCoordinates&                 c,
				    NEWIMAGE::volume<float>           *omask);

  
  static NEWMAT::Matrix make_XtX(const NEWIMAGE::volume4D<float>& vols,
				 const NEWIMAGE::volume<float>&   mask);

  
  
  static NEWMAT::ColumnVector make_Xty(const NEWIMAGE::volume4D<float>& Xvols,
				       const NEWIMAGE::volume<float>&   Yvol,
				       const NEWIMAGE::volume<float>&   mask);
  static bool get_groups(
			 const std::vector<DiffPara>&             dpv,
			 
			 std::vector<std::vector<unsigned int> >& grps,
			 std::vector<unsigned int>&               grpi,
			 std::vector<double>&                     grpb);

public:
  
  
  static NEWIMAGE::volume<float> transform_model_to_scan_space(
							       const NEWIMAGE::volume<float>&                    pred,
							       const EDDY::ECScan&                               scan,
							       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							       bool                                              jacmod,
							       
							       NEWIMAGE::volume<float>&                          omask,
							       NEWIMAGE::volume<float>                           *jac,
							       NEWIMAGE::volume4D<float>                         *grad);

  
  
  static bool AreInSameShell(const DiffPara& dp1,
                             const DiffPara& dp2) EddyTry { return(fabs(dp1.bVal()-dp2.bVal())<double(b_range)); } EddyCatch
  static bool IsDiffusionWeighted(const DiffPara& dp) EddyTry { return(dp.bVal() > double(b_range)); } EddyCatch
  static bool Isb0(const DiffPara& dp) EddyTry { return(!IsDiffusionWeighted(dp)); } EddyCatch
  
  static bool HaveSameDirection(const DiffPara& dp1,
				const DiffPara& dp2) EddyTry { return(NEWMAT::DotProduct(dp1.bVec(),dp2.bVec())>0.999); } EddyCatch
  
  static bool IsShelled(const std::vector<DiffPara>& dpv);
  
  static bool IsMultiShell(const std::vector<DiffPara>& dpv);
  
  static bool GetGroups(
			const std::vector<DiffPara>&             dpv,
			
			std::vector<unsigned int>&               grpi,
			std::vector<double>&                     grpb);
  
  static bool GetGroups(
			const std::vector<DiffPara>&             dpv,
			
			std::vector<std::vector<unsigned int> >& grps,
			std::vector<double>&                     grpb);
  
  static bool GetGroups(
			const std::vector<DiffPara>&             dpv,
			
			std::vector<std::vector<unsigned int> >& grps,
			std::vector<unsigned int>&               grpi,
			std::vector<double>&                     grpb);
  
  template <class V>
  static void SetTrilinearInterp(V& vol) EddyTry {
    if (vol.getinterpolationmethod() != NEWIMAGE::trilinear) vol.setinterpolationmethod(NEWIMAGE::trilinear);
    if (vol.getextrapolationmethod() != NEWIMAGE::mirror) vol.setextrapolationmethod(NEWIMAGE::mirror);
  } EddyCatch
  template <class V>
  static void SetSplineInterp(V& vol) EddyTry {
    if (vol.getinterpolationmethod() != NEWIMAGE::spline) vol.setinterpolationmethod(NEWIMAGE::spline);
    if (vol.getsplineorder() != 3) vol.setsplineorder(3);
    if (vol.getextrapolationmethod() != NEWIMAGE::mirror) vol.setextrapolationmethod(NEWIMAGE::mirror);
  } EddyCatch

  
  static bool AreMatchingPair(const ECScan& s1, const ECScan& s2);
  
  
  static std::vector<unsigned int> GetIndiciesOfDWIs(const std::vector<DiffPara>& dpars);

  
  static std::vector<NEWMAT::Matrix> GetSliceWiseForwardMovementMatrices(const EDDY::ECScan& scan);

  
  static std::vector<NEWMAT::Matrix> GetSliceWiseInverseMovementMatrices(const EDDY::ECScan& scan);

  
  static std::vector<DiffPara> GetDWIDiffParas(const std::vector<DiffPara>&   dpars);

  
  static int read_DWI_volume4D(NEWIMAGE::volume4D<float>&     dwivols,
			       const std::string&             fname,
			       const std::vector<DiffPara>&   dpars);

  
  static NEWIMAGE::volume<float> ConvertMaskToFloat(const NEWIMAGE::volume<char>& charmask);
  
  
  static NEWIMAGE::volume<float> Smooth(const NEWIMAGE::volume<float>& ima,   
					float                          fwhm,  
					const NEWIMAGE::volume<float>& mask); 

  
  

  
  static NEWIMAGE::volume<float> MakeNoiseIma(const NEWIMAGE::volume<float>&   ima,     
					      float                            mu,      
					      float                            stdev);  

  
  static DiffStats GetSliceWiseStats(
				     const NEWIMAGE::volume<float>&                  pred,      
				     std::shared_ptr<const NEWIMAGE::volume<float> > susc,      
				     const NEWIMAGE::volume<float>&                  pmask,     
				     const NEWIMAGE::volume<float>&                  bmask,     
				     const EDDY::ECScan&                             scan);     

  
  static double MovParamUpdate(
			       const NEWIMAGE::volume<float>&                    pred,      
			       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
			       const NEWIMAGE::volume<float>&                    pmask,     
			       bool                                              cbs,       
                               float                                             fwhm,      
			       
			       EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,pmask,MOVEMENT,cbs,fwhm,0,0,0,scan,NULL));
  } EddyCatch

  
  static double ECParamUpdate(
			      const NEWIMAGE::volume<float>&                    pred,      
			      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
			      const NEWIMAGE::volume<float>&                    pmask,     
			      bool                                              cbs,       
                              float                                             fwhm,      
			      
			      EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,pmask,EC,cbs,fwhm,0,0,0,scan,NULL));
  } EddyCatch

  
  static double MovAndECParamUpdate(
				    const NEWIMAGE::volume<float>&                    pred,      
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
				    const NEWIMAGE::volume<float>&                    pmask,     
				    bool                                              cbs,       
				    float                                             fwhm,      
				    
				    EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,pmask,ALL,cbs,fwhm,0,0,0,scan,NULL));
  } EddyCatch

  
  static double MovAndECParamUpdate(
				    const NEWIMAGE::volume<float>&                    pred,      
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      
				    const NEWIMAGE::volume<float>&                    pmask,     
				    bool                                              cbs,       
				    float                                             fwhm,      
				    
				    unsigned int                                      scindx,    
				    unsigned int                                      iter,      
				    unsigned int                                      level,     
				    
				    EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,pmask,ALL,cbs,fwhm,scindx,iter,level,scan,NULL));
  } EddyCatch

  
  static NEWIMAGE::volume<float> TransformModelToScanSpace(
							   const NEWIMAGE::volume<float>&                    pred,
							   const EDDY::ECScan&                               scan,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   susc) EddyTry {
    NEWIMAGE::volume<float> mask(pred.xsize(),pred.ysize(),pred.zsize()); 
    NEWIMAGE::volume<float> jac(pred.xsize(),pred.ysize(),pred.zsize()); 
    return(transform_model_to_scan_space(pred,scan,susc,true,mask,&jac,NULL));
  } EddyCatch
  static NEWIMAGE::volume<float> TransformScanToModelSpace(
							   const EDDY::ECScan&                             scan,
							   std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							   
							   NEWIMAGE::volume<float>&                        omask);

  
  
  
  static NEWIMAGE::volume<float> DirectTransformScanToModelSpace(
								 const EDDY::ECScan&                             scan,
								 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								 
								 NEWIMAGE::volume<float>&                        omask);
  
  static NEWIMAGE::volume4D<float> DerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                             scan,
									   const NEWIMAGE::volume<float>&                  mima,
									   std::shared_ptr<const NEWIMAGE::volume<float> > susc)
  {
    return(EddyUtils::get_partial_derivatives_in_scan_space(mima,scan,susc,EDDY::ALL));
  }

  static NEWIMAGE::volume4D<float> DirectDerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
										 const NEWIMAGE::volume<float>&                    mima,
										 std::shared_ptr<const NEWIMAGE::volume<float> >   susc) EddyTry
  {
    return(EddyUtils::get_direct_partial_derivatives_in_scan_space(mima,scan,susc,EDDY::ALL));
  } EddyCatch
  
};











class FieldUtils
{
public:
  
  static NEWIMAGE::volume<float> RigidBodyTransformHzField(const NEWIMAGE::volume<float>& hzfield);

  
  static NEWIMAGE::volume4D<float> Hz2VoxelDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                         const AcqPara&                 acqp);
  static NEWIMAGE::volume4D<float> Hz2MMDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                      const AcqPara&                 acqp);
  static NEWIMAGE::volume4D<float> Voxel2MMDisplacements(const NEWIMAGE::volume4D<float>& voxdisp) EddyTry { 
    NEWIMAGE::volume4D<float> mmd=voxdisp; mmd[0] *= mmd.xdim(); mmd[1] *= mmd.ydim(); mmd[2] *= mmd.zdim(); return(mmd);
  } EddyCatch

  
  static NEWIMAGE::volume4D<float> Invert3DDisplacementField(
							     const NEWIMAGE::volume4D<float>& dfield,
							     const AcqPara&                   acqp,
							     const NEWIMAGE::volume<float>& inmask,
							     
							     NEWIMAGE::volume<float>&       omask);

  
  static NEWIMAGE::volume<float> Invert1DDisplacementField(
							   const NEWIMAGE::volume<float>& dfield,
							   const AcqPara&                 acqp,
							   const NEWIMAGE::volume<float>& inmask,
							   
							   NEWIMAGE::volume<float>&       omask);

  
  static NEWIMAGE::volume<float> GetJacobian(const NEWIMAGE::volume4D<float>& dfield,
                                             const AcqPara&                   acqp);

  
  static NEWIMAGE::volume<float> GetJacobianFrom1DField(const NEWIMAGE::volume<float>& dfield,
							unsigned int                   dir);
private:
};

 
class s2vQuant
{
public:
  s2vQuant(const ECScanManager&  sm) EddyTry : _sm(sm), _trth(0.3), _rotth(0.3) { common_construction(); } EddyCatch
  s2vQuant(const ECScanManager&  sm,
	   double                trth,
	   double                rotth) EddyTry : _sm(sm), _trth(trth), _rotth(rotth) { common_construction(); } EddyCatch
  
  std::vector<unsigned int> FindStillVolumes(ScanType st, const std::vector<unsigned int>& mbsp) const;
private:
  
  void common_construction();
  const ECScanManager&    _sm;     
  NEWMAT::Matrix          _tr;     
  NEWMAT::Matrix          _rot;    
  double                  _trth;   
  double                  _rotth;  
};

} 

#endif 

