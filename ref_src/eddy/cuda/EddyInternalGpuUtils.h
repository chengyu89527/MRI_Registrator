/////////////////////////////////////////////////////////////////////
///
/// \file EddyInternalGpuUtils.h
/// \brief Declarations of static class with collection of GPU routines used in the eddy project
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///


#ifndef EddyInternalGpuUtils_h
#define EddyInternalGpuUtils_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#include "CudaVolume.h"
#pragma pop
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {

class EddyGpuUtils; 








class EddyInternalGpuUtils
{
private:
  friend class EddyGpuUtils;
  friend class PostEddyCFImpl;

  static const int threads_per_block_make_deriv = 128;
  static const int threads_per_block_ec_field = 128;

  
  static void load_prediction_maker(
				    const EddyCommandLineOptions&        clo,
				    ScanType                             st,
				    const ECScanManager&                 sm,
				    unsigned int                         iter,
				    float                                fwhm,
		                    bool                                 use_orig,
				    const EDDY::PolationPara&            pp,
				    
				    std::shared_ptr<DWIPredictionMaker>  pmp,
				    NEWIMAGE::volume<float>&             mask);

  

  
  static void get_motion_corrected_scan(
					const EDDY::ECScan&     scan,
					bool                    use_orig,
					
					EDDY::CudaVolume&       oima,
					
					EDDY::CudaVolume&       omask);

  
  static void get_unwarped_scan(
				const EDDY::ECScan&        scan,
				const EDDY::CudaVolume&    susc,
				const EDDY::CudaVolume&    pred,
				bool                       jacmod,
				bool                       use_orig,
				const EDDY::PolationPara&  pp,
				
				EDDY::CudaVolume&          oima,
				EDDY::CudaVolume&          omask);

  
  
  
  static void get_volumetric_unwarped_scan(
					   const EDDY::ECScan&        scan,
					   const EDDY::CudaVolume&    susc,
					   bool                       jacmod,
					   bool                       use_orig,
					   const EDDY::PolationPara&  pp,
					   
					   EDDY::CudaVolume&          oima,
					   EDDY::CudaVolume&          omask,
					   EDDY::CudaVolume4D&        deriv);

  static void detect_outliers(
			      const EddyCommandLineOptions&             clo,
			      ScanType                                  st,
			      const std::shared_ptr<DWIPredictionMaker> pmp,
			      const NEWIMAGE::volume<float>&            mask,
			      const ECScanManager&                      sm,
			      
			      ReplacementManager&                       rm,
			      DiffStatsVector&                          dsv);

  static void replace_outliers(
			       const EddyCommandLineOptions&             clo,
			       ScanType                                  st,
			       const std::shared_ptr<DWIPredictionMaker> pmp,
			       const NEWIMAGE::volume<float>&            mask,
			       const ReplacementManager&                 rm,
			       bool                                      add_noise,
			       
			       ECScanManager&                            sm);

  
  static void field_for_scan_to_model_transform(
						const EDDY::ECScan&     scan,
						const EDDY::CudaVolume& susc,
						
						EDDY::CudaVolume4D&     dfield,
						EDDY::CudaVolume&       omask,
						EDDY::CudaVolume&       jac);

  
  static void field_for_scan_to_model_volumetric_transform(
							   const EDDY::ECScan&            scan,
							   const EDDY::CudaVolume&        susc,
							   
							   EDDY::CudaVolume4D&            dfield,
							   
							   EDDY::CudaVolume&              omask,
							   EDDY::CudaVolume&              jac);

  
  static double param_update(
			     const NEWIMAGE::volume<float>&                  pred,     
			     std::shared_ptr<const NEWIMAGE::volume<float> > susc,     
			     const NEWIMAGE::volume<float>&                  pmask, 
			     EDDY::Parameters                                whichp,   
			     bool                                            cbs,      
			     float                                           fwhm,
			     const EDDY::PolationPara&                       pp,
			     
			     unsigned int                                    scindex,
			     unsigned int                                    iter,
			     unsigned int                                    level,
			     
			     EDDY::ECScan&                                   scan,     
			     
			     NEWMAT::ColumnVector                            *rupdate);

  
  static void transform_model_to_scan_space(
					    const EDDY::CudaVolume&     pred,
					    const EDDY::ECScan&         scan,
					    const EDDY::CudaVolume&     susc,
					    bool                        jacmod,
					    
					    EDDY::CudaVolume&           oima,
					    EDDY::CudaVolume&           omask,
					    
					    EDDY::CudaVolume&           jac,
					    EDDY::CudaVolume4D&         grad);

  
  static void field_for_model_to_scan_transform(
						const EDDY::ECScan&       scan,
						const EDDY::CudaVolume&   susc,
						
						EDDY::CudaVolume4D&       dfield,
						EDDY::CudaVolume&         omask,
						
						EDDY::CudaVolume&         jac);

  
  static EDDY::CudaImageCoordinates transform_coordinates_from_model_to_scan_space(
										   const EDDY::CudaVolume&     pred,
										   const EDDY::ECScan&         scan,
										   const EDDY::CudaVolume&     susc,
										   
										   EDDY::CudaImageCoordinates& coord,
										   
										   EDDY::CudaVolume&           omask,
										   EDDY::CudaVolume&           jac);

  
  static void get_partial_derivatives_in_scan_space(const EDDY::CudaVolume&  pred,
						    const EDDY::ECScan&      scan,
						    const EDDY::CudaVolume&  susc,
						    EDDY::Parameters         whichp,
						    EDDY::CudaVolume4D&      derivs);

  static void get_direct_partial_derivatives_in_scan_space(const EDDY::CudaVolume& pred,
							   const EDDY::ECScan&     scan,
							   const EDDY::CudaVolume& susc,
							   EDDY::Parameters        whichp,
							   EDDY::CudaVolume4D&     derivs);

  static void make_deriv_from_components(const EDDY::CudaImageCoordinates&  coord,
					 const EDDY::CudaVolume4D&          grad,
					 const EDDY::CudaVolume&            base,
					 const EDDY::CudaVolume&            jac,
					 const EDDY::CudaVolume&            basejac,
					 float                              dstep,
					 EDDY::CudaVolume4D&                deriv,
					 unsigned int                       indx);

  static NEWMAT::Matrix make_XtX(const EDDY::CudaVolume4D&  X,
				 const EDDY::CudaVolume&    mask);

  static NEWMAT::ColumnVector make_Xty(const EDDY::CudaVolume4D&  X,
				       const EDDY::CudaVolume&    y,
				       const EDDY::CudaVolume&    mask);

  static void general_transform(const EDDY::CudaVolume&    inima,
				const NEWMAT::Matrix&      A,
				const EDDY::CudaVolume4D&  dfield,
				const NEWMAT::Matrix&      M,
				EDDY::CudaVolume&          oima,
				EDDY::CudaVolume4D&        deriv,
				EDDY::CudaVolume&          omask);

  static void general_transform(
				const EDDY::CudaVolume&             inima,
				const std::vector<NEWMAT::Matrix>&  A,
				const EDDY::CudaVolume4D&           dfield,
				const std::vector<NEWMAT::Matrix>&  M,
				
				EDDY::CudaVolume&                   oima,
				
				EDDY::CudaVolume4D&                 deriv,
				EDDY::CudaVolume&                   omask);

  static void general_slice_to_vol_transform(
					     const EDDY::CudaVolume&             inima,
					     const std::vector<NEWMAT::Matrix>&  A,
					     const EDDY::CudaVolume4D&           dfield,
					     const EDDY::CudaVolume&             jac,
					     const EDDY::CudaVolume&             pred,
					     bool                                jacmod,
					     NEWIMAGE::interpolation             interp,
					     
					     EDDY::CudaVolume&                   oima,
					     
					     EDDY::CudaVolume&                   omask);

  static void affine_transform(const EDDY::CudaVolume&    inima,
			       const NEWMAT::Matrix&      R,
			       EDDY::CudaVolume&          oima,
			       EDDY::CudaVolume4D&        deriv,
			       EDDY::CudaVolume&          omask);

  static void affine_transform(const EDDY::CudaVolume&             inima,
			       const std::vector<NEWMAT::Matrix>&  R,
			       EDDY::CudaVolume&                   oima,
			       EDDY::CudaVolume4D&                 deriv,
			       EDDY::CudaVolume&                   omask);

  
  static void get_ec_field(
			   const EDDY::ECScan&        scan,
			   
			   EDDY::CudaVolume&          ecfield);

};








class FieldGpuUtils
{
private:
  friend class EddyInternalGpuUtils;  

  static const int threads_per_block_invert_field = 128;

  static void Hz2VoxelDisplacements(const EDDY::CudaVolume&  hzfield,
				    const EDDY::AcqPara&     acqp,
				    EDDY::CudaVolume4D&      dfield);

  static void Voxel2MMDisplacements(EDDY::CudaVolume4D&      dfield);

  static void InvertDisplacementField(
				      const EDDY::CudaVolume4D&  dfield,
				      const EDDY::AcqPara&       acqp,
				      const EDDY::CudaVolume&    inmask,
				      
				      EDDY::CudaVolume4D&        idfield,
				      EDDY::CudaVolume&          omask);
    
  static void GetJacobian(
			  const EDDY::CudaVolume4D&  dfield,
			  const EDDY::AcqPara&       acqp,
			  
			  EDDY::CudaVolume&          jac);

};

} 

#endif 

