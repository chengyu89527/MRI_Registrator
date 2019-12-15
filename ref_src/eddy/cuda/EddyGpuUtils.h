/////////////////////////////////////////////////////////////////////
///
/// \file EddyGpuUtils.h
/// \brief Declarations of static class with collection of GPU routines used in the eddy project
///
/// The routines declared here are "bridges" on to the actual GPU
/// routines. The interface to these routines only display classes
/// that are part of the "regular" FSL libraries. Hence this file
/// can be safely included by files that know nothing of the GPU
/// and that are compiled by gcc.
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///


#ifndef EddyGpuUtils_h
#define EddyGpuUtils_h

#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "ECScanClasses.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {








class EddyGpuUtils
{
public:

  
  static void InitGpu(bool verbose=true);

  
  static std::shared_ptr<DWIPredictionMaker> LoadPredictionMaker(
								 const EddyCommandLineOptions& clo,
							         ScanType                      st,
								 const ECScanManager&          sm,
								 unsigned int                  iter,
								 float                         fwhm,
								 
								 NEWIMAGE::volume<float>&      mask,
								 
								 bool                          use_orig=false);

  

  
  static NEWIMAGE::volume<float> GetUnwarpedScan(
						 const EDDY::ECScan&                               scan,
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 bool                                              use_orig,
						 const EDDY::PolationPara&                         pp,
						 
						 NEWIMAGE::volume<float>                           *omask=NULL);

  
  static NEWIMAGE::volume<float> GetUnwarpedScan(
						 const EDDY::ECScan&                               scan,
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 const NEWIMAGE::volume<float>&                    pred,
						 bool                                              use_orig,
						 const EDDY::PolationPara&                         pp,
						 
						 NEWIMAGE::volume<float>                           *omask=NULL);

  
  static NEWIMAGE::volume<float> GetVolumetricUnwarpedScan(
							   const EDDY::ECScan&                               scan,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							   bool                                              use_orig,
							   const EDDY::PolationPara&                         pp,
							   
							   NEWIMAGE::volume<float>                           *omask=nullptr,
							   NEWIMAGE::volume4D<float>                         *deriv=nullptr);

  
  static void GetMotionCorrectedScan(
				     const EDDY::ECScan&       scan,
				     bool                      use_orig,
				     
				     NEWIMAGE::volume<float>&  ovol,
				     
				     NEWIMAGE::volume<float>   *omask=NULL);

  
  static NEWIMAGE::volume<float> TransformModelToScanSpace(const EDDY::ECScan&                               scan,
							   const NEWIMAGE::volume<float>&                    mima,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							   bool                                              jacmod=true);

  static NEWIMAGE::volume4D<float> DerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
									   const NEWIMAGE::volume<float>&                    mima,
									   std::shared_ptr<const NEWIMAGE::volume<float> >   susc);

  static NEWIMAGE::volume4D<float> DirectDerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
										 const NEWIMAGE::volume<float>&                    mima,
										 std::shared_ptr<const NEWIMAGE::volume<float> >   susc);

  
  static NEWIMAGE::volume<float> Smooth(const NEWIMAGE::volume<float>&  ima,
					float                           fwhm);

  
  static DiffStatsVector DetectOutliers(
					const EddyCommandLineOptions&             clo,
					ScanType                                  st,
					const std::shared_ptr<DWIPredictionMaker> pmp,
					const NEWIMAGE::volume<float>&            mask,
					const ECScanManager&                      sm,
					
					ReplacementManager&                       rm);

  
  static void ReplaceOutliers(
			      const EddyCommandLineOptions&             clo,
			      ScanType                                  st,
			      const std::shared_ptr<DWIPredictionMaker> pmp,
			      const NEWIMAGE::volume<float>&            mask,
			      const ReplacementManager&                 rm,
			      bool                                      add_noise,
			      
			      ECScanManager&                            sm);

  
  static double MovAndECParamUpdate(
				    const NEWIMAGE::volume<float>&                    pred,
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
				    const NEWIMAGE::volume<float>&                    pmask,
				    bool                                              cbs,
				    float                                             fwhm,
				    const EDDY::PolationPara&                         pp,
				    
				    EDDY::ECScan&                                     scan);

  
  static double MovAndECParamUpdate(
				    const NEWIMAGE::volume<float>&                    pred,
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
				    const NEWIMAGE::volume<float>&                    pmask,
				    bool                                              cbs,
				    float                                             fwhm,
				    const EDDY::PolationPara&                         pp,
				    
				    unsigned int                                      scindex,
				    unsigned int                                      iter,
				    unsigned int                                      level,
				    
				    EDDY::ECScan&                                     scan);
};

} 

#endif 

