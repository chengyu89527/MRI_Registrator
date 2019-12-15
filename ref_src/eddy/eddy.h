/////////////////////////////////////////////////////////////////////
///
/// \file eddy.h
/// \brief Contains declarations of some very high level functions for eddy
///
/// This file contains declarations for some very high level functions
/// that are called in eddy.cpp. 
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///


#ifndef eddy_h
#define eddy_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {


ReplacementManager *DoVolumeToVolumeRegistration(
						 const EddyCommandLineOptions&  clo,     
						 
						 ECScanManager&                 sm);


ReplacementManager *DoSliceToVolumeRegistration(
						const EddyCommandLineOptions&  clo,    
						unsigned int                   oi,        
						bool                           dol,       
						
						ECScanManager&                 sm,
						ReplacementManager             *dwi_rm);  


    
  

ReplacementManager *Register(
			     const EddyCommandLineOptions&  clo,     
			     ScanType                       st,      
			     unsigned int                   niter,
			     const std::vector<float>&      fwhm,
			     SecondLevelECModel             slm,
			     bool                           dol,
			     
			     ECScanManager&                 sm,
			     ReplacementManager             *rm,
			     
			     NEWMAT::Matrix&                msshist, 
			     NEWMAT::Matrix&                phist);


ReplacementManager *FinalOLCheck(
				 const EddyCommandLineOptions&  clo,
				 
				 ReplacementManager             *rm,
				 ECScanManager&                 sm);


DiffStatsVector DetectAndReplaceOutliers(
					 const EddyCommandLineOptions& clo,
					 ScanType                      st,
					 
					 ECScanManager&                sm,
					 ReplacementManager&           rm);


std::shared_ptr<DWIPredictionMaker> LoadPredictionMaker(
							const EddyCommandLineOptions& clo,
							ScanType                      st,
							const ECScanManager&          sm,
							unsigned int                  iter,
							float                         fwhm,
							
							NEWIMAGE::volume<float>&      mask,
							
							bool                          use_orig=false);




DiffStatsVector DetectOutliers(
			       const EddyCommandLineOptions&               clo,
			       ScanType                                    st,
			       const std::shared_ptr<DWIPredictionMaker>   pmp,
			       const NEWIMAGE::volume<float>&              mask,
			       const ECScanManager&                        sm,
			       
			       ReplacementManager&                         rm);


void ReplaceOutliers(
		     const EddyCommandLineOptions&               clo,
		     ScanType                                    st,
		     const std::shared_ptr<DWIPredictionMaker>   pmp,
		     const NEWIMAGE::volume<float>&              mask,
		     const ReplacementManager&                   rm,
		     bool                                        add_noise,
		     
		     ECScanManager&                              sm);


void GetPredictionsForResampling(
				 const EddyCommandLineOptions&    clo,
				 ScanType                         st,
				 const ECScanManager&             sm,
				 
				 NEWIMAGE::volume4D<float>&       pred);


void WriteCNRMaps(
		  const EddyCommandLineOptions&   clo,
		  const ECScanManager&            sm,
		  const std::string&              spatial_fname,
		  const std::string&              range_fname,
		  const std::string&              temporal_fname);


void Diagnostics(
		 const EddyCommandLineOptions&  clo,      
		 unsigned int                   iter,     
		 ScanType                       st,       
		 const ECScanManager&           sm,       
                 const double                   *mss_tmp, 
                 const DiffStatsVector&         stats,    
		 const ReplacementManager&      rm,       
		 
		 NEWMAT::Matrix&                mss,      
		 NEWMAT::Matrix&                phist);


void AddRotation(ECScanManager&               sm,
		 const NEWMAT::ColumnVector&  rp);


void PrintMIValues(const EddyCommandLineOptions&  clo,      
                   const ECScanManager&           sm,
                   const std::string&             fname,
                   bool                           write_planes);

} 


#endif 

