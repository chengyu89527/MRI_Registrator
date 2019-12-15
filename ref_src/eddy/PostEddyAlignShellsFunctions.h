
#ifndef PostEddyAlignShellsFunctions_h
#define PostEddyAlignShellsFunctions_h

#include "EddyHelperClasses.h"
#include "ECScanClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {

 

class PEASUtils
{
public:
  
  static void PostEddyAlignShells(
				  const EddyCommandLineOptions&   clo,
				  bool                            upe, 
				  
				  ECScanManager&                  sm);
  
  static void PostEddyAlignShellsAlongPE(
					 const EddyCommandLineOptions&   clo,
					 bool                            upe, 
					 
					 ECScanManager&                  sm);
  
  static void WritePostEddyBetweenShellMIValues(
						const EddyCommandLineOptions&     clo,
						const ECScanManager&              sm,
						const std::vector<unsigned int>&  n,
						const std::vector<double>&        first,
						const std::vector<double>&        last,
						const std::string&                bfname);
private:
  static NEWIMAGE::volume<float> get_mean_scan(
					       const EddyCommandLineOptions&     clo,      
					       const ECScanManager&              sm,
					       const std::vector<unsigned int>&  indx,
					       
					       NEWIMAGE::volume<float>&          mask);

  static NEWMAT::ColumnVector register_volumes(
					       const NEWIMAGE::volume<float>& ref,
					       const NEWIMAGE::volume<float>& ima,
					       const NEWIMAGE::volume<float>& mask,
					       
					       NEWIMAGE::volume<float>&       rima);

  static NEWMAT::ColumnVector register_volumes_along_PE(
							const NEWIMAGE::volume<float>& ref,
							const NEWIMAGE::volume<float>& ima,
							const NEWIMAGE::volume<float>& mask,
							unsigned int                   pe_dir,
							
							NEWIMAGE::volume<float>&       rima);

  static std::vector<NEWMAT::ColumnVector> collate_mov_par_estimates_for_use(const std::vector<NEWMAT::ColumnVector>&                mp,
									     const std::vector<std::vector<NEWMAT::ColumnVector> >&  cmp,
									     const NEWIMAGE::volume<float>&                          ima);

  static void write_post_eddy_align_shells_report(const std::vector<NEWMAT::ColumnVector>&                mi_dmp,
						  const std::vector<NEWMAT::ColumnVector>&                mi_ump,
						  const std::vector<std::vector<NEWMAT::ColumnVector> >&  mi_cmp,
						  const std::vector<NEWMAT::ColumnVector>&                b0_dmp,
						  const std::vector<NEWMAT::ColumnVector>&                b0_ump,
						  const std::vector<double>&                              grpb,
						  bool                                                    upe,
						  const EddyCommandLineOptions&                           clo);

  static void write_post_eddy_align_shells_along_PE_report(const std::vector<NEWMAT::ColumnVector>&                mi_dmp,
							   const std::vector<NEWMAT::ColumnVector>&                mi_ump,
							   const std::vector<std::vector<NEWMAT::ColumnVector> >&  mi_cmp,
							   const std::vector<double>&                              grpb,
							   bool                                                    upe,
							   const EddyCommandLineOptions&                           clo);

  static void update_mov_par_estimates(
				       const NEWMAT::ColumnVector&       mp,
				       const std::vector<unsigned int>&  indx,
				       
				       ECScanManager&                    sm);

  static void align_shells_using_MI(
				    const EddyCommandLineOptions&                      clo,
				    bool                                               pe_only,
				    
				    ECScanManager&                                     sm,
				    
				    std::vector<double>&                               grpb,
				    std::vector<NEWMAT::ColumnVector>&                 mov_par,
				    std::vector<std::vector<NEWMAT::ColumnVector> >&   cmov_par,
				    std::vector<NEWMAT::ColumnVector>&                 mp_for_updates);

  static void write_between_shell_MI_values(const NEWIMAGE::volume<float>&    b0_mean,
					    const NEWIMAGE::volume<float>&    dwi_means,
					    const NEWIMAGE::volume<float>&    mask,
					    const std::string&                fname,
					    const std::vector<unsigned int>&  n,
					    const std::vector<double>&        first,
					    const std::vector<double>&        last);

  static void align_shells_using_interspersed_B0_scans(
						       const EddyCommandLineOptions&                      clo,
						       
						       ECScanManager&                                     sm,
						       
						       std::vector<double>&                               grpb,
						       std::vector<NEWMAT::ColumnVector>&                 mov_par,
						       std::vector<NEWMAT::ColumnVector>&                 mp_for_updates);
}; 

} 

#endif 

