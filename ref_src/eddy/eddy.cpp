
#pragma GCC diagnostic ignored "-Wunknown-pragmas" 
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "utils/stack_dump.h"
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "EddyCommandLineOptions.h"
#include "PostEddyCF.h"
#include "PostEddyAlignShellsFunctions.h"
#include "MoveBySuscCF.h"

#include "eddy.h"
#ifdef COMPILE_GPU
#include "cuda/GpuPredictorChunk.h"
#include "cuda/EddyGpuUtils.h"
#endif

using namespace EDDY;

int main(int argc, char *argv[]) try
{
  StackDump::Install(); 

  
  EddyCommandLineOptions clo(argc,argv); 

  
  if (clo.Verbose()) cout << "Reading images" << endl;
  ECScanManager sm(clo.ImaFname(),clo.MaskFname(),clo.AcqpFname(),clo.TopupFname(),clo.FieldFname(),clo.FieldMatFname(),
                   clo.BVecsFname(),clo.BValsFname(),clo.FirstLevelModel(),clo.b0_FirstLevelModel(),clo.Indicies(),
		   clo.PolationParameters(),clo.MultiBand(),clo.DontCheckShelling()); 
  if (clo.FillEmptyPlanes()) { if (clo.Verbose()) cout << "Filling empty planes" << endl; sm.FillEmptyPlanes(); }
  if (clo.ResamplingMethod() == LSR) {
    if (!sm.CanDoLSRResampling()) throw EddyException("These data do not support least-squares resampling");
  }
  if (clo.UseB0sToAlignShellsPostEddy() && !sm.B0sAreUsefulForPEAS()) {
    throw EddyException("These data do not support using b0s for Post Eddy Alignment of Shells");
  }
  if (clo.RefScanNumber()) sm.SetLocationReference(clo.RefScanNumber());


  
  if (clo.DebugLevel() && sm.HasSuscHzOffResField()) {
    std::string fname = "EDDY_DEBUG_susc_00_0000"; NEWIMAGE::write_volume(*(sm.GetSuscHzOffResField()),fname);
  }

  
  if (clo.InitFname() != std::string("")) {
    if (clo.RegisterDWI() && clo.Registerb0()) sm.SetParameters(clo.InitFname(),ANY);
    else if (clo.RegisterDWI()) sm.SetParameters(clo.InitFname(),DWI);
    else sm.SetParameters(clo.InitFname(),B0);
  }

  
  

  
  
  
  

  
  
  

  if (clo.Verbose()) cout << "Performing volume-to-volume registration" << endl;
  ReplacementManager *dwi_rm=NULL;
  if (clo.EstimateMoveBySusc()) { 
    EDDY::SecondLevelECModel b0_slm = clo.b0_SecondLevelModel();
    EDDY::SecondLevelECModel dwi_slm = clo.SecondLevelModel();
    cout << "Setting linear second level model" << endl;
    clo.Set_b0_SecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
    clo.SetSecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
    dwi_rm = DoVolumeToVolumeRegistration(clo,sm);
    cout << "Resetting second level model" << endl;
    clo.Set_b0_SecondLevelModel(b0_slm);
    clo.SetSecondLevelModel(dwi_slm);
  }
  else dwi_rm = DoVolumeToVolumeRegistration(clo,sm);
  sm.ApplyLocationReference();
  
  
  
  

  
  if (clo.PrintMIValues()) {
    if (clo.Verbose()) cout << "Writing MI values between shells" << endl;
    PrintMIValues(clo,sm,clo.MIPrintFname(),clo.PrintMIPlanes()); 
  }

  
  

  
  
  if (clo.RegisterDWI()) {
    if (!clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
      if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
      PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
      if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
      PEASUtils::PostEddyAlignShells(clo,false,sm);    
    }
    else if (clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
      if (clo.Verbose()) cout << "Aligning shells along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
      PEASUtils::PostEddyAlignShellsAlongPE(clo,true,sm);
      if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
      PEASUtils::PostEddyAlignShells(clo,false,sm);    
    }
    else if (clo.AlignShellsPostEddy()) {
      if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
      PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
      if (clo.Verbose()) cout << "Aligning shells (running PostEddyAlignShells)" << endl;
      PEASUtils::PostEddyAlignShells(clo,true,sm);
    }
  }

  
  
  

  
  
  
  
  
  
  

  
  
  
  if (clo.IsSliceToVol()) {
    if (clo.Verbose()) cout << "Performing slice-to-volume registration" << endl;
    if (clo.EstimateMoveBySusc()) { 
      EDDY::SecondLevelECModel b0_slm = clo.b0_SecondLevelModel();
      EDDY::SecondLevelECModel dwi_slm = clo.SecondLevelModel();
      clo.Set_b0_SecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
      clo.SetSecondLevelModel(EDDY::Linear_2nd_lvl_mdl);
      for (unsigned int i=0; i<clo.NumOfNonZeroMovementModelOrder(); i++) {
	if (clo.Verbose()) cout << "Setting slice-to-volume order to " << clo.MovementModelOrder(i) << endl;
	sm.SetMovementModelOrder(clo.MovementModelOrder(i));
	sm.Set_S2V_Lambda(clo.S2V_Lambda(i));
	dwi_rm = DoSliceToVolumeRegistration(clo,i,false,sm,dwi_rm);
	sm.ApplyLocationReference();
      }
      clo.Set_b0_SecondLevelModel(b0_slm);
      clo.SetSecondLevelModel(dwi_slm);
    }
    else {
      for (unsigned int i=0; i<clo.NumOfNonZeroMovementModelOrder(); i++) {
	if (clo.Verbose()) cout << "Setting slice-to-volume order to " << clo.MovementModelOrder(i) << endl;
	sm.SetMovementModelOrder(clo.MovementModelOrder(i));
	sm.Set_S2V_Lambda(clo.S2V_Lambda(i));
	dwi_rm = DoSliceToVolumeRegistration(clo,i,false,sm,dwi_rm);
	sm.ApplyLocationReference();
      }
    }

    
    if (clo.RegisterDWI()) {
      if (!clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
	if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
	PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
	if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
	PEASUtils::PostEddyAlignShells(clo,false,sm);    
      }
      else if (clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
	if (clo.Verbose()) cout << "Aligning shells along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
	PEASUtils::PostEddyAlignShellsAlongPE(clo,true,sm);
	if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
	PEASUtils::PostEddyAlignShells(clo,false,sm);    
      }
      else if (clo.AlignShellsPostEddy()) {
	if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
	PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
	if (clo.Verbose()) cout << "Aligning shells (running PostEddyAlignShells)" << endl;
	PEASUtils::PostEddyAlignShells(clo,true,sm);
      }
    }
  }

  
  
  
  if (clo.EstimateMoveBySusc()) {

    
    

    sm.SetUseB0sToInformDWIRegistration(false); 
    std::vector<unsigned int> b0s;
    std::vector<unsigned int> dwis;
    if (clo.IsSliceToVol()) {           
      EDDY::s2vQuant s2vq(sm,1.0,1.0);  
      if (clo.Registerb0()) b0s = s2vq.FindStillVolumes(B0,clo.MoveBySuscParam());
      if (clo.RegisterDWI()) dwis = s2vq.FindStillVolumes(DWI,clo.MoveBySuscParam());
    }
    else { 
      if (clo.Registerb0()) { b0s.resize(sm.NScans(B0)); for (unsigned int i=0; i<sm.NScans(B0); i++) b0s[i] = i; }
      if (clo.RegisterDWI()) { dwis.resize(sm.NScans(DWI)); for (unsigned int i=0; i<sm.NScans(DWI); i++) dwis[i] = i; }
    }
    if (clo.RegisterDWI()) { 
      unsigned int mbs_niter = (clo.MoveBySuscNiter() / clo.N_MBS_Interleaves()) + 1;
      unsigned int niter, s2vi=0;
      if (clo.IsSliceToVol()) {
	s2vi = clo.NumOfNonZeroMovementModelOrder()-1;
	niter = clo.S2V_NIter(s2vi);
	sm.SetMovementModelOrder(clo.MovementModelOrder(s2vi));
	sm.Set_S2V_Lambda(clo.S2V_Lambda(s2vi));
	clo.SetS2VParam(clo.MovementModelOrder(s2vi),clo.S2V_Lambda(s2vi),0.0,(niter/clo.N_MBS_Interleaves())+1);
      }
      else { niter = clo.NIter(); clo.SetNIterAndFWHM((niter/clo.N_MBS_Interleaves())+1,std::vector<float>(1,0.0)); }
      NEWMAT::ColumnVector spar;
      EDDY::MoveBySuscCF cf(sm,clo,b0s,dwis,clo.MoveBySuscParam(),clo.MoveBySuscOrder(),clo.MoveBySuscKsp());

      

      for (unsigned int i=0; i<clo.N_MBS_Interleaves(); i++) {
	if (clo.Verbose()) cout << "Running interleave " << i+1 << " of MBS" << endl;
	if (!i) spar = cf.Par(); 
	cf.SetLambda(clo.MoveBySuscLambda());
	MISCMATHS::NonlinParam nlp(cf.NPar(),MISCMATHS::NL_LM,spar);
	nlp.SetMaxIter(mbs_niter);

	
	

	MISCMATHS::nonlin(nlp,cf);

	
	
	

	spar = cf.Par(); 
	if (clo.IsSliceToVol()) {
	  if (clo.Verbose()) cout << "Running slice-to-vol interleaved with MBS" << endl;
	  dwi_rm = DoSliceToVolumeRegistration(clo,s2vi,false,sm,dwi_rm);
	  sm.ApplyLocationReference();
	}
	else {
	  if (clo.Verbose()) cout << "Running vol-to-vol interleaved with MBS" << endl;

	  
	  

	  dwi_rm = DoVolumeToVolumeRegistration(clo,sm);

	  
	  
	  

	  sm.ApplyLocationReference();
	}
      }
      cf.WriteFirstOrderFields(clo.MoveBySuscFirstOrderFname());
      if (clo.MoveBySuscOrder() > 1) cf.WriteSecondOrderFields(clo.MoveBySuscSecondOrderFname());
    }
    else { 
      
      EDDY::MoveBySuscCF cf(sm,clo,b0s,dwis,clo.MoveBySuscParam(),clo.MoveBySuscOrder(),clo.MoveBySuscKsp());
      NEWMAT::ColumnVector spar = cf.Par(); 
      cf.SetLambda(clo.MoveBySuscLambda());
      MISCMATHS::NonlinParam nlp(cf.NPar(),MISCMATHS::NL_LM,spar);
      nlp.SetMaxIter(clo.MoveBySuscNiter());
      MISCMATHS::nonlin(nlp,cf);            
      cf.WriteFirstOrderFields(clo.MoveBySuscFirstOrderFname());
      if (clo.MoveBySuscOrder() > 1) cf.WriteSecondOrderFields(clo.MoveBySuscSecondOrderFname());
    }

    
    
    

    
    

    
    if (clo.RegisterDWI()) {
      if (!clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
	if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
	PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
	if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
	PEASUtils::PostEddyAlignShells(clo,false,sm);    
      }
      else if (clo.SeparateOffsetFromMovement() && !clo.AlignShellsPostEddy()) {
	if (clo.Verbose()) cout << "Aligning shells along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
	PEASUtils::PostEddyAlignShellsAlongPE(clo,true,sm);
	if (clo.Verbose()) cout << "Checking shell alignment (running PostEddyAlignShells)" << endl;
	PEASUtils::PostEddyAlignShells(clo,false,sm);    
      }
      else if (clo.AlignShellsPostEddy()) {
	if (clo.Verbose()) cout << "Checking shell alignment along PE-direction (running PostEddyAlignShellsAlongPE)" << endl;
	PEASUtils::PostEddyAlignShellsAlongPE(clo,false,sm);
	if (clo.Verbose()) cout << "Aligning shells (running PostEddyAlignShells)" << endl;
	PEASUtils::PostEddyAlignShells(clo,true,sm);
      }
    }

    
    
    

  }

  
  
  
  
  if (clo.RegisterDWI()) {
    if (clo.Verbose()) cout << "Performing final outlier check" << endl;
    double old_hypar_ff = 1.0;
    if (clo.HyParFudgeFactor() != 1.0) { old_hypar_ff = clo.HyParFudgeFactor(); clo.SetHyParFudgeFactor(1.0); }
    dwi_rm = FinalOLCheck(clo,dwi_rm,sm);
    if (old_hypar_ff != 1.0) clo.SetHyParFudgeFactor(old_hypar_ff);
    
    std::vector<unsigned int> i2i = sm.GetDwi2GlobalIndexMapping();
    dwi_rm->WriteReport(i2i,clo.OLReportFname());
    dwi_rm->WriteMatrixReport(i2i,sm.NScans(),clo.OLMapReportFname(),clo.OLNStDevMapReportFname(),clo.OLNSqrStDevMapReportFname());
    if (clo.WriteOutlierFreeData()) {
      if (clo.Verbose()) cout << "Running sm.WriteOutlierFreeData" << endl;
      sm.WriteOutlierFreeData(clo.OLFreeDataFname());
    }
  }

  
  
  if (clo.DoTestRot()) {
    if (clo.Verbose()) cout << "Running sm.AddRotation" << endl;
    sm.AddRotation(clo.TestRotAngles());
  }


  
  if (clo.Verbose()) cout << "Running sm.WriteParameterFile" << endl;
  if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteParameterFile(clo.ParOutFname());
  else if (clo.RegisterDWI()) sm.WriteParameterFile(clo.ParOutFname(),DWI);
  else sm.WriteParameterFile(clo.ParOutFname(),B0);

  
  if (sm.IsSliceToVol()) {
    if (clo.Verbose()) cout << "Running sm.WriteMovementOverTimeFile" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteMovementOverTimeFile(clo.MovementOverTimeOutFname());
    else if (clo.RegisterDWI()) sm.WriteMovementOverTimeFile(clo.MovementOverTimeOutFname(),DWI);
    else sm.WriteMovementOverTimeFile(clo.MovementOverTimeOutFname(),B0);    
  }

  
  if (clo.Verbose()) cout << "Running sm.WriteRegisteredImages" << endl;
  if (!clo.ReplaceOutliers()) { if (clo.Verbose()) { cout << "Running sm.RecycleOutliers" << endl; } sm.RecycleOutliers(); } 
  if (sm.IsSliceToVol()) { 
    NEWIMAGE::volume4D<float> pred;
    ScanType st; 
    if (clo.RegisterDWI() && clo.Registerb0()) st=ANY; else if (clo.RegisterDWI()) st=DWI; else st=B0;
    GetPredictionsForResampling(clo,st,sm,pred);
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred);
    else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,DWI);
    else sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,B0);
  }
  else {
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput());
    else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),DWI);
    else sm.WriteRegisteredImages(clo.IOutFname(),clo.OutMaskFname(),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),B0);
  }

  
  
  if (clo.ReplaceOutliers() && clo.WriteAdditionalResultsWithOutliersRetained()) {
    if (clo.Verbose()) cout << "Running sm.WriteRegisteredImages" << endl;
    if (clo.Verbose()) { cout << "Running sm.RecycleOutliers" << endl; } 
    sm.RecycleOutliers(); 
    if (sm.IsSliceToVol()) { 
      NEWIMAGE::volume4D<float> pred;
      ScanType st; 
      if (clo.RegisterDWI() && clo.Registerb0()) st=ANY; else if (clo.RegisterDWI()) st=DWI; else st=B0;
      GetPredictionsForResampling(clo,st,sm,pred);
      if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred);
      else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,DWI);
      else sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),pred,B0);
    }
    else {
      if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput());
      else if (clo.RegisterDWI()) sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),DWI);
      else sm.WriteRegisteredImages(clo.AdditionalWithOutliersOutFname(),std::string(""),clo.ResamplingMethod(),clo.LSResamplingLambda(),clo.MaskOutput(),B0);
    }
  }

  
  if (clo.WriteFields()) {
    if (clo.Verbose()) cout << "Running sm.WriteECFields" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteECFields(clo.ECFOutFname());
    else if (clo.RegisterDWI()) sm.WriteECFields(clo.ECFOutFname(),DWI);
    else sm.WriteECFields(clo.ECFOutFname(),B0);
  }

  
  if (clo.WriteRotatedBVecs()) {
    if (clo.Verbose()) cout << "Running sm.WriteRotatedBVecs" << endl;
    sm.WriteRotatedBVecs(clo.RotatedBVecsOutFname());
  }

  
  if (clo.WriteMovementRMS()) {
    if (clo.Verbose()) cout << "Running sm.WriteMovementRMS" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) { sm.WriteMovementRMS(clo.RMSOutFname()); sm.WriteRestrictedMovementRMS(clo.RestrictedRMSOutFname()); }
    else if (clo.RegisterDWI()) { sm.WriteMovementRMS(clo.RMSOutFname(),DWI); sm.WriteRestrictedMovementRMS(clo.RestrictedRMSOutFname(),DWI); }
    else { sm.WriteMovementRMS(clo.RMSOutFname(),B0); sm.WriteRestrictedMovementRMS(clo.RestrictedRMSOutFname(),B0); }   
  }

  
  if (clo.WriteCNRMaps() || clo.WriteRangeCNRMaps() || clo.WriteResiduals()) {
    double old_hypar_ff = 1.0;
    if (clo.HyParFudgeFactor() != 1.0) { old_hypar_ff = clo.HyParFudgeFactor(); clo.SetHyParFudgeFactor(1.0); }
    if (clo.Verbose()) cout << "Running EDDY::WriteCNRMaps" << endl;
    WriteCNRMaps(clo,sm,clo.CNROutFname(),clo.RangeCNROutFname(),clo.ResidualsOutFname());
    if (old_hypar_ff != 1.0) clo.SetHyParFudgeFactor(old_hypar_ff);
  }

  
  if (clo.WriteDisplacementFields()) {
    if (clo.Verbose()) cout << "Running sm.WriteDisplacementFields" << endl;
    if (clo.RegisterDWI() && clo.Registerb0()) sm.WriteDisplacementFields(clo.DFieldOutFname());
    else if (clo.RegisterDWI()) sm.WriteDisplacementFields(clo.DFieldOutFname(),DWI);
    else sm.WriteDisplacementFields(clo.DFieldOutFname(),B0);    
  }

  exit(EXIT_SUCCESS);
}
 catch(const std::exception& e) 
{
  cout << "EDDY::: Eddy failed with message " << e.what() << endl;
  exit(EXIT_FAILURE);
}
catch(...)
{
  cout << "EDDY::: Eddy failed" << endl;
  exit(EXIT_FAILURE);
} 

namespace EDDY {

 
ReplacementManager *DoVolumeToVolumeRegistration(
						 const EddyCommandLineOptions&  clo,     
						 
						 ECScanManager&                 sm) EddyTry     
{
  
  NEWMAT::Matrix b0_mss, b0_ph;
  ReplacementManager *b0_rm = NULL;
  if (clo.NIter() && clo.Registerb0() && sm.NScans(B0)>1) {
    if (clo.Verbose()) cout << "Running Register" << endl;
    b0_rm = Register(clo,B0,clo.NIter(),clo.FWHM(),clo.b0_SecondLevelModel(),false,sm,b0_rm,b0_mss,b0_ph);
    if (clo.IsSliceToVol()) { 
      double minmss=1e20;
      unsigned int mindx=0;
      for (unsigned int i=0; i<sm.NScans(B0); i++) {
	if (b0_mss(b0_mss.Nrows(),i+1) < minmss) { minmss=b0_mss(b0_mss.Nrows(),i+1); mindx=i; }
      }
      if (clo.Verbose()) cout << "Setting scan " << sm.Getb02GlobalIndexMapping(mindx) << " as b0 shape-reference."<< endl; 
      sm.SetB0ShapeReference(sm.Getb02GlobalIndexMapping(mindx));
    }
    
    if (clo.Verbose()) cout << "Running sm.ApplyB0LocationReference" << endl; 
    sm.ApplyB0LocationReference(); 
  }
  
  if (sm.B0sAreInterspersed() && sm.UseB0sToInformDWIRegistration() && clo.Registerb0() && clo.RegisterDWI()) {
    if (clo.Verbose()) cout << "Running sm.PolateB0MovPar" << endl; 
    sm.PolateB0MovPar();
  }
  
  NEWMAT::Matrix dwi_mss, dwi_ph;
  ReplacementManager *dwi_rm = NULL;
  if (clo.NIter() && clo.RegisterDWI()) {
    if (clo.Verbose()) cout << "Running Register" << endl;
    dwi_rm = Register(clo,DWI,clo.NIter(),clo.FWHM(),clo.SecondLevelModel(),true,sm,dwi_rm,dwi_mss,dwi_ph);
    if (clo.IsSliceToVol()) { 
      std::vector<double> bvals;
      std::vector<std::vector<unsigned int> > shindx = sm.GetShellIndicies(bvals);
      for (unsigned int shell=0; shell<shindx.size(); shell++) {
	double minmss=1e20;
	unsigned int mindx=0;
	bool found_vol_with_no_outliers=false;
	for (unsigned int i=0; i<shindx[shell].size(); i++) {
	  if (!sm.Scan(shindx[shell][i]).HasOutliers()) { 
	    found_vol_with_no_outliers=true;
	    if (dwi_mss(dwi_mss.Nrows(),sm.GetGlobal2DWIIndexMapping(shindx[shell][i])+1) < minmss) {
	      minmss=dwi_mss(dwi_mss.Nrows(),sm.GetGlobal2DWIIndexMapping(shindx[shell][i])+1); 
	      mindx=shindx[shell][i];
	    }
	  }
	}
	if (!found_vol_with_no_outliers) {
	  std::vector<unsigned int> i2i = sm.GetDwi2GlobalIndexMapping();
	  dwi_rm->WriteReport(i2i,clo.OLReportFname());
	  dwi_rm->WriteMatrixReport(i2i,sm.NScans(),clo.OLMapReportFname(true),clo.OLNStDevMapReportFname(true),clo.OLNSqrStDevMapReportFname(true));
	  std::ostringstream errtxt; 
	  errtxt << "DoVolumeToVolumeRegistration: Unable to find volume with no outliers in shell " << shell << " with b-value=" << bvals[shell];
	  throw EddyException(errtxt.str());
	}
	if (clo.Verbose()) cout << "Setting scan " << mindx << " as shell shape-reference for shell "<< shell << " with b-value= " << bvals[shell] << endl; 
	sm.SetShellShapeReference(shell,mindx);
      }
      
    }
    if (clo.Verbose()) cout << "Running sm.ApplyDWILocationReference" << endl; 
    sm.ApplyDWILocationReference(); 
  }
  
  if (clo.NIter() && clo.History()) { 
    if (clo.RegisterDWI()) {
      MISCMATHS::write_ascii_matrix(clo.DwiMssHistoryFname(),dwi_mss); 
      MISCMATHS::write_ascii_matrix(clo.DwiParHistoryFname(),dwi_ph);
    }
    if (clo.Registerb0()) {
      MISCMATHS::write_ascii_matrix(clo.B0MssHistoryFname(),b0_mss); 
      MISCMATHS::write_ascii_matrix(clo.B0ParHistoryFname(),b0_ph);
    }
  }
  return(dwi_rm);
} EddyCatch

 

ReplacementManager *DoSliceToVolumeRegistration(
						const EddyCommandLineOptions&  clo,    
						unsigned int                   oi,        
						bool                           dol,       
						
						ECScanManager&                 sm,
						ReplacementManager             *dwi_rm) EddyTry     
{
  
  NEWMAT::Matrix b0_mss, b0_ph;
  ReplacementManager *b0_rm = NULL;
  if (clo.S2V_NIter(oi) && clo.Registerb0() && sm.NScans(B0)>1) {
    if (clo.Verbose()) cout << "Running Register" << endl;
    b0_rm = Register(clo,B0,clo.S2V_NIter(oi),clo.S2V_FWHM(oi),clo.b0_SecondLevelModel(),false,sm,b0_rm,b0_mss,b0_ph);
    
    if (clo.Verbose()) cout << "Running sm.ApplyB0ShapeReference" << endl; 
    sm.ApplyB0ShapeReference();
    
    if (clo.Verbose()) cout << "Running sm.ApplyB0LocationReference" << endl; 
    sm.ApplyB0LocationReference();
  }
  
  NEWMAT::Matrix dwi_mss, dwi_ph;
  if (clo.S2V_NIter(oi) && clo.RegisterDWI()) {
    if (clo.Verbose()) cout << "Running Register" << endl;
    dwi_rm = Register(clo,DWI,clo.S2V_NIter(oi),clo.S2V_FWHM(oi),clo.SecondLevelModel(),dol,sm,dwi_rm,dwi_mss,dwi_ph);
    
    if (clo.Verbose()) cout << "Running sm.ApplyShellShapeReference" << endl; 
    for (unsigned int si=0; si<sm.NoOfShells(DWI); si++) sm.ApplyShellShapeReference(si);
    
    if (clo.Verbose()) cout << "Running sm.ApplyDWILocationReference" << endl; 
    sm.ApplyDWILocationReference();
  }
  
  if (clo.S2V_NIter(oi) && clo.History()) { 
    if (clo.RegisterDWI()) {
      MISCMATHS::write_ascii_matrix(clo.DwiMssS2VHistoryFname(),dwi_mss); 
      MISCMATHS::write_ascii_matrix(clo.DwiParS2VHistoryFname(),dwi_ph);
    }
    if (clo.Registerb0()) {
      MISCMATHS::write_ascii_matrix(clo.B0MssS2VHistoryFname(),b0_mss); 
      MISCMATHS::write_ascii_matrix(clo.B0ParS2VHistoryFname(),b0_ph);
    }
  }
  return(dwi_rm);
} EddyCatch

 



 
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
			     NEWMAT::Matrix&                phist) EddyTry
{
  msshist.ReSize(niter,sm.NScans(st));
  phist.ReSize(niter,sm.NScans(st)*sm.Scan(0,st).NParam());
  double *mss_tmp = new double[sm.NScans(st)]; 
  if (rm == NULL) { 
    rm = new ReplacementManager(sm.NScans(st),static_cast<unsigned int>(sm.Scan(0,st).GetIma().zsize()),clo.OLDef(),clo.OLErrorType(),clo.OLType(),clo.MultiBand());
  }
  NEWIMAGE::volume<float> mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; 

  for (unsigned int iter=0; iter<niter; iter++) {
    
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
    #endif
    
    DiffStatsVector stats(sm.NScans(st));
    if (dol) {
      std::shared_ptr<DWIPredictionMaker> od_pmp;
      #ifdef COMPILE_GPU
      od_pmp = pmp;
      stats = EddyGpuUtils::DetectOutliers(clo,st,od_pmp,mask,sm,*rm);
      if (iter) {
	EddyGpuUtils::ReplaceOutliers(clo,st,od_pmp,mask,*rm,false,sm);
	
        pmp = EddyGpuUtils::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
      }
      #else
      od_pmp = pmp;
      stats = EDDY::DetectOutliers(clo,st,od_pmp,mask,sm,*rm);
      if (iter) {
	EDDY::ReplaceOutliers(clo,st,od_pmp,mask,*rm,false,sm);
	
	pmp = EDDY::LoadPredictionMaker(clo,st,sm,iter,fwhm[iter],mask);
      }
      #endif
    }
    
    
    
    
    
    
    if (clo.Verbose()) cout << "Calculating parameter updates" << endl;
    #ifdef COMPILE_GPU
    for (GpuPredictorChunk c(sm.NScans(st),mask); c<sm.NScans(st); c++) {
      std::vector<unsigned int> si = c.Indicies();
      if (clo.VeryVerbose()) cout << "Making predictions for scans: " << c << endl;
      std::vector<NEWIMAGE::volume<float> > pred = pmp->Predict(si);
      if (clo.VeryVerbose()) cout << "Finished making predictions for scans: " << c << endl;
      for (unsigned int i=0; i<si.size(); i++) {
        unsigned int global_indx = (st==EDDY::DWI) ? sm.GetDwi2GlobalIndexMapping(si[i]) : sm.Getb02GlobalIndexMapping(si[i]);
	if (clo.DebugLevel() && clo.DebugIndicies().IsAmongIndicies(global_indx)) {
	  mss_tmp[si[i]] = EddyGpuUtils::MovAndECParamUpdate(pred[i],sm.GetSuscHzOffResField(si[i],st),mask,true,fwhm[iter],sm.GetPolation(),global_indx,iter,clo.DebugLevel(),sm.Scan(si[i],st));
	}
	else mss_tmp[si[i]] = EddyGpuUtils::MovAndECParamUpdate(pred[i],sm.GetSuscHzOffResField(si[i],st),mask,true,fwhm[iter],sm.GetPolation(),sm.Scan(si[i],st));
	
	if (clo.VeryVerbose()) printf("Iter: %d, scan: %d, gpu_mss = %f\n",iter,si[i],mss_tmp[si[i]]);
      }
    }
    #else
    # pragma omp parallel for shared(mss_tmp, pmp)
    for (int s=0; s<int(sm.NScans(st)); s++) {
      
      NEWIMAGE::volume<float> pred = pmp->Predict(s);
      
      if (clo.DebugLevel()) {
    mss_tmp[s] = EddyUtils::MovAndECParamUpdate(pred,sm.GetSuscHzOffResField(s,st),mask,true,fwhm[iter],s,iter,clo.DebugLevel(),sm.Scan(s,st));
      }
      else mss_tmp[s] = EddyUtils::MovAndECParamUpdate(pred,sm.GetSuscHzOffResField(s,st),mask,true,fwhm[iter],sm.Scan(s,st));
      if (clo.VeryVerbose()) printf("Iter: %d, scan: %d, mss = %f\n",iter,s,mss_tmp[s]);
    }
    #endif

    
    Diagnostics(clo,iter,st,sm,mss_tmp,stats,*rm,msshist,phist);

    
    if (slm != No_2nd_lvl_mdl) {
      if (clo.VeryVerbose()) cout << "Performing 2nd level modelling of estimated parameters" << endl;
      sm.SetPredictedECParam(st,slm);
    }

    
    if (clo.SeparateOffsetFromMovement()) {
      if (clo.VeryVerbose()) cout << "Attempting to separate field-offset from subject movement" << endl;
      sm.SeparateFieldOffsetFromMovement(st,clo.OffsetModel());
    }    
  }

  delete [] mss_tmp;
  return(rm);
} EddyCatch

ReplacementManager *FinalOLCheck(
				 const EddyCommandLineOptions&  clo,
				 
				 ReplacementManager             *rm,
				 ECScanManager&                 sm) EddyTry
{
  NEWIMAGE::volume<float> mask = sm.Scan(0,DWI).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; 
  if (rm == NULL) {
    rm = new ReplacementManager(sm.NScans(DWI),static_cast<unsigned int>(sm.Scan(0,DWI).GetIma().zsize()),clo.OLDef(),clo.OLErrorType(),clo.OLType(),clo.MultiBand());
  }

  
  #ifdef COMPILE_GPU
  std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  #else
  std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  #endif
  
  DiffStatsVector stats(sm.NScans(DWI));
  std::shared_ptr<DWIPredictionMaker> od_pmp;
  bool add_noise = clo.AddNoiseToReplacements();
  #ifdef COMPILE_GPU
  stats = EddyGpuUtils::DetectOutliers(clo,DWI,pmp,mask,sm,*rm);
  EddyGpuUtils::ReplaceOutliers(clo,DWI,pmp,mask,*rm,add_noise,sm);
  #else
  stats = EDDY::DetectOutliers(clo,DWI,pmp,mask,sm,*rm);
  EDDY::ReplaceOutliers(clo,DWI,pmp,mask,*rm,add_noise,sm);
  #endif
  return(rm);
} EddyCatch

 
std::shared_ptr<DWIPredictionMaker> LoadPredictionMaker(
							const EddyCommandLineOptions& clo,
							ScanType                      st,
							const ECScanManager&          sm,
							unsigned int                  iter,
							float                         fwhm,
							
							NEWIMAGE::volume<float>&      mask,
							
							bool                          use_orig) EddyTry
{
  std::shared_ptr<DWIPredictionMaker>  pmp;                                 
  if (st==DWI) { 
    std::shared_ptr<KMatrix> K;
    if (clo.CovarianceFunction() == Spherical) K = std::shared_ptr<SphericalKMatrix>(new SphericalKMatrix(clo.DontCheckShelling()));
    else if (clo.CovarianceFunction() == Exponential) K = std::shared_ptr<ExponentialKMatrix>(new ExponentialKMatrix(clo.DontCheckShelling()));
    else if (clo.CovarianceFunction() == NewSpherical) K = std::shared_ptr<NewSphericalKMatrix>(new NewSphericalKMatrix(clo.DontCheckShelling()));
    else throw EddyException("LoadPredictionMaker: Unknown covariance function");
    std::shared_ptr<HyParCF> hpcf;
    std::shared_ptr<HyParEstimator> hpe;
    if (clo.HyperParFixed()) hpe = std::shared_ptr<FixedValueHyParEstimator>(new FixedValueHyParEstimator(clo.HyperParValues()));
    else {
      if (clo.HyParCostFunction() == CC) hpe = std::shared_ptr<CheapAndCheerfulHyParEstimator>(new CheapAndCheerfulHyParEstimator(clo.NVoxHp(),clo.InitRand()));
      else {
	if (clo.HyParCostFunction() == MML) hpcf = std::shared_ptr<MMLHyParCF>(new MMLHyParCF);
	else if (clo.HyParCostFunction() == CV) hpcf = std::shared_ptr<CVHyParCF>(new CVHyParCF);
	else if (clo.HyParCostFunction() == GPP) hpcf = std::shared_ptr<GPPHyParCF>(new GPPHyParCF);
	else throw EddyException("LoadPredictionMaker: Unknown hyperparameter cost-function");
	hpe = std::shared_ptr<FullMontyHyParEstimator>(new FullMontyHyParEstimator(hpcf,clo.HyParFudgeFactor(),clo.NVoxHp(),clo.InitRand(),clo.VeryVerbose()));
      }
    }
    pmp = std::shared_ptr<DWIPredictionMaker>(new DiffusionGP(K,hpe));  
  }
  else pmp = std::shared_ptr<DWIPredictionMaker>(new b0Predictor);  
  pmp->SetNoOfScans(sm.NScans(st));
  mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0;

  if (clo.Verbose()) cout << "Loading prediction maker";
  if (clo.VeryVerbose()) cout << endl << "Scan: " << endl;
#pragma omp parallel for shared(pmp,st)
  for (int s=0; s<int(sm.NScans(st)); s++) {
    if (clo.VeryVerbose()) printf(" %d\n",s);
    NEWIMAGE::volume<float> tmpmask = sm.Scan(s,st).GetIma(); 
    EddyUtils::SetTrilinearInterp(tmpmask); tmpmask = 1.0;
    if (use_orig) pmp->SetScan(sm.GetUnwarpedOrigScan(s,tmpmask,st),sm.Scan(s,st).GetDiffPara(clo.RotateBVecsDuringEstimation()),s);
    else pmp->SetScan(sm.GetUnwarpedScan(s,tmpmask,st),sm.Scan(s,st).GetDiffPara(clo.RotateBVecsDuringEstimation()),s);
#pragma omp critical
    {
      mask *= tmpmask;
    }
  }
  if (clo.Verbose()) cout << endl << "Evaluating prediction maker model" << endl;
  pmp->EvaluateModel(sm.Mask()*mask,fwhm,clo.Verbose());
  if (clo.DebugLevel() > 2 && st==DWI) {
    char fname[256];
    sprintf(fname,"EDDY_DEBUG_K_Mat_Data_%02d",iter);
    pmp->WriteMetaData(fname);
  }

  return(pmp);
} EddyCatch

DiffStatsVector DetectOutliers(
			       const EddyCommandLineOptions&             clo,
			       ScanType                                  st,
			       const std::shared_ptr<DWIPredictionMaker> pmp,
			       const NEWIMAGE::volume<float>&            mask,
			       const ECScanManager&                      sm,
			       
			       ReplacementManager&                       rm) EddyTry
{
  if (clo.VeryVerbose()) cout << "Checking for outliers" << endl;
  
  DiffStatsVector stats(sm.NScans(st));
#pragma omp parallel for shared(stats,st)
  for (int s=0; s<int(sm.NScans(st)); s++) {
    NEWIMAGE::volume<float> pred = pmp->Predict(s);
    stats[s] = EddyUtils::GetSliceWiseStats(pred,sm.GetSuscHzOffResField(s,st),mask,sm.Mask(),sm.Scan(s,st));
  }
  
  rm.Update(stats);
  return(stats);
} EddyCatch

void ReplaceOutliers(
		     const EddyCommandLineOptions&             clo,
		     ScanType                                  st,
		     const std::shared_ptr<DWIPredictionMaker> pmp,
		     const NEWIMAGE::volume<float>&            mask,
		     const ReplacementManager&                 rm,
		     bool                                      add_noise,
		     
		     ECScanManager&                            sm) EddyTry
{
  
  if (clo.VeryVerbose()) cout << "Replacing outliers with predictions" << endl;
#pragma omp parallel for shared(st)
  for (int s=0; s<int(sm.NScans(st)); s++) {
    std::vector<unsigned int> ol = rm.OutliersInScan(s);
    if (ol.size()) { 
      if (clo.VeryVerbose()) cout << "Scan " << s << " has " << ol.size() << " outlier slices" << endl;
      NEWIMAGE::volume<float> pred = pmp->Predict(s,true);
      if (add_noise) {
	double vp = pmp->PredictionVariance(s,true);
	double ve = pmp->ErrorVariance(s);
	double stdev = std::sqrt(vp+ve) - std::sqrt(vp);
	pred += EddyUtils::MakeNoiseIma(pred,0.0,stdev);
      }
      sm.Scan(s,st).SetAsOutliers(pred,sm.GetSuscHzOffResField(s,st),mask,ol);
    }
  }
  return;
} EddyCatch

 
void GetPredictionsForResampling(
				 const EddyCommandLineOptions&    clo,
				 ScanType                         st,
				 const ECScanManager&             sm,
				 
				 NEWIMAGE::volume4D<float>&       pred) EddyTry
{
  pred.reinitialize(sm.Scan(0,st).GetIma().xsize(),sm.Scan(0,st).GetIma().ysize(),sm.Scan(0,st).GetIma().zsize(),sm.NScans(st));
  NEWIMAGE::copybasicproperties(sm.Scan(0,st).GetIma(),pred);
  NEWIMAGE::volume<float> mask = sm.Scan(0,st).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; 
  EddyCommandLineOptions lclo = clo;
  if (lclo.HyParFudgeFactor() != 1.0) lclo.SetHyParFudgeFactor(1.0);
  if (st == ANY || st == B0) {
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(lclo,B0,sm,0,0.0,mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(lclo,B0,sm,0,0.0,mask);
    #endif
    for (unsigned int s=0; s<sm.NScans(B0); s++) {
      if (st == B0) pred[s] = pmp->Predict(s,true);
      else pred[sm.Getb02GlobalIndexMapping(s)] = pmp->Predict(s,true);
    }
  }
  if (st == ANY || st == DWI) {
    #ifdef COMPILE_GPU
    std::shared_ptr<DWIPredictionMaker> pmp = EddyGpuUtils::LoadPredictionMaker(lclo,DWI,sm,0,0.0,mask);
    #else
    std::shared_ptr<DWIPredictionMaker> pmp = EDDY::LoadPredictionMaker(lclo,DWI,sm,0,0.0,mask);
    #endif
    for (unsigned int s=0; s<sm.NScans(DWI); s++) {
      if (st == DWI) pred[s] = pmp->Predict(s,true);
      else pred[sm.GetDwi2GlobalIndexMapping(s)] = pmp->Predict(s,true);
    }
  }
  return;
} EddyCatch

void WriteCNRMaps(
		  const EddyCommandLineOptions&   clo,
		  const ECScanManager&            sm,
		  const std::string&              spatial_fname,
		  const std::string&              range_fname,
		  const std::string&              residual_fname) EddyTry
{
  if (spatial_fname == std::string("") && residual_fname == std::string("")) throw EddyException("EDDY::WriteCNRMaps: At least one of spatial and residual fname must be set");

  
  NEWIMAGE::volume<float> mask = sm.Scan(0,DWI).GetIma(); EddyUtils::SetTrilinearInterp(mask); mask = 1.0; 
  #ifdef COMPILE_GPU
  std::shared_ptr<DWIPredictionMaker> dwi_pmp = EddyGpuUtils::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  std::shared_ptr<DWIPredictionMaker> b0_pmp = EddyGpuUtils::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
  #else 
  std::shared_ptr<DWIPredictionMaker> dwi_pmp = EDDY::LoadPredictionMaker(clo,DWI,sm,0,0.0,mask);
  std::shared_ptr<DWIPredictionMaker> b0_pmp = EDDY::LoadPredictionMaker(clo,B0,sm,0,0.0,mask);
  #endif

  if (sm.IsShelled()) {
    if (spatial_fname != std::string("") || range_fname != std::string("")) {  
      std::vector<double>                      grpb;
      std::vector<std::vector<unsigned int> >  dwi_indx = sm.GetShellIndicies(grpb); 
      std::vector<NEWIMAGE::volume<float> >    mvols(grpb.size());
      
      for (unsigned int i=0; i<grpb.size(); i++) {
	mvols[i] = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
	for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
	  mvols[i] += dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j]));
	}
	mvols[i] /= float(dwi_indx[i].size());
      }
      
      std::vector<NEWIMAGE::volume<float> >    stdvols(grpb.size());
      for (unsigned int i=0; i<grpb.size(); i++) {
	NEWIMAGE::volume<float> tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0])) - mvols[i];
	stdvols[i] = tmp*tmp;
	for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
	  tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])) - mvols[i];
	  stdvols[i] += tmp*tmp;
	}
	stdvols[i] /= float(dwi_indx[i].size()-1);
	stdvols[i] = NEWIMAGE::sqrt(stdvols[i]);
      }
      
      std::vector<NEWIMAGE::volume<float> >    minvols(grpb.size());
      std::vector<NEWIMAGE::volume<float> >    maxvols(grpb.size());
      for (unsigned int i=0; i<grpb.size(); i++) {
	minvols[i] = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
	maxvols[i] = minvols[i];
	for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
	  minvols[i] = NEWIMAGE::min(minvols[i],dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])));
	  maxvols[i] = NEWIMAGE::max(maxvols[i],dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])));
	}
	maxvols[i] -= minvols[i]; 
      }
      
      std::vector<NEWIMAGE::volume<float> >  stdres(grpb.size());
      for (unsigned int i=0; i<grpb.size(); i++) {
	NEWIMAGE::volume<float> tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0])) - dwi_pmp->InputData(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][0]));
	stdres[i] = tmp*tmp;
	for (unsigned int j=1; j<dwi_indx[i].size(); j++) {
	  tmp = dwi_pmp->Predict(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j])) - dwi_pmp->InputData(sm.GetGlobal2DWIIndexMapping(dwi_indx[i][j]));
	  stdres[i] += tmp*tmp;
	}
	stdres[i] /= float(dwi_indx[i].size()-1);
	stdres[i] = NEWIMAGE::sqrt(stdres[i]);
      }
      
      for (unsigned int i=0; i<grpb.size(); i++) {
	maxvols[i] /= stdres[i];
	stdvols[i] /= stdres[i];
      }
      
      std::vector<unsigned int>   b0_indx = sm.GetB0Indicies();
      NEWIMAGE::volume<float>     b0_SNR = b0_pmp->Predict(0) - b0_pmp->InputData(0);
      if (b0_indx.size() > 1) {
	b0_SNR *= b0_SNR;
	for (unsigned int i=1; i<b0_indx.size(); i++) {
	  NEWIMAGE::volume<float> tmp = b0_pmp->Predict(i) - b0_pmp->InputData(i);
	  b0_SNR += tmp*tmp;
	}
	b0_SNR /= float(b0_indx.size()-1);
	b0_SNR = NEWIMAGE::sqrt(b0_SNR);
	b0_SNR = b0_pmp->Predict(0) /= b0_SNR;
      }
      else b0_SNR = 0.0; 
      
      NEWIMAGE::volume4D<float> spat_cnr(stdvols[0].xsize(),stdvols[0].ysize(),stdvols[0].zsize(),stdvols.size()+1);
      NEWIMAGE::copybasicproperties(stdvols[0],spat_cnr);
      spat_cnr[0] = b0_SNR;
      if (spatial_fname != std::string("")) {
	for (unsigned int i=0; i<stdvols.size(); i++) spat_cnr[i+1] = stdvols[i];
	NEWIMAGE::write_volume(spat_cnr,spatial_fname);
      }
      
      if (range_fname != std::string("")) {
	for (unsigned int i=0; i<maxvols.size(); i++) spat_cnr[i+1] = maxvols[i]; 
	NEWIMAGE::write_volume(spat_cnr,range_fname);
      }
    }
    if (residual_fname != std::string("")) {   
      NEWIMAGE::volume4D<float> residuals(dwi_pmp->InputData(0).xsize(),dwi_pmp->InputData(0).ysize(),dwi_pmp->InputData(0).zsize(),sm.NScans(ANY));
      NEWIMAGE::copybasicproperties(dwi_pmp->InputData(0),residuals);
      for (unsigned int i=0; i<sm.NScans(DWI); i++) {
	residuals[sm.GetDwi2GlobalIndexMapping(i)] = dwi_pmp->InputData(i) - dwi_pmp->Predict(i);
      }
      for (unsigned int i=0; i<sm.NScans(B0); i++) {
	residuals[sm.Getb02GlobalIndexMapping(i)] = b0_pmp->InputData(i) - b0_pmp->Predict(i);
      }
      NEWIMAGE::write_volume(residuals,residual_fname);    
    }
  }
  else {
    throw EddyException("WriteCNRMaps: Cannot calculate CNR for non-shelled data.");
  }
} EddyCatch

void Diagnostics(
		 const EddyCommandLineOptions&  clo,      
		 unsigned int                   iter,     
		 ScanType                       st,       
		 const ECScanManager&           sm,       
                 const double                   *mss_tmp, 
                 const DiffStatsVector&         stats,    
		 const ReplacementManager&      rm,       
		 
		 NEWMAT::Matrix&                mss,      
		 NEWMAT::Matrix&                phist) EddyTry   
{
  if (clo.Verbose()) {
    double tss=0.0;
    for (unsigned int s=0; s<sm.NScans(st); s++) tss+=mss_tmp[s]; 
    cout << "Iter: " << iter << ", Total mss = " << tss/sm.NScans(st) << endl;
  }

  for (unsigned int s=0; s<sm.NScans(st); s++) {
    mss(iter+1,s+1) = mss_tmp[s];
    phist.SubMatrix(iter+1,iter+1,s*sm.Scan(0,st).NParam()+1,(s+1)*sm.Scan(0,st).NParam()) = sm.Scan(s,st).GetParams().t();
  }

  if (clo.WriteSliceStats()) {
    char istring[256];
    if (st==EDDY::DWI) sprintf(istring,"%s.EddyDwiSliceStatsIteration%02d",clo.IOutFname().c_str(),iter);
    else sprintf(istring,"%s.Eddyb0SliceStatsIteration%02d",clo.IOutFname().c_str(),iter);
    stats.Write(string(istring));
    rm.DumpOutlierMaps(string(istring));
  }
} EddyCatch

void AddRotation(ECScanManager&               sm,
		 const NEWMAT::ColumnVector&  rp) EddyTry
{
  for (unsigned int i=0; i<sm.NScans(); i++) {
    NEWMAT::ColumnVector mp = sm.Scan(i).GetParams(EDDY::MOVEMENT);
    mp(4) += rp(1); mp(5) += rp(2); mp(6) += rp(3); 
    sm.Scan(i).SetParams(mp,EDDY::MOVEMENT);
  } 
} EddyCatch

void PrintMIValues(const EddyCommandLineOptions&  clo,      
                   const ECScanManager&           sm,
                   const std::string&             fname,
                   bool                           write_planes) EddyTry
{
  std::vector<std::string> dir(6);
  dir[0]="xt"; dir[1]="yt"; dir[2]="zt";
  dir[3]="xr"; dir[4]="yr"; dir[5]="zr";
  
  for (unsigned int i=0; i<6; i++) {
    std::vector<unsigned int> n(6,1); n[i] = 100;
    std::vector<double> first(6,0.0); first[i] = -2.5;
    std::vector<double> last(6,0.0); last[i] = 2.5;
    if (clo.VeryVerbose()) cout << "Writing MI values for direction " << i << endl;
    PEASUtils::WritePostEddyBetweenShellMIValues(clo,sm,n,first,last,fname+"_"+dir[i]);
  }
  
  for (unsigned int i=0; i<6; i++) {
    for (unsigned int j=i+1; j<6; j++) {
      std::vector<unsigned int> n(6,1); n[i] = 20; n[j] = 20; 
      std::vector<double> first(6,0.0); first[i] = -1.0; first[j] = -1.0;
      std::vector<double> last(6,0.0); last[i] = 1.0; last[j] = 1.0;
      if (clo.VeryVerbose()) cout << "Writing MI values for plane " << i << "-" << j << endl;
      PEASUtils::WritePostEddyBetweenShellMIValues(clo,sm,n,first,last,fname+"_"+dir[i]+"_"+dir[j]);
    }
  }
} EddyCatch

} 




