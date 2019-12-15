/////////////////////////////////////////////////////////////////////
/// \file ECScanClasses.h
/// \brief Declarations of classes that implements a scan or a collection of scans within the EC project.
///
/// \author Jesper Andersson
/// \version 1.0b, Sep., 2012.
/// \Copyright (C) 2012 University of Oxford 
///




#pragma GCC diagnostic ignored "-Wunknown-pragmas" 

#ifndef ECScanClasses_h
#define ECScanClasses_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"

namespace EDDY {







class ECScan
{
public:
  
  ECScan(const NEWIMAGE::volume<float>&   ima,
         const AcqPara&                   acqp,
         const DiffPara&                  diffp,
	 const ScanMovementModel&         mp,
	 const MultiBandGroups&           mbg,
         std::shared_ptr<ScanECModel>     ecp,
	 double                           mrl=1.0) EddyTry : _ima(ima), _ols(ima.zsize(),0ul), _acqp(acqp), _diffp(diffp), _mp(mp), _mbg(mbg), _ecp(ecp->Clone()), _mrl(mrl) {} EddyCatch
  
  ECScan(const ECScan& inp) EddyTry : _ima(inp._ima), _ols(inp._ols), _acqp(inp._acqp), _diffp(inp._diffp), _mp(inp._mp), _mbg(inp._mbg), _ecp(inp._ecp->Clone()), _mrl(inp._mrl) {
    for (int sl=0; sl<_ima.zsize(); sl++) {
      if (_ols[sl]) { _ols[sl] = new float[_ima.xsize()*_ima.ysize()]; std::memcpy(_ols[sl],inp._ols[sl],_ima.xsize()*_ima.ysize()*sizeof(float)); }
    }
  } EddyCatch
  virtual ~ECScan() EddyTry { for (int sl=0; sl<_ima.zsize(); sl++) { if (_ols[sl]) delete[] _ols[sl]; } } EddyCatch
  
  ECScan& operator=(const ECScan& rhs) EddyTry {
    if (this == &rhs) return(*this);
    _ima=rhs._ima; _ols=rhs._ols; _acqp=rhs._acqp; _diffp=rhs._diffp; _mp=rhs._mp; _mbg=rhs._mbg; _ecp=rhs._ecp->Clone(); _mrl=rhs._mrl;
    for (int sl=0; sl<_ima.zsize(); sl++) {
      if (_ols[sl]) { _ols[sl] = new float[_ima.xsize()*_ima.ysize()]; std::memcpy(_ols[sl],rhs._ols[sl],_ima.xsize()*_ima.ysize()*sizeof(float)); }
    }    
    return(*this);
  } EddyCatch
  
  ECModel Model() const EddyTry { return(_ecp->WhichModel()); } EddyCatch
  
  void SetMovementModelOrder(unsigned int order) EddyTry { if (int(order)>_ima.zsize()) throw EddyException("ECScan::SetMovementModelOrder: order too high"); else _mp.SetOrder(order); } EddyCatch
  
  unsigned int GetMovementModelOrder() const EddyTry { return(_mp.Order()); } EddyCatch
  
  ScanMovementModel& GetMovementModel() EddyTry { return(_mp); } EddyCatch
  
  bool IsSliceToVol() const EddyTry { return(GetMovementModelOrder()); } EddyCatch
  
  double GetMovementStd(unsigned int mi) const EddyTry { std::vector<unsigned int> empty; return(this->GetMovementStd(mi,empty)); } EddyCatch
  
  double GetMovementStd(unsigned int mi, std::vector<unsigned int> icsl) const;  
  
  const MultiBandGroups& GetMBG() const EddyTry { return(_mbg); } EddyCatch
  
  bool HasOutliers() const EddyTry { for (unsigned int i=0; i<_ols.size(); i++) { if (_ols[i]) return(true); } return(false); } EddyCatch
  
  bool IsOutlier(unsigned int sl) const EddyTry { if (_ols.at(sl)) return(true); else return(false); } EddyCatch
  
  bool HasFieldOffset() const EddyTry { return(_ecp->HasFieldOffset()); } EddyCatch
  
  double GetFieldOffset() const EddyTry { return(_ecp->GetFieldOffset()); } EddyCatch
  
  void SetFieldOffset(double ofst) EddyTry { _ecp->SetFieldOffset(ofst); } EddyCatch
  
  void SetPolation(const PolationPara& pp);
  
  bool HasEmptyPlane(std::vector<unsigned int>&  pi) const;
  
  void FillEmptyPlane(const std::vector<unsigned int>&  pi);
  
  NEWIMAGE::volume<float> GetOriginalIma() const;
  
  NEWIMAGE::volume<float> GetMotionCorrectedOriginalIma(NEWIMAGE::volume<float>& omask) const EddyTry { return(motion_correct(GetOriginalIma(),&omask)); } EddyCatch
  
  NEWIMAGE::volume<float> GetMotionCorrectedOriginalIma() const EddyTry { return(motion_correct(GetOriginalIma(),NULL)); } EddyCatch
  
  NEWIMAGE::volume<float> GetUnwarpedOriginalIma(
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 
						 NEWIMAGE::volume<float>&                          omask) const;
  
  NEWIMAGE::volume<float> GetUnwarpedOriginalIma(
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc) const;
  
  const NEWIMAGE::volume<float>& GetIma() const EddyTry { return(_ima); } EddyCatch
  
  NEWIMAGE::volume<float> GetMotionCorrectedIma(NEWIMAGE::volume<float>& omask) const EddyTry { return(motion_correct(GetIma(),&omask)); } EddyCatch
  
  NEWIMAGE::volume<float> GetMotionCorrectedIma() const EddyTry { return(motion_correct(GetIma(),NULL)); } EddyCatch
  
  NEWIMAGE::volume<float> GetUnwarpedIma(
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
					 
					 NEWIMAGE::volume<float>&                          omask) const;
  
  NEWIMAGE::volume<float> GetUnwarpedIma(
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc) const;
  
  NEWIMAGE::volume<float> GetVolumetricUnwarpedIma(
						   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						   
						   NEWIMAGE::volume4D<float>&                        deriv) const;
  
  AcqPara GetAcqPara() const EddyTry { return(_acqp); } EddyCatch
  
  DiffPara GetDiffPara(bool rot=false) const;
  
  NEWMAT::ColumnVector GetHz2mmVector() const;
  
  unsigned int NParam(EDDY::Parameters whichp=EDDY::ALL) const EddyTry { 
    if (whichp==EDDY::ZERO_ORDER_MOVEMENT) return(static_cast<unsigned int>(_mp.GetZeroOrderParams().Nrows()));
    if (whichp==EDDY::MOVEMENT) return(_mp.NParam());
    else if (whichp==EDDY::EC) return(_ecp->NParam());
    else return(_mp.NParam()+_ecp->NParam());
  } EddyCatch
  
  NEWMAT::ColumnVector GetParams(EDDY::Parameters whichp=EDDY::ALL) const EddyTry { 
    if (whichp==EDDY::ZERO_ORDER_MOVEMENT) return(_mp.GetZeroOrderParams()); 
    else if (whichp==EDDY::MOVEMENT) return(_mp.GetParams()); 
    else if (whichp==EDDY::EC) return(_ecp->GetParams()); 
    else return(_mp.GetParams() & _ecp->GetParams());
  } EddyCatch
  
  unsigned int NDerivs(EDDY::Parameters whichp=EDDY::ALL) const EddyTry { 
    if (whichp==EDDY::MOVEMENT) return(_mp.NDerivs()); 
    else if (whichp==EDDY::EC) return(_ecp->NDerivs()); 
    else return(_mp.NDerivs() + _ecp->NDerivs());
  } EddyCatch
  
  void SetRegLambda(double lambda) { _mrl=lambda; }
  
  double GetRegLambda() const { return(_mrl); }
  
  double GetReg(EDDY::Parameters whichp=EDDY::ALL) const;
  
  NEWMAT::ColumnVector GetRegGrad(EDDY::Parameters whichp=EDDY::ALL) const;
  
  NEWMAT::Matrix GetRegHess(EDDY::Parameters whichp=EDDY::ALL) const;
  
  double GetDerivParam(unsigned int indx, EDDY::Parameters whichp=EDDY::ALL) const;
  
  double GetDerivScale(unsigned int indx, EDDY::Parameters whichp=EDDY::ALL) const;
  
  void SetDerivParam(unsigned int indx, double p, EDDY::Parameters whichp=EDDY::ALL);
  
  NEWMAT::Matrix ForwardMovementMatrix() const EddyTry { return(_mp.ForwardMovementMatrix(_ima)); } EddyCatch
  
  NEWMAT::Matrix ForwardMovementMatrix(unsigned int grp) const EddyTry { return(_mp.ForwardMovementMatrix(_ima,grp,_mbg.NGroups())); } EddyCatch
  
  NEWMAT::Matrix RestrictedForwardMovementMatrix(const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedForwardMovementMatrix(_ima,rindx)); } EddyCatch
  
  NEWMAT::Matrix RestrictedForwardMovementMatrix(unsigned int grp, const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedForwardMovementMatrix(_ima,grp,_mbg.NGroups(),rindx)); } EddyCatch
  
  NEWMAT::Matrix InverseMovementMatrix() const EddyTry { return(_mp.InverseMovementMatrix(_ima)); } EddyCatch
  
  NEWMAT::Matrix InverseMovementMatrix(unsigned int grp) const EddyTry { return(_mp.InverseMovementMatrix(_ima,grp,_mbg.NGroups())); } EddyCatch
  
  NEWMAT::Matrix RestrictedInverseMovementMatrix(const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedInverseMovementMatrix(_ima,rindx)); } EddyCatch
  
  NEWMAT::Matrix RestrictedInverseMovementMatrix(unsigned int grp, const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedInverseMovementMatrix(_ima,grp,_mbg.NGroups(),rindx)); } EddyCatch
  
  EDDY::ImageCoordinates SamplingPoints() const;
  
  NEWIMAGE::volume<float> ECField() const EddyTry { return(_ecp->ECField(_ima)); } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForScanToModelTransform(
							 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							 
							 NEWIMAGE::volume<float>&                          omask,
							 NEWIMAGE::volume<float>&                          jac) const EddyTry {
    return(field_for_scan_to_model_transform(susc,&omask,&jac));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForScanToModelTransform(
							 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							 
							 NEWIMAGE::volume<float>&                          omask) const EddyTry {
    return(field_for_scan_to_model_transform(susc,&omask,NULL));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForScanToModelTransform(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(field_for_scan_to_model_transform(susc,NULL,NULL));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForModelToScanTransform(
							 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							 
							 NEWIMAGE::volume<float>&                        omask,
							 NEWIMAGE::volume<float>&                        jac) const EddyTry {
    return(field_for_model_to_scan_transform(susc,&omask,&jac));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForModelToScanTransform(
							 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							 
							 NEWIMAGE::volume<float>&                        omask) const EddyTry {
    return(field_for_model_to_scan_transform(susc,&omask,NULL));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForModelToScanTransform(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(field_for_model_to_scan_transform(susc,NULL,NULL));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> FieldForModelToScanTransformWithJac(
								std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								
								NEWIMAGE::volume<float>&                        jac) const EddyTry {
    return(field_for_model_to_scan_transform(susc,NULL,&jac));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> TotalDisplacementToModelSpace(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(total_displacement_to_model_space(GetOriginalIma(),susc));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> MovementDisplacementToModelSpace(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(total_displacement_to_model_space(GetOriginalIma(),susc,true));
  } EddyCatch
  
  NEWIMAGE::volume4D<float> RestrictedMovementDisplacementToModelSpace(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(total_displacement_to_model_space(GetOriginalIma(),susc,true,true));
  } EddyCatch
  
  void SetParams(const NEWMAT::ColumnVector& mpep, EDDY::Parameters whichp=EDDY::ALL);
  
  void SetAsOutliers(const NEWIMAGE::volume<float>&                     rep,
		     std::shared_ptr<const NEWIMAGE::volume<float> >    susc,
		     const NEWIMAGE::volume<float>&                     inmask, 
		     const std::vector<unsigned int>&                   ol);
  
  void SetAsOutliers(const NEWIMAGE::volume<float>&                     rep,
		     const NEWIMAGE::volume<float>&                     mask, 
		     const std::vector<unsigned int>&                   ol);
  
  void RecycleOutliers();
private:
  NEWIMAGE::volume<float>         _ima;   
  std::vector<float*>             _ols;   
  AcqPara                         _acqp;  
  DiffPara                        _diffp; 
  ScanMovementModel               _mp;    
  EDDY::MultiBandGroups           _mbg;   
  std::shared_ptr<ScanECModel>    _ecp;   
  double                          _mrl;   

  NEWIMAGE::volume<float> motion_correct(
					 const NEWIMAGE::volume<float>&  inima,
					 
					 NEWIMAGE::volume<float>         *omask) const;
  NEWIMAGE::volume<float> transform_to_model_space(
						   const NEWIMAGE::volume<float>&                    inima,
						   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						   
						   NEWIMAGE::volume<float>&                          omask,
						   
						   bool                                              jacmod=true) const;
  NEWIMAGE::volume<float> transform_volumetric_to_model_space(
							      const NEWIMAGE::volume<float>&                    inima,
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      
							      NEWIMAGE::volume<float>&                          omask,
							      NEWIMAGE::volume4D<float>&                        deriv,
							      
							      bool                                              jacmod=true) const;
  NEWIMAGE::volume4D<float> total_displacement_to_model_space(
							      const NEWIMAGE::volume<float>&                    inima,
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      
							      bool                                              movement_only=false,
							      bool                                              exclude_PE_tr=false) const;
  NEWIMAGE::volume4D<float> field_for_scan_to_model_transform(
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      
							      NEWIMAGE::volume<float>                           *omask,
							      NEWIMAGE::volume<float>                           *jac) const;
  NEWIMAGE::volume4D<float> field_for_model_to_scan_transform(
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      
							      NEWIMAGE::volume<float>                           *omask,
							      NEWIMAGE::volume<float>                           *jac) const;
  bool in_list(unsigned int s, const std::vector<unsigned int>& l) const EddyTry { for (unsigned int i=0; i<l.size(); i++) { if (l[i]==s) return(true); } return(false); } EddyCatch
};







class ECScanManager
{
public:
  
  ECScanManager(const std::string&               imafname,
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
		bool                             fsh); 
  ~ECScanManager() EddyTry {} EddyCatch
  
  unsigned int NScans(ScanType st=ANY) const;
  
  const MultiBandGroups& MultiBand() const EddyTry { return(Scan(0).GetMBG()); } EddyCatch
  
  void SetMovementModelOrder(unsigned int order) EddyTry { for (unsigned int i=0; i<NScans(); i++) Scan(i).SetMovementModelOrder(order); } EddyCatch
  
  unsigned int GetMovementModelOrder() const EddyTry { return(Scan(0).GetMovementModelOrder()); } EddyCatch
  
  bool IsSliceToVol() const EddyTry { return(Scan(0).IsSliceToVol()); } EddyCatch
  
  void Set_S2V_Lambda(double lambda) EddyTry { for (unsigned int i=0; i<NScans(); i++) Scan(i).SetRegLambda(lambda); } EddyCatch
  
  bool IsShelled() const;
  
  unsigned int NoOfShells(ScanType st=ANY) const;
  
  std::vector<DiffPara> GetDiffParas(ScanType st=ANY) const;
  
  std::vector<unsigned int> GetB0Indicies() const;
  
  std::vector<std::vector<unsigned int> > GetShellIndicies(std::vector<double>& bvals) const;
  
  unsigned int NLSRPairs(ScanType st=ANY) const;
  
  bool HasPEinX() const EddyTry { return(has_pe_in_direction(1)); } EddyCatch
  
  bool HasPEinY() const EddyTry { return(has_pe_in_direction(2)); } EddyCatch
  
  bool HasPEinXandY() const EddyTry { return(HasPEinX() && HasPEinY()); } EddyCatch
  
  bool IsDWI(unsigned int indx) const EddyTry { if (indx>_fi.size()-1) throw EddyException("ECScanManager::IsDWI: index out of range"); else return(!_fi[indx].first); } EddyCatch
  
  bool IsB0(unsigned int indx) const EddyTry { return(!IsDWI(indx)); } EddyCatch
  
  double ScaleFactor() const { return(_sf); }
  
  void SetPolation(const PolationPara& pp) EddyTry { _pp=pp; set_polation(pp); } EddyCatch
  
  PolationPara GetPolation() const EddyTry { return(_pp); } EddyCatch
  
  void FillEmptyPlanes();
  
  std::vector<unsigned int> GetDwi2GlobalIndexMapping() const;
  
  unsigned int GetDwi2GlobalIndexMapping(unsigned int dwindx) const;
  
  std::vector<unsigned int> Getb02GlobalIndexMapping() const;
  
  unsigned int Getb02GlobalIndexMapping(unsigned int b0indx) const;
  
  unsigned int GetGlobal2DWIIndexMapping(unsigned int gindx) const;
  
  unsigned int GetGlobal2b0IndexMapping(unsigned int gindx) const;
  
  PolationPara GetPolationMethods();
  
  void SetPolationMethods(const PolationPara& pp);
  
  bool B0sAreInterspersed() const;
  
  bool B0sAreUsefulForPEAS() const;
  
  void PolateB0MovPar();
  
  bool UseB0sToInformDWIRegistration() const { return(_use_b0_4_dwi); }
  
  void SetUseB0sToInformDWIRegistration(bool use_b0_4_dwi) { _use_b0_4_dwi = use_b0_4_dwi; }
  
  bool HasSuscHzOffResField() const { return(_has_susc_field); }
  
  bool HasSuscHzOffResDerivField() const EddyTry { return(this->has_move_by_susc_fields()); } EddyCatch
  
  bool HasBiasField() const { return(_bias_field != nullptr); }
  
  bool HasFieldOffset(ScanType st) const EddyTry { 
    if (st==B0 || st==DWI) return(Scan(0,st).HasFieldOffset()); 
    else return(Scan(0,B0).HasFieldOffset() || Scan(0,DWI).HasFieldOffset());
  } EddyCatch
  
  bool CanDoLSRResampling() const;
  
  std::pair<unsigned int,unsigned int> GetLSRPair(unsigned int i, ScanType st) const;
  
  void SetParameters(const NEWMAT::Matrix& pM, ScanType st=ANY);
  
  void SetParameters(const std::string& fname, ScanType st=ANY) EddyTry { NEWMAT::Matrix pM = MISCMATHS::read_ascii_matrix(fname); SetParameters(pM,st); } EddyCatch
  
  const ECScan& Scan(unsigned int indx, ScanType st=ANY) const;
  
  ECScan& Scan(unsigned int indx, ScanType st=ANY);
  
  NEWIMAGE::volume<float> GetUnwarpedOrigScan(unsigned int              indx,
					      NEWIMAGE::volume<float>&  omask,
					      ScanType                  st=ANY) const; 
  NEWIMAGE::volume<float> GetUnwarpedOrigScan(unsigned int indx,
					      ScanType     st=ANY) const EddyTry { 
    NEWIMAGE::volume<float> mask=_scans[0].GetIma(); mask=1.0; 
    return(GetUnwarpedOrigScan(indx,mask,st));
  } EddyCatch
  
  NEWIMAGE::volume<float> GetUnwarpedScan(unsigned int              indx,
					  NEWIMAGE::volume<float>&  omask,
					  ScanType                  st=ANY) const; 
  NEWIMAGE::volume<float> GetUnwarpedScan(unsigned int indx,
					  ScanType     st=ANY) const EddyTry { 
    NEWIMAGE::volume<float> mask=_scans[0].GetIma(); mask=1.0; 
    return(GetUnwarpedScan(indx,mask,st));
  } EddyCatch
  
  NEWIMAGE::volume<float> LSRResamplePair(
					  unsigned int              i, 
					  unsigned int              j, 
					  ScanType                  st,
					  
					  NEWIMAGE::volume<float>&  omask) const;
  
  void SetScanParameters(unsigned int indx, const NEWMAT::ColumnVector& p, ScanType st=ANY) EddyTry { Scan(indx,st).SetParams(p,ALL); } EddyCatch
  
  void AddRotation(const std::vector<float>& rot);
  
  const NEWIMAGE::volume<float>& Mask() const EddyTry { return(_mask); } EddyCatch
  
  std::vector<unsigned int> IntraCerebralSlices(unsigned int nvox) const;
  
  std::shared_ptr<const NEWIMAGE::volume<float> > GetSuscHzOffResField() const EddyTry { return(_susc_field); } EddyCatch
  
  std::shared_ptr<const NEWIMAGE::volume<float> > GetSuscHzOffResField(unsigned int indx, ScanType st=ANY) const;
  
  std::shared_ptr<const NEWIMAGE::volume<float> > GetBiasField() const EddyTry { return(_bias_field); } EddyCatch
  
  void SetDerivSuscField(unsigned int pi, const NEWIMAGE::volume<float>& dfield);
  
  void Set2ndDerivSuscField(unsigned int pi, unsigned int pj, const NEWIMAGE::volume<float>& dfield);
  
  void SetBiasField(const NEWIMAGE::volume<float>& bfield) EddyTry { _bias_field = std::shared_ptr<NEWIMAGE::volume<float> >(new NEWIMAGE::volume<float>(bfield)); } EddyCatch
  
  void ResetBiasField() { _bias_field = nullptr; }
  
  NEWIMAGE::volume<float> GetScanHzECOffResField(unsigned int indx, ScanType st=ANY) const EddyTry { return(Scan(indx,st).ECField()); } EddyCatch
  
  void SeparateFieldOffsetFromMovement(ScanType      st,
				       OffsetModel   m=LinearOffset);
  
  void RecycleOutliers() EddyTry { for (unsigned int s=0; s<NScans(); s++) Scan(s).RecycleOutliers(); } EddyCatch
  
  void SetPredictedECParam(ScanType st, SecondLevelECModel slm);
  
  void SetLocationReference(unsigned int ref=0) EddyTry { _refs.SetLocationReference(ref); } EddyCatch
  
  void SetDWILocationReference(unsigned int ref) EddyTry { _refs.SetDWILocationReference(ref); } EddyCatch
  
  void SetB0LocationReference(unsigned int ref) EddyTry { _refs.SetB0LocationReference(ref); } EddyCatch
  
  void SetShellLocationReference(unsigned int si, unsigned int ref) EddyTry { _refs.SetShellLocationReference(si,ref); } EddyCatch
  
  void SetB0ShapeReference(unsigned int ref) EddyTry { _refs.SetB0ShapeReference(ref); } EddyCatch
  
  void SetShellShapeReference(unsigned int si, unsigned int ref) EddyTry { _refs.SetShellShapeReference(si,ref); } EddyCatch
  
  void ApplyB0ShapeReference() EddyTry { set_slice_to_vol_reference(_refs.GetB0ShapeReference(),B0); } EddyCatch
  
  void ApplyB0LocationReference() EddyTry { set_reference(GetGlobal2b0IndexMapping(_refs.GetB0LocationReference()),B0); } EddyCatch
  
  void ApplyShellShapeReference(unsigned int si) EddyTry { set_slice_to_vol_reference(_refs.GetShellShapeReference(si),DWI,si); } EddyCatch
  
  void ApplyDWILocationReference() EddyTry { set_reference(GetGlobal2DWIIndexMapping(_refs.GetDWILocationReference()),DWI); } EddyCatch
  
  void ApplyLocationReference() EddyTry { set_reference(_refs.GetLocationReference(),ANY); } EddyCatch

  
  void WriteRegisteredImages(const std::string& fname, const std::string& maskfname, FinalResampling resmethod, double LSR_lambda, bool mask_output, ScanType st=ANY) EddyTry
  {
    if (resmethod==EDDY::LSR && !mask_output) throw EddyException("ECScanManager::WriteRegisteredImages: Must mask images when resampling method is LSR");
    PolationPara old_pp = this->GetPolation(); 
    PolationPara pp(NEWIMAGE::spline,NEWIMAGE::periodic,true,NEWIMAGE::spline);
    this->SetPolation(pp);
    if (resmethod==EDDY::JAC) write_jac_registered_images(fname,maskfname,mask_output,st);
    else if (resmethod==EDDY::LSR) {
      write_lsr_registered_images(fname,LSR_lambda,st);
    }
    else throw EddyException("ECScanManager::WriteRegisteredImages: Unknown resampling method");
    this->SetPolation(old_pp);
  } EddyCatch
  
  void WriteRegisteredImages(const std::string& fname, const std::string& maskfname, FinalResampling resmethod, double LSR_lambda, bool mask_output, const NEWIMAGE::volume4D<float>& pred, ScanType st=ANY) EddyTry
  {
    if (resmethod==EDDY::LSR && !mask_output) throw EddyException("ECScanManager::WriteRegisteredImages: Must mask images when resampling method is LSR");
    if (pred.tsize() != int(NScans(st))) throw EddyException("ECScanManager::WriteRegisteredImages: Size mismatch between pred and NScans");
    PolationPara old_pp = this->GetPolation(); 
    PolationPara pp(NEWIMAGE::spline,NEWIMAGE::periodic,true,NEWIMAGE::spline);
    this->SetPolation(pp);
    if (resmethod==EDDY::JAC) write_jac_registered_images(fname,maskfname,mask_output,pred,st);
    else if (resmethod==EDDY::LSR) {
      write_lsr_registered_images(fname,LSR_lambda,st);
    }
    else throw EddyException("ECScanManager::WriteRegisteredImages: Unknown resampling method");
    this->SetPolation(old_pp);
  } EddyCatch
  
  void WriteParameterFile(const std::string& fname, ScanType st=ANY) const;
  
  void WriteMovementOverTimeFile(const std::string& fname, ScanType st=ANY) const;
  
  void WriteECFields(const std::string& fname, ScanType st=ANY) const;
  
  void WriteRotatedBVecs(const std::string& fname, ScanType st=ANY) const;
  
  void WriteMovementRMS(const std::string& fname, ScanType st=ANY) const;
  
  void WriteRestrictedMovementRMS(const std::string& fname, ScanType st=ANY) const;
  
  void WriteDisplacementFields(const std::string& basefname, ScanType st=ANY) const;
  
  void WriteOutlierFreeData(const std::string& fname, ScanType st=ANY) const;
private:
  bool                                                                  _has_susc_field; 
  std::shared_ptr<NEWIMAGE::volume<float> >                             _susc_field;     
  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > >               _susc_d1;        
  std::vector<std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > > _susc_d2;        
  std::shared_ptr<NEWIMAGE::volume<float> >                             _bias_field;     
  NEWIMAGE::volume<float>                                               _mask;           
  double                                                                _sf;             
  std::vector<pair<int,int> >                                           _fi;             
  std::vector<ECScan>                                                   _scans;          
  std::vector<ECScan>                                                   _b0scans;        
  ReferenceScans                                                        _refs;           
  PolationPara                                                          _pp;             
  bool                                                                  _fsh;            
  bool                                                                  _use_b0_4_dwi;   

  
  NEWMAT::ColumnVector hz_vector_with_everything(ScanType st) const;
  
  NEWMAT::Matrix linear_design_matrix(ScanType st) const;
  
  NEWMAT::Matrix quadratic_design_matrix(ScanType st) const;
  
  NEWMAT::Matrix get_b0_movement_vector(ScanType st=DWI) const;
  
  void set_reference(unsigned int ref, ScanType st);
  
  void set_slice_to_vol_reference(unsigned int ref, ScanType st, int si=-1);
  double mean_of_first_b0(const NEWIMAGE::volume4D<float>&   vols,
                          const NEWIMAGE::volume<float>&     mask,
                          const NEWMAT::Matrix&              bvecs,
                          const NEWMAT::Matrix&              bvals) const;
  bool index_kosher(unsigned int indx, ScanType st) const EddyTry
  {
    if (st==DWI) return(indx<_scans.size());
    else if (st==B0) return(indx<_b0scans.size());
    else return(indx<_fi.size());
  } EddyCatch
  
  void set_polation(const PolationPara& pp) EddyTry { 
    for (unsigned int i=0; i<_scans.size(); i++) _scans[i].SetPolation(pp);
    for (unsigned int i=0; i<_b0scans.size(); i++) _b0scans[i].SetPolation(pp);
  } EddyCatch
  
  void write_jac_registered_images(const std::string& fname, const std::string& maskfname, bool mask_output, ScanType st) const;
  void write_jac_registered_images(const std::string& fname, const std::string& maskfname, bool mask_output, const NEWIMAGE::volume4D<float>& pred, ScanType st) const;
  void write_lsr_registered_images(const std::string& fname, double lambda, ScanType st) const;
  bool has_pe_in_direction(unsigned int dir, ScanType st=ANY) const;
  
  std::pair<int,int> bracket(unsigned int i, const std::vector<unsigned int>& ii) const;
  
  NEWMAT::ColumnVector interp_movpar(unsigned int i, const std::pair<int,int>& br) const; 
  
  NEWMAT::Matrix read_rb_matrix(const std::string& fname) const;
  bool indicies_clustered(const std::vector<unsigned int>& indicies,
			  unsigned int                     N) const;
  
  bool has_move_by_susc_fields() const;
};

} 

#endif 





























