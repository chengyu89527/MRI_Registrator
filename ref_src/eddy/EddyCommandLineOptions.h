#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include "utils/options.h"
#include "newimage/newimage.h"
#include "EddyHelperClasses.h"

#ifndef EddyCommandLineOptions_h
#define EddyCommandLineOptions_h

namespace EDDY {

 
class EddyInputError: public std::exception
{
public:
  EddyInputError(const std::string& msg) noexcept : m_msg(msg) {}
  ~EddyInputError() noexcept {}
  virtual const char * what() const noexcept { return string("EddyInputError:  " + m_msg).c_str(); }
private:
  std::string m_msg;
};

enum CovarianceFunctionType { Spherical, Exponential, NewSpherical, UnknownCF };


enum HyParCostFunctionType { MML, CV, GPP, CC, UnknownHyParCF };

class S2VParam {
public:
  S2VParam() EddyTry : _order(std::vector<int>(1,0)), _lambda(std::vector<float>(1,0.0)), 
	       _fwhm(std::vector<float>(1,0.0)), _niter(std::vector<int>(1,0)) {} EddyCatch
  S2VParam(const std::vector<int>& order, const std::vector<float>& lambda, 
	   const std::vector<float>& fwhm, const std::vector<int>& niter);
  unsigned int NOrder() const;
  unsigned int Order(unsigned int oi) const;
  std::vector<unsigned int> Order() const;
  unsigned int NIter(unsigned int oi) const;
  double Lambda(unsigned int oi) const;
  std::vector<double> Lambda() const;
  float FWHM(unsigned int oi, unsigned int iter) const;
  std::vector<float> FWHM(unsigned int oi) const;
private:
  std::vector<int>      _order;
  std::vector<float>    _lambda;
  std::vector<float>    _fwhm;
  std::vector<int>      _niter;
  unsigned int total_niter() const EddyTry { 
    unsigned int rval=0; 
    for (unsigned int i=0; i<_niter.size(); i++) rval += static_cast<unsigned int>(_niter[i]);
    return(rval);
  } EddyCatch
};

class DebugIndexClass {
public:
  DebugIndexClass(const std::string& in);
  DebugIndexClass(unsigned int frst, unsigned int lst) EddyTry { _indx.resize(lst-frst+1); for (unsigned int i=0; i<lst-frst+1; i++) _indx[i]=frst+i; } EddyCatch
  DebugIndexClass() : _indx(0) {} 
  bool IsAmongIndicies(unsigned int indx) const EddyTry { for (unsigned int i=0; i<_indx.size(); i++) if (indx==_indx[i]) return(true); return(false); } EddyCatch
  unsigned int Min() const EddyTry { 
    unsigned int rval=std::numeric_limits<unsigned int>::max(); for (unsigned int i=0; i<_indx.size(); i++) rval=min(rval,_indx[i]); return(rval); 
  } EddyCatch
  unsigned int Max() const EddyTry { 
    unsigned int rval=std::numeric_limits<unsigned int>::min(); for (unsigned int i=0; i<_indx.size(); i++) rval=max(rval,_indx[i]); return(rval); 
  } EddyCatch
  void SetIndicies(const std::vector<unsigned int>& indx) EddyTry { _indx=indx; } EddyCatch
private:
  std::vector<unsigned int> _indx;

  std::vector<unsigned int> parse_commaseparated_numbers(const std::string& list) const;
  std::vector<std::string> parse_commaseparated_list(const std::string&  list) const;
  unsigned int min(unsigned int i, unsigned int j) const { if (i<=j) return(i); else return(j); }
  unsigned int max(unsigned int i, unsigned int j) const { if (i>=j) return(i); else return(j); } 
};

class EddyCommandLineOptions {
public:
  EddyCommandLineOptions(int argc, char *argv[]);
  std::string ImaFname() const EddyTry { return(_imain.value()); } EddyCatch
  std::string MaskFname() const EddyTry { return(_mask.value()); } EddyCatch
  std::string AcqpFname() const EddyTry { return(_acqp.value()); } EddyCatch
  std::string IndexFname() const EddyTry { return(_index.value()); } EddyCatch
  std::string TopupFname() const EddyTry { return(_topup.value()); } EddyCatch
  std::string FieldFname() const EddyTry { return(_field.value()); } EddyCatch
  std::string BVecsFname() const EddyTry { return(_bvecs.value()); } EddyCatch
  std::string BValsFname() const EddyTry { return(_bvals.value()); } EddyCatch
  std::string ParOutFname() const EddyTry { return(_out.value()+std::string(".eddy_parameters")); } EddyCatch
  std::string MovementOverTimeOutFname() const EddyTry { return(_out.value()+std::string(".eddy_movement_over_time")); } EddyCatch
  std::string IOutFname() const EddyTry { return(_out.value()); } EddyCatch
  std::string OutMaskFname() const EddyTry { return((_dont_mask_output.value()) ? _out.value()+std::string(".eddy_output_mask") : std::string(""));} EddyCatch
  std::string ECFOutFname() const EddyTry { return(_out.value()+std::string(".eddy_fields")); } EddyCatch
  std::string MoveBySuscFirstOrderFname() const EddyTry { return(_out.value()+std::string(".eddy_mbs_first_order_fields")); } EddyCatch
  std::string MoveBySuscSecondOrderFname() const EddyTry { return(_out.value()+std::string(".eddy_mbs_second_order_fields")); } EddyCatch
  std::string RotatedBVecsOutFname() const EddyTry { return(_out.value()+std::string(".eddy_rotated_bvecs")); } EddyCatch
  std::string RMSOutFname() const EddyTry { return(_out.value()+std::string(".eddy_movement_rms")); } EddyCatch
  std::string RestrictedRMSOutFname() const EddyTry { return(_out.value()+std::string(".eddy_restricted_movement_rms")); } EddyCatch
  std::string DFieldOutFname() const EddyTry { return(_out.value()+std::string(".eddy_displacement_fields")); } EddyCatch
  std::string OLReportFname() const EddyTry { return(_out.value()+std::string(".eddy_outlier_report")); } EddyCatch
  std::string OLMapReportFname(bool problem=false) const EddyTry { 
    if (problem) return(_out.value()+std::string(".eddy_outlier_map_written_when_all_volumes_in_a_shell_has_outliers")); 
    else return(_out.value()+std::string(".eddy_outlier_map")); 
  } EddyCatch
  std::string OLNStDevMapReportFname(bool problem=false) const EddyTry { 
    if (problem) return(_out.value()+std::string(".eddy_outlier_n_stdev_map_written_when_all_volumes_in_a_shell_has_outliers")); 
    else return(_out.value()+std::string(".eddy_outlier_n_stdev_map"));
  } EddyCatch
  std::string OLNSqrStDevMapReportFname(bool problem=false) const EddyTry { 
    if (problem) return(_out.value()+std::string(".eddy_outlier_n_sqr_stdev_map_written_when_all_volumes_in_a_shell_has_outliers")); 
    else return(_out.value()+std::string(".eddy_outlier_n_sqr_stdev_map")); 
  } EddyCatch
  std::string OLFreeDataFname() const EddyTry { return(_out.value()+std::string(".eddy_outlier_free_data")); } EddyCatch
  std::string DwiMssHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_dwi_mss_history")); } EddyCatch
  std::string DwiParHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_dwi_parameter_history")); } EddyCatch
  std::string B0MssHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_b0_mss_history")); } EddyCatch
  std::string B0ParHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_b0_parameter_history")); } EddyCatch
  std::string DwiMssS2VHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_slice_to_vol_dwi_mss_history")); } EddyCatch
  std::string DwiParS2VHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_slice_to_vol_dwi_parameter_history")); } EddyCatch
  std::string B0MssS2VHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_slice_to_vol_b0_mss_history")); } EddyCatch
  std::string B0ParS2VHistoryFname() const EddyTry { return(_out.value()+std::string(".eddy_slice_to_vol_b0_parameter_history")); } EddyCatch
  std::string PeasReportFname() const EddyTry { return(_out.value()+std::string(".eddy_post_eddy_shell_alignment_parameters")); } EddyCatch
  std::string PeasAlongPEReportFname() const EddyTry { return(_out.value()+std::string(".eddy_post_eddy_shell_PE_translation_parameters")); } EddyCatch
  std::string CNROutFname() const EddyTry { return((_cnr_maps.value()) ? _out.value()+std::string(".eddy_cnr_maps") : std::string("")); } EddyCatch
  std::string RangeCNROutFname() const EddyTry { return((_range_cnr_maps.value()) ? _out.value()+std::string(".eddy_range_cnr_maps") : std::string("")); } EddyCatch
  std::string ResidualsOutFname() const EddyTry { return((_residuals.value()) ? _out.value()+std::string(".eddy_residuals") : std::string("")); } EddyCatch
  std::string AdditionalWithOutliersOutFname() const EddyTry { return(_out.value()+std::string(".eddy_results_with_outliers_retained")); } EddyCatch
  std::string MIPrintFname() const EddyTry { return(_out.value()+std::string(".eddy_PEAS_MI_values")); } EddyCatch
  std::string FieldMatFname() const EddyTry { return(_field_mat.value()); } EddyCatch
  std::vector<unsigned int> Indicies() const EddyTry { return(_indvec); } EddyCatch
  float FWHM(unsigned int iter) const;
  std::vector<float> FWHM() const EddyTry { return(_fwhm); } EddyCatch
  unsigned int NIter() const { return(_niter); } 
  void SetNIterAndFWHM(unsigned int niter, const std::vector<float>& fwhm);
  unsigned int S2V_NIter(unsigned int  oi) const EddyTry { return(_s2vparam.NIter(oi)); } EddyCatch
  void SetS2VParam(unsigned int order, float lambda, float fwhm, unsigned int niter);
  unsigned int Index(unsigned int i) const EddyTry { return(_indvec[i]); } EddyCatch
  EDDY::MultiBandGroups MultiBand() const;
  unsigned int NumOfNonZeroMovementModelOrder() const EddyTry { return(_s2vparam.NOrder()); } EddyCatch
  unsigned int MovementModelOrder(unsigned int oi) const EddyTry { return(_s2vparam.Order(oi)); } EddyCatch
  std::vector<unsigned int> MovementModelOrder() const EddyTry { return(_s2vparam.Order()); } EddyCatch
  double S2V_Lambda(unsigned int oi) const EddyTry { return(_s2vparam.Lambda(oi)); } EddyCatch
  std::vector<double> S2V_Lambda() const EddyTry { return(_s2vparam.Lambda()); } EddyCatch
  float S2V_FWHM(unsigned int oi, unsigned int iter) const EddyTry { return(_s2vparam.FWHM(oi,iter)); } EddyCatch
  std::vector<float> S2V_FWHM(unsigned int oi) const EddyTry { return(_s2vparam.FWHM(oi)); } EddyCatch
  bool IsSliceToVol() const EddyTry { return(NumOfNonZeroMovementModelOrder()>0); } EddyCatch
  bool EstimateMoveBySusc() const EddyTry { return(_estimate_mbs.value()); } EddyCatch
  unsigned int MoveBySuscNiter() const EddyTry { return(static_cast<unsigned int>(_mbs_niter.value())); } EddyCatch
  unsigned int N_MBS_Interleaves() const { return(3); } 
  double MoveBySuscLambda() const EddyTry { return(static_cast<double>(_mbs_lambda.value())); } EddyCatch
  std::vector<unsigned int> MoveBySuscParam() const EddyTry { std::vector<unsigned int> rval(2); rval[0] = 3; rval[1] = 4; return(rval); } EddyCatch
  unsigned int MoveBySuscOrder() const { return(1); } 
  bool MoveBySuscUseJacobian() const { return(true); }
  double MoveBySuscKsp() const EddyTry { return(static_cast<double>(_mbs_ksp.value())); } EddyCatch
  EDDY::ECModel FirstLevelModel() const;
  EDDY::ECModel b0_FirstLevelModel() const;
  EDDY::SecondLevelECModel SecondLevelModel() const { return(_slm); }
  EDDY::SecondLevelECModel b0_SecondLevelModel() const { return(_b0_slm); }
  void SetSecondLevelModel(EDDY::SecondLevelECModel slm) { _slm = slm; }
  void Set_b0_SecondLevelModel(EDDY::SecondLevelECModel b0_slm) { _b0_slm = b0_slm; }
  bool HasSecondLevelModel() const EddyTry { return(SecondLevelModel() != EDDY::No_2nd_lvl_mdl); } EddyCatch
  bool Has_b0_SecondLevelModel() const EddyTry { return(b0_SecondLevelModel() != EDDY::No_2nd_lvl_mdl); } EddyCatch
  bool SeparateOffsetFromMovement() const EddyTry { return(!_dont_sep_offs_move.value()); } EddyCatch
  EDDY::OffsetModel OffsetModel() const;
  bool ReplaceOutliers() const EddyTry { return(_rep_ol.value()); } EddyCatch
  bool AddNoiseToReplacements() const EddyTry { return(_rep_noise.value()); } EddyCatch
  bool WriteOutlierFreeData() const EddyTry { return(_rep_ol.value()); } EddyCatch
  const EDDY::OutlierDefinition& OLDef() const EddyTry { return(_oldef); } EddyCatch
  unsigned int OLErrorType() const EddyTry { return(_ol_ec.value()); } EddyCatch
  unsigned int RefScanNumber() const EddyTry { return(static_cast<unsigned int>(_ref_scan_no.value())); } EddyCatch
  EDDY::OLType OLType() const EddyTry { if (_ol_type.value()==string("sw")) return(EDDY::SliceWise); 
                                    else if (_ol_type.value()==string("gw")) return(EDDY::GroupWise);
                                    else return(EDDY::Both); } EddyCatch
  bool RegisterDWI() const { return(_rdwi); } 
  bool Registerb0() const { return(_rb0); } 
  bool WriteFields() const EddyTry { return(_fields.value()); } EddyCatch
  bool WriteMovementRMS() const { return(true); }
  bool WriteDisplacementFields() const EddyTry { return(_dfields.value()); } EddyCatch
  bool WriteRotatedBVecs() const { return(true); } 
  bool WriteCNRMaps() const EddyTry { return(_cnr_maps.value()); } EddyCatch
  bool WriteRangeCNRMaps() const EddyTry { return(_range_cnr_maps.value()); } EddyCatch
  bool WriteResiduals() const EddyTry { return(_residuals.value()); } EddyCatch
  bool WriteAdditionalResultsWithOutliersRetained() const EddyTry { return(_with_outliers.value()); } EddyCatch
  bool History() const EddyTry { return(_history.value()); } EddyCatch
  bool FillEmptyPlanes() const EddyTry { return(_fep.value()); } EddyCatch
  bool RotateBVecsDuringEstimation() const EddyTry { return(_rbvde.value()); } EddyCatch
  std::string InitFname() const EddyTry { return(_init.value()); } EddyCatch
  bool Verbose() const EddyTry { return(_verbose.value()); } EddyCatch
  bool VeryVerbose() const EddyTry { return(_very_verbose.value()); } EddyCatch
  bool WriteSliceStats() const EddyTry { return(_write_slice_stats.value()); } EddyCatch
  int DebugLevel() const { return(_debug); } 
  const DebugIndexClass& DebugIndicies() const EddyTry { return(_dbg_indx); } EddyCatch
  void SetDebug(unsigned int level, const std::vector<unsigned int>& indx) EddyTry { _debug=level; _dbg_indx.SetIndicies(indx); } EddyCatch
  FinalResampling ResamplingMethod() const;
  CovarianceFunctionType CovarianceFunction() const;
  HyParCostFunctionType HyParCostFunction() const;
  bool MaskOutput() const EddyTry { return(!_dont_mask_output.value()); } EddyCatch
  bool InitRand() const { return(_init_rand); }
  bool AlignShellsPostEddy() const EddyTry { return(!_dont_peas.value()); } EddyCatch
  bool UseB0sToAlignShellsPostEddy() const EddyTry { return(_use_b0s_for_peas.value()); } EddyCatch
  bool DontCheckShelling() const EddyTry { return(_data_is_shelled.value()); } EddyCatch
  unsigned int NVoxHp() const { return(_nvoxhp_internal); }
  void SetNVoxHp(unsigned int n) { _nvoxhp_internal = n; }
  double HyParFudgeFactor() const { return(_hypar_ff_internal); }
  void SetHyParFudgeFactor(double ff) { _hypar_ff_internal = ff; }
  bool HyperParFixed() const { return(_fixed_hpar); } 
  void SetHyperParFixed(bool val=true) { _fixed_hpar = val; }
  NEWMAT::ColumnVector HyperParValues() const;
  void SetHyperParValues(const NEWMAT::ColumnVector& p); 
  bool DoTestRot() const EddyTry { return(_test_rot.set()); } EddyCatch
  bool PrintMIValues() const EddyTry { return(_print_mi_values.value()); } EddyCatch
  bool PrintMIPlanes() const EddyTry { return(_print_mi_planes.value()); } EddyCatch
  std::vector<float> TestRotAngles() const;
  double LSResamplingLambda() const EddyTry { return(_lsr_lambda.value()); } EddyCatch
  NEWIMAGE::interpolation InterpolationMethod() const;
  NEWIMAGE::extrapolation ExtrapolationMethod() const;
  bool ExtrapolationValidInPE() const EddyTry { return(_epvalid.value()); } EddyCatch
  NEWIMAGE::interpolation S2VInterpolationMethod() const;
  EDDY::PolationPara PolationParameters() const EddyTry { 
    return(EDDY::PolationPara(InterpolationMethod(),ExtrapolationMethod(),ExtrapolationValidInPE(),S2VInterpolationMethod())); 
  } EddyCatch
private:
  std::string                             _title;
  std::string                             _examples;
  Utilities::Option<bool>                 _verbose;
  Utilities::Option<bool>                 _help;
  Utilities::Option<string>               _imain;
  Utilities::Option<string>               _mask;
  Utilities::Option<string>               _acqp;
  Utilities::Option<string>               _index;
  Utilities::HiddenOption<string>         _session;    
  Utilities::Option<int>                  _mb;
  Utilities::Option<int>                  _mb_offs;
  Utilities::HiddenOption<string>         _slorder;    
  Utilities::Option<string>               _slspec;
  Utilities::Option<string>               _json;
  Utilities::Option<vector<int> >         _mp_order;
  Utilities::Option<vector<float> >       _s2v_lambda;  
  Utilities::Option<string>               _topup;
  Utilities::Option<string>               _field;
  Utilities::Option<string>               _field_mat;
  Utilities::Option<string>               _bvecs;
  Utilities::Option<string>               _bvals;
  Utilities::Option<vector<float> >       _fwhm_tmp;
  Utilities::FmribOption<vector<float> >  _s2v_fwhm;
  Utilities::Option<int>                  _niter_tmp;
  Utilities::Option<vector<int> >         _s2v_niter;
  Utilities::Option<string>               _out;
  Utilities::Option<string>               _flm;
  Utilities::Option<string>               _slm_str;
  Utilities::FmribOption<string>          _b0_flm;
  Utilities::FmribOption<string>          _b0_slm_str;
  Utilities::Option<string>               _interp;
  Utilities::Option<string>               _s2v_interp;
  Utilities::FmribOption<string>          _extrap;
  Utilities::FmribOption<bool>            _epvalid;
  Utilities::Option<string>               _resamp;
  Utilities::FmribOption<string>          _covfunc;
  Utilities::FmribOption<string>          _hyparcostfunc;
  Utilities::Option<int>                  _nvoxhp;
  Utilities::Option<bool>                 _initrand;
  Utilities::Option<float>                _hyparfudgefactor;
  Utilities::FmribOption<vector<float> >  _hypar;
  Utilities::Option<bool>                 _rep_ol;
  Utilities::FmribOption<bool>            _rep_noise;
  Utilities::Option<float>                _ol_nstd;
  Utilities::Option<int>                  _ol_nvox;
  Utilities::FmribOption<int>             _ol_ec;
  Utilities::Option<string>               _ol_type;
  Utilities::Option<bool>                 _ol_pos;
  Utilities::Option<bool>                 _ol_sqr;
  Utilities::Option<bool>                 _estimate_mbs;
  Utilities::Option<int>                  _mbs_niter;
  Utilities::Option<float>                _mbs_lambda;
  Utilities::Option<float>                _mbs_ksp;
  Utilities::HiddenOption<bool>           _sep_offs_move;       
  Utilities::Option<bool>                 _dont_sep_offs_move;     
  Utilities::FmribOption<string>          _offset_model;
  Utilities::HiddenOption<bool>           _peas;                
  Utilities::Option<bool>                 _dont_peas;
  Utilities::FmribOption<bool>            _use_b0s_for_peas;
  Utilities::Option<bool>                 _data_is_shelled;
  Utilities::FmribOption<bool>            _very_verbose;
  Utilities::FmribOption<bool>            _dwi_only;
  Utilities::FmribOption<bool>            _b0_only;
  Utilities::FmribOption<bool>            _fields;
  Utilities::HiddenOption<bool>           _rms;                 
  Utilities::FmribOption<bool>            _dfields;
  Utilities::Option<bool>                 _cnr_maps;
  Utilities::FmribOption<bool>            _range_cnr_maps;
  Utilities::Option<bool>                 _residuals;
  Utilities::FmribOption<bool>            _with_outliers;
  Utilities::FmribOption<bool>            _history;
  Utilities::Option<bool>                 _fep;
  Utilities::FmribOption<bool>            _dont_mask_output;
  Utilities::FmribOption<bool>            _write_slice_stats;
  Utilities::FmribOption<string>          _init;
  Utilities::FmribOption<int>             _debug_tmp;
  Utilities::FmribOption<string>          _dbg_indx_str;
  Utilities::FmribOption<float>           _lsr_lambda;
  Utilities::FmribOption<int>             _ref_scan_no;
  Utilities::FmribOption<bool>            _rbvde;
  Utilities::HiddenOption<vector<float> > _test_rot;
  Utilities::HiddenOption<bool>           _print_mi_values;
  Utilities::HiddenOption<bool>           _print_mi_planes;
  bool                                    _rdwi;
  bool                                    _rb0;
  bool                                    _init_rand;
  bool                                    _fixed_hpar;
  double                                  _hypar_ff_internal;
  unsigned int                            _nvoxhp_internal;
  std::vector<float>                      _hypar_internal;
  std::vector<unsigned int>               _indvec;
  unsigned int                            _debug;
  DebugIndexClass                         _dbg_indx;
  EDDY::OutlierDefinition                 _oldef;
  EDDY::SecondLevelECModel                _slm;
  EDDY::SecondLevelECModel                _b0_slm;
  unsigned int                            _niter;
  std::vector<float>                      _fwhm;
  EDDY::S2VParam                          _s2vparam;

  void do_initial_parsing(int argc, char *argv[]);
  bool indicies_kosher(NEWMAT::Matrix indx, NEWMAT::Matrix acqp);
  std::vector<unsigned int> get_slorder(const std::string& fname,
					unsigned int       ngrp) const;
};

} 

#endif 

