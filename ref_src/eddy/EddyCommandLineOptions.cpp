
// Definitions of a class that does the parsing and
// sanity checking of commnad line options for the 
// "eddy" application.
// 
// EddyCommandLineOptions.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include "utils/options.h"
#include "topup/topup_file_io.h"
#include "EddyHelperClasses.h"
#include "EddyCommandLineOptions.h"

using namespace EDDY;

EddyCommandLineOptions::EddyCommandLineOptions(int  argc, char *argv[]) try : 
  _title("eddy \nCopyright(c) 2015, University of Oxford (Jesper Andersson)"),
  _examples("eddy --monsoon"),
  _verbose(string("-v,--verbose"),false,string("switch on diagnostic messages"),false, Utilities::no_argument), 
  _help(string("-h,--help"),false,string("display this message"),false, Utilities::no_argument),
  _imain(string("--imain"),string(""),string("File containing all the images to estimate distortions for"),true,Utilities::requires_argument), 
  _mask(string("--mask"),string(""),string("Mask to indicate brain"),true,Utilities::requires_argument),
  _acqp(string("--acqp"),string(""),string("File containing acquisition parameters"),true,Utilities::requires_argument),
  _index(string("--index"),string(""),string("File containing indices for all volumes in --imain into --acqp and --topup"),true,Utilities::requires_argument),
  _session(string("--session"),string(""),string("File containing session indices for all volumes in --imain"),false,Utilities::requires_argument),
  _mb(string("--mb"),1,string("Multi-band factor"),false,Utilities::requires_argument),
  _mb_offs(string("--mb_offs"),0,string("Multi-band offset (-1 if bottom slice removed, 1 if top slice removed)"),false,Utilities::requires_argument),
  _slorder(string("--slorder"),string(""),string("Name of text file defining slice/group order"),false,Utilities::requires_argument),
  _slspec(string("--slspec"),string(""),string("Name of text file completely specifying slice/group acuistion. N.B. --slspec and --json are mutually exclusive."),false,Utilities::requires_argument),
  _json(string("--json"),string(""),string("Name of .json text file with information about slice timing. N.B. --json and --slspec are mutually exclusive."),false,Utilities::requires_argument),
  _mp_order(string("--mporder"),std::vector<int>(1,0),string("Order of slice-to-vol movement model"),false,Utilities::requires_argument),
  _s2v_lambda(string("--s2v_lambda"),std::vector<float>(1,1.0),string("Regularisation weight for slice-to-vol movement. (default 1, reasonable range 1--10"),false,Utilities::requires_argument),
  _topup(string("--topup"),string(""),string("Base name for output files from topup"),false,Utilities::requires_argument),
  _field(string("--field"),string(""),string("Name of file with susceptibility field (in Hz)"),false,Utilities::requires_argument),
  _field_mat(string("--field_mat"),string(""),string("Name of rigid body transform for susceptibility field"),false,Utilities::requires_argument),
  _bvecs(string("--bvecs"),string(""),string("File containing the b-vectors for all volumes in --imain"),true,Utilities::requires_argument),
  _bvals(string("--bvals"),string(""),string("File containing the b-values for all volumes in --imain"),true,Utilities::requires_argument),
  _fwhm_tmp(string("--fwhm"),std::vector<float>(1,0.0),string("FWHM for conditioning filter when estimating the parameters (default 0)"),false,Utilities::requires_argument),
  _s2v_fwhm(string("--s2v_fwhm"),std::vector<float>(1,0.0),string("FWHM for conditioning filter when estimating slice-to-vol parameters (default 0)"),false,Utilities::requires_argument),
  _niter_tmp(string("--niter"),5,string("Number of iterations (default 5)"),false,Utilities::requires_argument),
  _s2v_niter(string("--s2v_niter"),std::vector<int>(1,5),string("Number of iterations for slice-to-vol (default 5)"),false,Utilities::requires_argument),
  _out(string("--out"),string(""),string("Basename for output"),true,Utilities::requires_argument),
  _flm(string("--flm"),string("quadratic"),string("First level EC model (movement/linear/quadratic/cubic, default quadratic)"),false,Utilities::requires_argument),
  _slm_str(string("--slm"),string("none"),string("Second level EC model (none/linear/quadratic, default none)"),false,Utilities::requires_argument),
  _b0_flm(string("--b0_flm"),string("movement"),string("First level EC model for b0 scans (movement/linear/quadratic, default movement)"),false,Utilities::requires_argument),
  _b0_slm_str(string("--b0_slm"),string("none"),string("Second level EC model for b0 scans (none/linear/quadratic, default none)"),false,Utilities::requires_argument),
  _interp(string("--interp"),string("spline"),string("Interpolation model for estimation step (spline/trilinear, default spline)"),false,Utilities::requires_argument),
  _s2v_interp(string("--s2v_interp"),string("trilinear"),string("Slice-to-vol interpolation model for estimation step (spline/trilinear, default trilinear)"),false,Utilities::requires_argument),
  _extrap(string("--extrap"),string("periodic"),string("Extrapolation model for estimation step (periodic/mirror, default periodic)"),false,Utilities::requires_argument),
  _epvalid(string("--epvalid"),false,string("Indicates that extrapolation is valid in EP direction (default false)"),false, Utilities::no_argument), 
  _resamp(string("--resamp"),string("jac"),string("Final resampling method (jac/lsr, default jac)"),false, Utilities::requires_argument), 
  _covfunc(string("--covfunc"),string("spheri"),string("Covariance function for GP (spheri/expo/old, default spheri)"),false, Utilities::requires_argument), 
  _hyparcostfunc(string("--hpcf"),string("CV"),string("Cost-function for GP hyperparameters (MML/CV/GPP/CC, default CV)"),false, Utilities::requires_argument), 
  _nvoxhp(string("--nvoxhp"),1000,string("# of voxels used to estimate the hyperparameters (default 1000)"),false, Utilities::requires_argument), 
  _initrand(string("--initrand"),false,string("Resets rand for when selecting voxels (default false)"),false, Utilities::no_argument), 
  _hyparfudgefactor(string("--ff"),10.0,string("Fudge factor for hyperparameter error variance (default 10.0)"),false, Utilities::requires_argument), 
  _hypar(string("--hypar"),std::vector<float>(0),string("User specified values for GP hyperparameters"),false,Utilities::requires_argument),
  _rep_ol(string("--repol"),false,string("Detect and replace outlier slices (default false))"),false, Utilities::no_argument), 
  _rep_noise(string("--rep_noise"),false,string("Add noise to replaced outliers (default false)"),false, Utilities::no_argument), 
  _ol_nstd(string("--ol_nstd"),4.0,string("Number of std off to qualify as outlier (default 4)"),false,Utilities::requires_argument),
  _ol_nvox(string("--ol_nvox"),250,string("Min # of voxels in a slice for inclusion in outlier detection (default 250)"),false,Utilities::requires_argument),
  _ol_ec(string("--ol_ec"),1,string("Error type (1 or 2) to keep constant for outlier detection (default 1)"),false,Utilities::requires_argument),
  _ol_type(string("--ol_type"),string("sw"),string("Type of outliers, slicewise (sw), groupwise (gw) or both (both). (default sw)"),false,Utilities::requires_argument),
  _ol_pos(string("--ol_pos"),false,string("Consider both positive and negative outliers if set (default false)"),false,Utilities::no_argument),
  _ol_sqr(string("--ol_sqr"),false,string("Consider outliers among sums-of-squared differences if set (default false)"),false,Utilities::no_argument),
  _estimate_mbs(string("--estimate_move_by_susceptibility"),false,string("Estimate how susceptibility field changes with subject movement (default false)"),false,Utilities::no_argument),
  _mbs_niter(string("--mbs_niter"),10,string("Number of iterations for MBS estimation (default 10)"),false,Utilities::requires_argument),
  _mbs_lambda(string("--mbs_lambda"),10.0,string("Weighting of regularisation for MBS estimation (default 10)"),false,Utilities::requires_argument),
  _mbs_ksp(string("--mbs_ksp"),10.0,string("Knot-spacing for MBS field estimation (default 10mm)"),false,Utilities::requires_argument),
  _sep_offs_move(string("--sep_offs_move"),true,string("Attempt to separate field offset from subject movement (deprecated, its use will crash future versions)"),false, Utilities::no_argument), 
  _dont_sep_offs_move(string("--dont_sep_offs_move"),false,string("Do NOT attempt to separate field offset from subject movement (default false)"),false, Utilities::no_argument), 
  _offset_model(string("--offset_model"),string("linear"),string("Second level model for field offset"),false, Utilities::requires_argument), 
  _peas(string("--peas"),true,string("Perform a post-eddy alignment of shells (deprecated, its use will crash future versions)"),false, Utilities::no_argument), 
  _dont_peas(string("--dont_peas"),false,string("Do NOT perform a post-eddy alignment of shells (default false)"),false, Utilities::no_argument),
  _use_b0s_for_peas(string("--b0_peas"),false,string("Use interspersed b0s to perform post-eddy alignment of shells (default false)"),false, Utilities::no_argument),
  _data_is_shelled(string("--data_is_shelled"),false,string("Assume, don't check, that data is shelled (default false)"),false, Utilities::no_argument), 
  _very_verbose(string("--very_verbose"),false,string("Switch on detailed diagnostic messages (default false)"),false, Utilities::no_argument), 
  _dwi_only(string("--dwi_only"),false,string("Only register the dwi images (default false)"),false, Utilities::no_argument), 
  _b0_only(string("--b0_only"),false,string("Only register the b0 images (default false)"),false, Utilities::no_argument), 
  _fields(string("--fields"),false,string("Write EC fields as images (default false)"),false, Utilities::no_argument), 
  _rms(string("--rms"),true,string("Write movement induced RMS (deprecated, its use will crash future versions)"),false, Utilities::no_argument), 
  _dfields(string("--dfields"),false,string("Write total displacement fields (default false)"),false, Utilities::no_argument), 
  _cnr_maps(string("--cnr_maps"),false,string("Write shell-wise cnr-maps (default false)"),false, Utilities::no_argument), 
  _range_cnr_maps(string("--range_cnr_maps"),false,string("Write shell-wise range-cnr-maps (default false)"),false, Utilities::no_argument), 
  _residuals(string("--residuals"),false,string("Write residuals (between GP and observations), (default false)"),false, Utilities::no_argument), 
  _with_outliers(string("--with_outliers"),false,string("Write corrected data (additionally) with outliers retained (default false)"),false, Utilities::no_argument), 
  _history(string("--history"),false,string("Write history of mss and parameter estimates (default false)"),false, Utilities::no_argument), 
  _fep(string("--fep"),false,string("Fill empty planes in x- or y-directions (default false)"),false, Utilities::no_argument),
  _dont_mask_output(string("--dont_mask_output"),false,string("Do not mask output to include only voxels present for all volumes (default false)"),false, Utilities::no_argument),
  _write_slice_stats(string("--wss"),false,string("Write slice-wise stats for each iteration (default false)"),false,Utilities::no_argument), 
  _init(string("--init"),string(""),string("Text file with parameters for initialisation"),false,Utilities::requires_argument),
  _debug_tmp(string("--debug"),0,string("Level of debug print-outs (default 0)"),false,Utilities::requires_argument),
  _dbg_indx_str(string("--dbgindx"),string(""),string("Indicies (zero offset) of volumes for debug print-outs"),false,Utilities::requires_argument),
  _lsr_lambda(string("--lsr_lambda"),0.01,string("Regularisation weight for LSR-resampling."),false,Utilities::requires_argument),
  _ref_scan_no(string("--ref_scan_no"),0,string("Zero-offset # of ref (for location) volume (default 0)"),false,Utilities::requires_argument),
  _rbvde(string("--rbvde"),false,string("Rotate b-vecs during estimation (default false)"),false,Utilities::no_argument),
  _test_rot(string("--test_rot"),std::vector<float>(0),string("Do a large rotation to test b-vecs"),false,Utilities::requires_argument),
  _print_mi_values(string("--pmiv"),false,string("Write text file of MI values between shells (default false)"),false,Utilities::no_argument),
  _print_mi_planes(string("--pmip"),false,string("Write text file of (2D) MI values between shells (default false)"),false,Utilities::no_argument),
  _rdwi(true), _rb0(true)
{
  
  do_initial_parsing(argc,argv);

  
  
  NEWIMAGE::volume4D<float> imahdr;
  NEWIMAGE::read_volume_hdr_only(imahdr,_imain.value());  
  NEWIMAGE::volume<float> maskhdr;
  NEWIMAGE::read_volume_hdr_only(maskhdr,_mask.value());    
  if (!samesize(imahdr[0],maskhdr,3,true)) throw EddyInputError("--imain and --mask images must have the same dimensions");
  
  NEWMAT::Matrix index;
  try {
    index = MISCMATHS::read_ascii_matrix(_index.value());
    if (std::min(index.Nrows(),index.Ncols()) != 1 || std::max(index.Nrows(),index.Ncols()) != imahdr.tsize()) {
      throw EddyInputError("--index must be an 1xN or Nx1 matrix where N is the number of volumes in --imain");
    }
  }
  catch (...) { throw EddyInputError("Error when attempting to read --index file"); }
  
  if (_session.value().size()) {
    cout << "Warning: --session parameter is deprecated";
  }
  
  NEWMAT::Matrix acqpM;
  try {
    acqpM = MISCMATHS::read_ascii_matrix(_acqp.value());
    if (acqpM.Ncols() != 4) throw EddyInputError("Each row of the --acqp file must contain 4 values");
  }
  catch (...) { throw EddyInputError("Error when attempting to read --acqp file"); }
  if (!this->indicies_kosher(index,acqpM)) throw EddyInputError("Mismatch between --index and --acqp");
  
  try {
    NEWMAT::Matrix bvecsM = MISCMATHS::read_ascii_matrix(_bvecs.value());
    if (std::min(bvecsM.Nrows(),bvecsM.Ncols()) != 3 || std::max(bvecsM.Nrows(),bvecsM.Ncols()) != imahdr.tsize()) {
      throw EddyInputError("--bvecs should contain a 3xN or Nx3 matrix where N is the number of volumes in --imain");    
    }
  }
  catch (...) { throw EddyInputError("Error when attempting to read --bvecs file"); }
  try {
    NEWMAT::Matrix bvalsM = MISCMATHS::read_ascii_matrix(_bvals.value());
    if (std::min(bvalsM.Nrows(),bvalsM.Ncols()) != 1 || std::max(bvalsM.Nrows(),bvalsM.Ncols()) != imahdr.tsize()) {
      throw EddyInputError("--bvals should contain a 1xN or Nx1 matrix where N is the number of volumes in --imain");    
    }
  }
  catch (...) { throw EddyInputError("Error when attempting to read --bvals file"); }
  
  if (_topup.set() && _topup.value() != string("") && _field.set() && _field.value() != string("")) {
    throw EddyInputError("One cannot specify both --field and --topup file");    
  }
  
  if (_topup.set() && _topup.value() != string("")) {
    try { TOPUP::TopupFileReader  tfr(_topup.value()); }
    catch (const TOPUP::TopupFileIOException& e) { cout << e.what() << endl; throw EddyInputError("Error when attempting to read --topup file"); }
    catch (...) { throw EddyInputError("Error when attempting to read --topup files"); }
  }
  
  if (_field.set() && _field.value() != string("")) {
    try { TOPUP::TopupFileReader  tfr(_field.value()); }
    catch (const TOPUP::TopupFileIOException& e) { cout << e.what() << endl; throw EddyInputError("Error when attempting to read --field file"); }
    catch (...) { throw EddyInputError("Error when attempting to read --field file"); }
  }
  
  if (_field_mat.set() && _field_mat.value() != string("")) {
    try {
      NEWMAT::Matrix fieldM = MISCMATHS::read_ascii_matrix(_field_mat.value());
      if (fieldM.Nrows() != 4 || fieldM.Ncols() != 4) throw EddyInputError("--field_mat must be a 4x4 matrix");    
      NEWMAT::Matrix ul = fieldM.SubMatrix(1,3,1,3);
      float det = ul.Determinant();
      if (std::abs(det-1.0) > 1e-6) throw EddyInputError("--field_mat must be a rigid body transformation");    
    }
    catch (...) { throw EddyInputError("Error when attempting to read --field_mat file"); }
  }
  
  if (_interp.value() != string("spline") && _interp.value() != string("trilinear")) throw EddyInputError("Invalid --interp parameter");
  
  if (_s2v_interp.value() != string("spline") && _s2v_interp.value() != string("trilinear")) throw EddyInputError("Invalid --s2v_interp parameter");
  
  if (_extrap.value() != string("mirror") && _extrap.value() != string("periodic")) throw EddyInputError("Invalid --extrap parameter");
  
  if (_extrap.value() == string("mirror") && _epvalid.value()) throw EddyInputError("--extrap=mirror cannot be combined with --epvalid");
  
  if (_resamp.value() != string("jac") && _resamp.value() != string("lsr")) throw EddyInputError("Invalid --resamp parameter");
  
  if (_resamp.value() == string("lsr") && !_fep.value()) throw EddyInputError("You need to specify --fep with --resamp=lsr");
  
  if (_resamp.value() == string("lsr") && _dont_mask_output.value()) throw EddyInputError("You cannot combine --resamp=lsr with --dont_mask_output"); 
  
  if (_covfunc.value() != string("spheri") && _covfunc.value() != string("expo") && _covfunc.value() != string("old")) throw EddyInputError("Invalid --covfunc parameter"); 
  
  if (_hyparcostfunc.value() != string("MML") && _hyparcostfunc.value() != string("CV") && _hyparcostfunc.value() != string("GPP") && _hyparcostfunc.value() != string("CC")) throw EddyInputError("Invalid --hpcf parameter");
  
  if (_hypar.value().size()) { _fixed_hpar = true; _hypar_internal = _hypar.value(); } else { _fixed_hpar = false; }
  
  if (_flm.value() != string("movement") && _flm.value() != string("linear") && _flm.value() != string("quadratic") && _flm.value() != string("cubic")) throw EddyInputError("Invalid --flm parameter");
  
  if (_slm_str.value() == string("none")) _slm = EDDY::No_2nd_lvl_mdl;
  else if (_slm_str.value() == string("linear")) _slm = EDDY::Linear_2nd_lvl_mdl;
  else if (_slm_str.value() == string("quadratic")) _slm = EDDY::Quadratic_2nd_lvl_mdl;
  else throw EddyInputError("Invalid --slm parameter");
  
  if (_b0_flm.value() != string("movement") && _b0_flm.value() != string("linear") && _b0_flm.value() != string("quadratic")) throw EddyInputError("Invalid --b0_flm parameter");
  
  if (_b0_slm_str.value() == string("none")) _b0_slm = EDDY::No_2nd_lvl_mdl;
  else if (_b0_slm_str.value() == string("linear")) _b0_slm = EDDY::Linear_2nd_lvl_mdl;
  else if (_b0_slm_str.value() == string("quadratic")) _b0_slm = EDDY::Quadratic_2nd_lvl_mdl;
  else throw EddyInputError("Invalid --b0_slm parameter");
  
  if (_offset_model.value() != string("linear") && _offset_model.value() != string("quadratic")) throw EddyInputError("Invalid --offset_model parameter");
  
  _niter = static_cast<unsigned int>(_niter_tmp.value());
  _fwhm = _fwhm_tmp.value();
  if (_fwhm.size() != 1 && _fwhm.size() != _niter) throw EddyInputError("--fwhm must be one value or one per iteration");
  if (_fwhm.size() != _niter) _fwhm.resize(_niter,_fwhm[0]);
  
  for (unsigned int i=0; i<_fwhm.size(); i++) {
    if (_fwhm[i] < 0.0 || _fwhm[i] > 20.0) throw EddyInputError("--fwhm value outside valid range 0-20mm");
  }
  
  _s2vparam = EDDY::S2VParam(_mp_order.value(),_s2v_lambda.value(),_s2v_fwhm.value(),_s2v_niter.value());
  
  if (_nvoxhp.value() < 100  || _nvoxhp.value() > 50000) throw EddyInputError("--nvoxhp value outside valid range 100-50000");
  _nvoxhp_internal = static_cast<unsigned int>(_nvoxhp.value());
  _init_rand = _initrand.value();
  
  if (!_mb.set() && _mb_offs.set()) throw EddyInputError("--mb_offs makes no sense without --mb");
  if ((_mb.set() + _slspec.set() + _json.set()) > 1) throw EddyInputError("--mb, --slspec and --json mutually exclusive");
  if (_slorder.set()) throw EddyInputError("--slorder has been deprecated");
  
  if (_mb.value() < 1 || _mb.value() > 10) throw EddyInputError("--mb value outside valid range 1-10");
  if (std::abs(_mb_offs.value()) > 1) throw EddyInputError("--mb_offs must be -1, 0 or 1");
  
  if (float(*std::max_element(_mp_order.value().begin(),_mp_order.value().end())) > float(imahdr.zsize())/float(_mb.value())) throw EddyInputError("--mporder can not be greater than number of slices/mb-groups");
  
  if (_ol_nstd.value() < 1.96) throw EddyInputError("--ol_nstd value too low (below 1.96)");
  if (_ol_nvox.value() < 250) throw EddyInputError("--ol_nvox value below 250");
  _oldef = EDDY::OutlierDefinition(_ol_nstd.value(),_ol_nvox.value(),_ol_pos.value(),_ol_sqr.value());
  if (_ol_ec.value() != 1 && _ol_ec.value() != 2) throw EddyInputError("--ol_ec value must be 1 or 2");
  if ((_ol_type.value() == string("gw") || _ol_type.value() == string("both")) && MultiBand().MBFactor() < 2) throw EddyInputError("--ol_type indicating mb-groups without providing mb structure");
  
  if (_hyparfudgefactor.value() < 1.0 || _hyparfudgefactor.value() > 10.0) throw EddyInputError("--ff value outside valid range 1.0-10.0");
  else _hypar_ff_internal = static_cast<double>(_hyparfudgefactor.value());
  
  if (_mbs_niter.value() < 1 || _mbs_niter.value() > 50) throw EddyInputError("--mbs_niter value outside valid range 1-50");
  if (_mbs_lambda.value() < 1.0 || _mbs_lambda.value() > 10000.0) throw EddyInputError("--mbs_lambda value outside valid range 1.0-10000.0");
  if (_mbs_ksp.value() < 2.0 || _mbs_ksp.value() > 30.0) throw EddyInputError("--mbs_ksp value outside valid range 2.0-30.0");
  
  if (_lsr_lambda.value() < 0.0 || _lsr_lambda.value() > 1.0) throw EddyInputError("--lsr_lambda value outside valid range");
  
  if (_very_verbose.value() && !_verbose.value()) _verbose.set_T(true);
  
  if (_b0_only.value()) throw EddyInputError("--b0_only disabled awaiting release of eddy for fmri (freddy)");
  
  if (_dwi_only.value() && _b0_only.value()) throw EddyInputError("--dwi_only and --b0_only cannot both be set");
  if (_dwi_only.value()) _rb0 = false;
  else if (_b0_only.value()) _rdwi = false;
  
  if (_test_rot.set()) {
    if (_test_rot.value().size() < 1 || _test_rot.value().size() > 3) throw EddyInputError("--test_rot must be one, two or three angles");
  }
  
  if (_init.set() && _init.value() != string("")) {
    try { NEWMAT::Matrix  tmp = MISCMATHS::read_ascii_matrix(_init.value()); }
    catch (...) { throw EddyInputError("Error when attempting to read --init file"); }
  }
  
  if (_ref_scan_no.value() < 0 || _ref_scan_no.value() > imahdr.tsize()-1) throw EddyInputError("--ref_scan_no out of range");
  
  if (_debug_tmp.value() < 0 || _debug_tmp.value() > 3) { throw EddyInputError("--debug must be a value between 0 and 3"); }
  _debug = static_cast<unsigned int>(_debug_tmp.value());
  if (_debug > 0 && (!_dbg_indx_str.set() || _dbg_indx_str.value() == string(""))) {
    _dbg_indx = DebugIndexClass(0,imahdr.tsize()-1);
  }
  else if (_dbg_indx_str.set() && _dbg_indx_str.value() != string("")) {
    _dbg_indx = DebugIndexClass(_dbg_indx_str.value());
  }
  if (_dbg_indx.Max() > static_cast<unsigned int>(imahdr.tsize()-1)) { throw EddyInputError("--dbg_indx out of range"); }
  
  if (_debug > 0) _init_rand = true;
  
  if (_dont_peas.value() && _use_b0s_for_peas.value()) throw EddyInputError("--dont_peas and --b0_peas cannot both be set");
  
  if (_with_outliers.value() && !_rep_ol.value()) throw EddyInputError("--with_outliers does not make sense without also specifying --repol");
  
  #ifndef COMPILE_GPU
  if (_mp_order.set() || _slorder.set() || _slspec.set() || _s2v_fwhm.set() || _s2v_niter.set() || _s2v_interp.set()) {
    throw EddyInputError("Slice-to-vol is not yet implemented for the CPU version.");
  }  
  #endif
} 
catch (const EddyInputError& e)
{
  cout << e.what() << endl;
  cout << "Terminating program" << endl;
  exit(EXIT_FAILURE);
} EddyCatch

EDDY::FinalResampling EddyCommandLineOptions::ResamplingMethod() const EddyTry
{
  if (_resamp.value() == std::string("jac")) return(EDDY::JAC);
  else if (_resamp.value() == std::string("lsr")) return(EDDY::LSR);
  else return(EDDY::UNKNOWN_RESAMPLING);
} EddyCatch

EDDY::CovarianceFunctionType EddyCommandLineOptions::CovarianceFunction() const EddyTry
{
  if (_covfunc.value() == std::string("spheri")) return(EDDY::NewSpherical);
  else if (_covfunc.value() == std::string("expo")) return(EDDY::Exponential);
  else if (_covfunc.value() == std::string("old")) return(EDDY::Spherical);
  else return(EDDY::UnknownCF);
} EddyCatch

EDDY::HyParCostFunctionType EddyCommandLineOptions::HyParCostFunction() const EddyTry
{
  if (_hyparcostfunc.value() == std::string("MML")) return(EDDY::MML);
  else if (_hyparcostfunc.value() == std::string("CV")) return(EDDY::CV);
  else if (_hyparcostfunc.value() == std::string("GPP")) return(EDDY::GPP);
  else if (_hyparcostfunc.value() == std::string("CC")) return(EDDY::CC);
  else return(EDDY::UnknownHyParCF);
} EddyCatch

float EddyCommandLineOptions::FWHM(unsigned int iter) const EddyTry
{
  if (_fwhm.size()==1) return(_fwhm[0]);
  else if (iter >= _fwhm.size()) throw EddyException("EddyCommandLineOptions::FWHM: iter out of range");
  return(_fwhm[iter]);
} EddyCatch

void EddyCommandLineOptions::SetNIterAndFWHM(unsigned int niter, const std::vector<float>& fwhm) EddyTry
{
  if (fwhm.size() != 1 && fwhm.size() != niter) throw EddyException("EddyCommandLineOptions::SetNIterAndFWHM: mismatch between niter and fwhm");
  _niter = niter; 
  if (fwhm.size() == niter) _fwhm = fwhm;
  else _fwhm = std::vector<float>(niter,fwhm[0]);
} EddyCatch

void EddyCommandLineOptions::SetS2VParam(unsigned int order, float lambda, float fwhm, unsigned int niter) EddyTry
{
  std::vector<int>   lorder(1,static_cast<int>(order));
  std::vector<float> llambda(1,lambda);
  std::vector<float> lfwhm(1,fwhm);
  std::vector<int>   lniter(1,static_cast<int>(niter));

  _s2vparam = EDDY::S2VParam(lorder,llambda,lfwhm,lniter); 
} EddyCatch

unsigned int S2VParam::NIter(unsigned int  oi) const EddyTry
{ 
  if (oi >= _niter.size()) throw EddyException("S2VParam::NIter: oi out of range");
  return(static_cast<unsigned int>(_niter[oi])); 
} EddyCatch

unsigned int S2VParam::NOrder() const EddyTry
{
  if (_order.size() > 1) return(_order.size());
  else return(static_cast<unsigned int>(_order[0] > 0 ? 1 : 0));
} EddyCatch

unsigned int S2VParam::Order(unsigned int oi) const EddyTry
{
  if (oi >= _order.size()) throw EddyException("S2VParam::Order: oi out of range");
  return(static_cast<unsigned int>(_order[oi]));
} EddyCatch

std::vector<unsigned int> S2VParam::Order() const EddyTry
{
  std::vector<unsigned int> rval(_order.size());
  for (unsigned int i=0; i<_order.size(); i++) rval[i] = static_cast<unsigned int>(_order[i]);
  return(rval);
} EddyCatch

double S2VParam::Lambda(unsigned int oi) const EddyTry
{
  if (oi >= _lambda.size()) throw EddyException("S2VParam::Lambda: oi out of range");
  return(static_cast<double>(_lambda[oi]));
} EddyCatch

std::vector<double> S2VParam::Lambda() const EddyTry
{
  std::vector<double> rval(_lambda.size());
  for (unsigned int i=0; i<_lambda.size(); i++) rval[i] = static_cast<double>(_lambda[i]);
  return(rval);
} EddyCatch

float S2VParam::FWHM(unsigned int oi, unsigned int iter) const EddyTry
{
  if (oi >= this->NOrder()) throw EddyException("S2VParam::FWHM: oi out of range");
  if (iter >= this->NIter(oi)) throw EddyException("S2VParam::FWHM: iter out of range");
  if (_fwhm.size()==1) return(_fwhm[0]);
  else {
    unsigned int indx=0;
    for (unsigned int i=0; i<oi; i++) indx += this->NIter(i);
    return(_fwhm[indx+iter]);
  }
} EddyCatch

std::vector<float> S2VParam::FWHM(unsigned int oi) const EddyTry
{
  if (oi >= NOrder()) throw EddyException("S2VParam::FWHM: oi out of range");
  if (_fwhm.size()==1) {
    std::vector<float> rval(this->NIter(oi),_fwhm[0]);
    return(rval);
  }
  else {
    unsigned int indx=0;
    for (unsigned int i=0; i<oi; i++) indx += this->NIter(i);
    std::vector<float> rval(this->NIter(oi),0.0);
    for (unsigned int i=0; i<this->NIter(oi); i++) rval[i] = _fwhm[indx+i];
    return(rval);
  }
} EddyCatch

EDDY::MultiBandGroups EddyCommandLineOptions::MultiBand() const EddyTry
{ 
  NEWIMAGE::volume<float> mhdr; 
  NEWIMAGE::read_volume_hdr_only(mhdr,_mask.value());
  if (_slspec.set()) {
    EDDY::MultiBandGroups mbg(_slspec.value());
    return(mbg);
  }
  else if (_json.set()) {
    EDDY::JsonReader jr(_json.value());
    EDDY:: MultiBandGroups mbg(jr.SliceOrdering());
    return(mbg);
  }
  else {
    EDDY::MultiBandGroups mbg(mhdr.zsize(),static_cast<unsigned int>(_mb.value()),static_cast<unsigned int>(_mb_offs.value()));
    if (_slorder.set() && _slorder.value() != string("")) mbg.SetTemporalOrder(get_slorder(_slorder.value(),mbg.NGroups()));
    return(mbg);
  }
} EddyCatch

EDDY::ECModel EddyCommandLineOptions::FirstLevelModel() const EddyTry
{
  if (_flm.value() == string("movement")) return(EDDY::NoEC);
  else if (_flm.value() == string("linear")) return(EDDY::Linear);
  else if (_flm.value() == string("quadratic")) return(EDDY::Quadratic);
  else if (_flm.value() == string("cubic")) return(EDDY::Cubic);
  return(EDDY::Unknown);
} EddyCatch

EDDY::ECModel EddyCommandLineOptions::b0_FirstLevelModel() const EddyTry
{
  if (_b0_flm.value() == string("movement")) return(EDDY::NoEC);
  else if (_b0_flm.value() == string("linear")) return(EDDY::Linear);
  else if (_b0_flm.value() == string("quadratic")) return(EDDY::Quadratic);
  return(EDDY::Unknown);
} EddyCatch

EDDY::OffsetModel EddyCommandLineOptions::OffsetModel() const EddyTry
{
  if (_offset_model.value() == string("linear")) return(EDDY::LinearOffset);
  else if (_offset_model.value() == string("quadratic")) return(EDDY::QuadraticOffset);
  return(EDDY::UnknownOffset);
} EddyCatch

NEWMAT::ColumnVector EddyCommandLineOptions::HyperParValues() const EddyTry
{
  NEWMAT::ColumnVector rval(_hypar_internal.size());
  for (unsigned int i=0; i<_hypar_internal.size(); i++) rval(i+1) = _hypar_internal[i];
  return(rval);
} EddyCatch

void EddyCommandLineOptions::SetHyperParValues(const NEWMAT::ColumnVector& p) EddyTry
{
  _hypar_internal.resize(p.Nrows());
  for (unsigned int i=0; i<_hypar_internal.size(); i++) _hypar_internal[i] = p(i+1);
  this->SetHyperParFixed(true);
  return;
} EddyCatch

std::vector<float> EddyCommandLineOptions::TestRotAngles() const EddyTry
{
  std::vector<float> rot = _test_rot.value();
  if (rot.size() < 3) rot.resize(3,0.0);
  return(rot);
} EddyCatch

NEWIMAGE::interpolation EddyCommandLineOptions::InterpolationMethod() const EddyTry
{
  if (_interp.value() == string("spline")) return(NEWIMAGE::spline);
  else if (_interp.value() == string("trilinear")) return(NEWIMAGE::trilinear);  
  else throw EddyException("EddyCommandLineOptions::InterpolationMethod: Invalid interpolation option.");
} EddyCatch

NEWIMAGE::extrapolation EddyCommandLineOptions::ExtrapolationMethod() const EddyTry
{
  if (_extrap.value() == string("periodic")) return(NEWIMAGE::periodic);
  else if (_extrap.value() == string("mirror")) return(NEWIMAGE::mirror);  
  else throw EddyException("EddyCommandLineOptions::ExtrapolationMethod: Invalid extrapolation option.");
} EddyCatch

NEWIMAGE::interpolation EddyCommandLineOptions::S2VInterpolationMethod() const EddyTry
{
  if (_s2v_interp.value() == string("spline")) return(NEWIMAGE::spline);
  else if (_s2v_interp.value() == string("trilinear")) return(NEWIMAGE::trilinear);  
  else throw EddyException("EddyCommandLineOptions::S2VInterpolationMethod: Invalid interpolation option.");
} EddyCatch

DebugIndexClass::DebugIndexClass(const std::string& in) EddyTry 
{ 
  _indx = parse_commaseparated_numbers(in); 
} EddyCatch

std::vector<unsigned int> DebugIndexClass::parse_commaseparated_numbers(const std::string& list) const EddyTry
{
  std::vector<std::string> str_list = parse_commaseparated_list(list);
  std::vector<unsigned int> number_list(str_list.size(),0);
  for (unsigned int i=0; i<str_list.size(); i++) {
    number_list[i] = atoi(str_list[i].c_str());
  }
  
  return(number_list);
} EddyCatch

std::vector<std::string> DebugIndexClass::parse_commaseparated_list(const std::string&  list) const EddyTry
{
  std::vector<std::string> ostr;

  size_t cur_pos = 0;
  size_t new_pos = 0;
  unsigned int n=0;
  while ((new_pos = list.find_first_of(',',cur_pos)) != string::npos) {
    ostr.resize(++n);
    ostr[n-1] = list.substr(cur_pos,new_pos-cur_pos);
    cur_pos = new_pos+1;        
  }
  ostr.resize(++n);
  ostr[n-1] = list.substr(cur_pos,string::npos);

  return(ostr);
} EddyCatch 

S2VParam::S2VParam(const std::vector<int>& order, const std::vector<float>& lambda, const std::vector<float>& fwhm, 
		   const std::vector<int>& niter) EddyTry : _order(order), _lambda(lambda), _fwhm(fwhm), _niter(niter)
{
  if (_order.size() != _lambda.size()) throw EddyException("Size of --s2v_lambda must match size of --mporder");
  if (_order.size() != _niter.size()) throw EddyException("Size of --s2v_niter must match size of --mporder");
  if (_fwhm.size() != 1 && _fwhm.size() != this->total_niter()) throw EddyException("--s2v_fwhm value must be given once or once per iteration");
  if (this->total_niter() > 100) throw EddyException("You have asked for more than 100 slice-to-volume iterations. Seriously?");
  
  for (unsigned int i=0; i<_fwhm.size(); i++) {
    if (_fwhm[i] < 0.0 || _fwhm[i] > 20.0) throw EddyException("--s2v_fwhm value outside valid range 0-20mm");
  }
} EddyCatch

void EddyCommandLineOptions::do_initial_parsing(int argc, char *argv[]) EddyTry
{
  Utilities::OptionParser options(_title,_examples);

  try {
    options.add(_imain);
    options.add(_mask);
    options.add(_index);
    options.add(_session);
    options.add(_mb);
    options.add(_mb_offs);
    options.add(_slorder);
    options.add(_slspec);
    options.add(_json);
    options.add(_mp_order);
    options.add(_s2v_lambda);
    options.add(_acqp);
    options.add(_topup);
    options.add(_field);
    options.add(_field_mat);
    options.add(_bvecs);
    options.add(_bvals);
    options.add(_flm);
    options.add(_slm_str);
    options.add(_b0_flm);
    options.add(_b0_slm_str);
    options.add(_fwhm_tmp);
    options.add(_s2v_fwhm);
    options.add(_niter_tmp);
    options.add(_s2v_niter);
    options.add(_out);
    options.add(_very_verbose);
    options.add(_dwi_only);
    options.add(_b0_only);
    options.add(_fields);
    options.add(_rms);
    options.add(_dfields);
    options.add(_cnr_maps);
    options.add(_range_cnr_maps);
    options.add(_residuals);
    options.add(_with_outliers);
    options.add(_history);
    options.add(_fep);
    options.add(_dont_mask_output);
    options.add(_interp);
    options.add(_s2v_interp);
    options.add(_extrap);
    options.add(_epvalid);
    options.add(_resamp);
    options.add(_covfunc);
    options.add(_hyparcostfunc);
    options.add(_nvoxhp);
    options.add(_initrand);
    options.add(_hyparfudgefactor);
    options.add(_hypar);
    options.add(_write_slice_stats);
    options.add(_rep_ol);
    options.add(_rep_noise);
    options.add(_ol_nstd);
    options.add(_ol_nvox);
    options.add(_ol_ec);
    options.add(_ol_type);
    options.add(_ol_pos);
    options.add(_ol_sqr);
    options.add(_estimate_mbs);
    options.add(_mbs_niter);
    options.add(_mbs_lambda);
    options.add(_mbs_ksp);
    options.add(_sep_offs_move);
    options.add(_dont_sep_offs_move);
    options.add(_offset_model);
    options.add(_peas);
    options.add(_dont_peas);
    options.add(_use_b0s_for_peas);
    options.add(_data_is_shelled);
    options.add(_init);
    options.add(_debug_tmp);
    options.add(_dbg_indx_str);
    options.add(_lsr_lambda);
    options.add(_ref_scan_no);
    options.add(_rbvde);
    options.add(_test_rot);
    options.add(_print_mi_values);
    options.add(_print_mi_planes);
    options.add(_verbose);
    options.add(_help);

    int i=options.parse_command_line(argc, argv);
    if (i < argc) {
      for (; i<argc; i++) {
        cerr << "Unknown input: " << argv[i] << endl;
      }
      exit(EXIT_FAILURE);
    }

    if (_help.value() || !options.check_compulsory_arguments(true)) {
      options.usage();
      exit(EXIT_FAILURE);
    }
  }  
  catch(Utilities::X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } 

  
  
  
  ofstream cf;  
  cf.open(_out.value()+std::string(".eddy_command_txt"));
  cf << argv[0];
  for (int ii=1; ii<argc; ii++) cf << " " << argv[ii];
  cf << endl;
  cf.close();
  ofstream dpf; 
  Utilities::detailed_output(dpf);
  dpf.open(_out.value()+std::string(".eddy_values_of_all_input_parameters"));
  dpf << "This file contains values (including defaults) for all input parameters to eddy." << endl;
  dpf << options << endl;
  dpf.close();
} EddyCatch




bool EddyCommandLineOptions::indicies_kosher(NEWMAT::Matrix indx, NEWMAT::Matrix acqp) EddyTry
{
  if (indx.Ncols() > indx.Nrows()) indx = indx.t();
  _indvec.resize(indx.Nrows());
  unsigned int max_indx = static_cast<unsigned int>(indx(1,1));
  for (int i=0; i<indx.Nrows(); i++) {
    _indvec[i] = static_cast<unsigned int>(indx(i+1,1));
    if (fabs(static_cast<double>(_indvec[i])-indx(i+1,1)) > 1e-6) return(false);
    max_indx = std::max(max_indx,_indvec[i]);
  }
  if (max_indx > static_cast<unsigned int>(acqp.Nrows())) return(false);
  else return(true);
} EddyCatch

 
std::vector<unsigned int> EddyCommandLineOptions::get_slorder(const std::string& fname,
							      unsigned int       ngrp) const EddyTry
{
  std::vector<unsigned int> rval;
  try {
    NEWMAT::Matrix tmp = MISCMATHS::read_ascii_matrix(fname);
    if (tmp.Ncols() > tmp.Nrows()) tmp = tmp.t();
    int n = tmp.Nrows(); 
    int one = tmp.Ncols(); 
    rval.resize(n);
    if (n != int(ngrp) || one != 1) throw EddyException("Size mismatch between --imain and --slorder file");
    for (int i=0; i<n; i++) {
      if (tmp(i+1,1)<0 || tmp(i+1,1)>ngrp-1) throw EddyException("--slorder file contains invalid entry");
      rval[i] = tmp(i+1,1);
    }
  }
  catch(...) { throw EddyException("Error when attempting to read --slorder file"); }

  return(rval);
} EddyCatch

