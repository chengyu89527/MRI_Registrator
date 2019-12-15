// Definitions of classes and functions that
// estimates the derivative fields for a 
// movement-by-susceptibility model for the
// eddy project.
//
// This file contins all the code for both the
// CPU and the GPU implementations. The code
// generation is goverened #include:s and
// the COMPILE_GPU macro.
//
// MoveBySuscCF.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford 
//
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "newmat.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "EddyHelperClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "ECScanClasses.h"
#include "MoveBySuscCF.h"
#include "eddy.h"
#ifdef COMPILE_GPU
#include "cuda/EddyGpuUtils.h"
#include "CBFSplineField.cuh"
#endif

namespace EDDY {

class MoveBySuscCFImpl
{
public:
  MoveBySuscCFImpl(EDDY::ECScanManager&                 sm,
		   const EDDY::EddyCommandLineOptions&  clo,
		   const std::vector<unsigned int>&     b0s,
		   const std::vector<unsigned int>&     dwis,
		   const std::vector<unsigned int>&     mps,
		   unsigned int                         order,
		   bool                                 ujm,
		   double                               ksp);
  ~MoveBySuscCFImpl() {}
  double cf(const NEWMAT::ColumnVector&    p);
  NEWMAT::ReturnMatrix grad(const NEWMAT::ColumnVector&    p);
  boost::shared_ptr<BFMatrix> hess(const NEWMAT::ColumnVector& p,
				   boost::shared_ptr<BFMatrix> iptr=boost::shared_ptr<BFMatrix>());
  void SetLambda(double lambda) { _lmbd = lambda; }
  unsigned int NPar() const EddyTry { return(this->total_no_of_parameters()); } EddyCatch
  NEWMAT::ReturnMatrix Par() const EddyTry { return(this->get_field_parameters()); } EddyCatch
  void WriteFirstOrderFields(const std::string& fname) const;
  void WriteSecondOrderFields(const std::string& fname) const;
  
  void ResetCache() { _utd = false; _m_utd = false; }
private:
  EDDY::ECScanManager&                                                   _sm;    
  EDDY::EddyCommandLineOptions                                           _clo;   
  std::vector<unsigned int>                                              _b0s;   
  std::vector<unsigned int>                                              _dwis;  
  std::vector<unsigned int>                                              _mps;   
  unsigned int                                                           _order; 
  bool                                                                   _ujm;   
  double                                                                 _lmbd;  
  std::vector<unsigned int>                                              _ksp;   
#ifdef COMPILE_GPU
  std::vector<std::shared_ptr<CBF::CBFSplineField> >                 _sp1;   
  std::vector<std::vector<std::shared_ptr<CBF::CBFSplineField> > >    _sp2;   
#else
  std::vector<std::shared_ptr<BASISFIELD::splinefield> >                 _sp1;   
  std::vector<std::vector<std::shared_ptr<BASISFIELD::splinefield> > >   _sp2;   
#endif
  NEWMAT::ColumnVector                                                   _cp;    
  NEWMAT::ColumnVector                                                   _hypar; 
  NEWIMAGE::volume<float>                                                _mask;  
  NEWIMAGE::volume<char>                                                 _cmask; 
  unsigned int                                                           _nvox;  
  unsigned int                                                           _omp_num_threads; 
  
  MISCMATHS::BFMatrixPrecisionType  _hp = MISCMATHS::BFMatrixDoublePrecision;     
  bool                                                            _bno = true;    
  bool                                                            _utd = false;   
  bool                                                            _m_utd = false; 

  bool                                                            _chgr = false;  
  bool                                                            _chH = false;   
  unsigned int                                                    _grad_cnt = 0;  
  unsigned int                                                    _hess_cnt = 0;  

  
  
  
  

  
  typedef std::vector<NEWIMAGE::volume<float> > ImageVec;    
  typedef std::vector<unsigned int>             BpeVec;      
  
  NEWIMAGE::volume<float>                                    _sum_sqr_diff;      
  
  ImageVec                                                   _sum_fo_deriv_diff; 
  std::map<BpeVec,ImageVec>                                  _sum_fo_ima_diff;   
  std::vector<ImageVec>                                      _sum_so_deriv_diff; 
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_so_ima_diff;   
  
  std::vector<ImageVec>                                      _sum_fo_deriv_deriv;
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_fo_ima_ima;    
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_fo_deriv_ima;  
  std::vector<ImageVec>                                      _sum_so_deriv_deriv;
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_so_ima_ima;    
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_so_deriv_ima;  
  std::vector<ImageVec>                                      _sum_cross_deriv_deriv; 
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_cross_ima_ima;     
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_cross_deriv_ima;   


  

  unsigned int no_of_parameters_per_field() const EddyTry { return(_sp1[0]->CoefSz_x() * _sp1[0]->CoefSz_y() * _sp1[0]->CoefSz_z()); } EddyCatch
  unsigned int no_of_first_order_fields() const EddyTry { return(_mps.size()); } EddyCatch
  unsigned int no_of_second_order_fields() const EddyTry { return((_order == 1) ? 0 : _mps.size()*(_mps.size()+1)/2); } EddyCatch
  unsigned int no_of_fields() const EddyTry { return(no_of_first_order_fields() + no_of_second_order_fields()); } EddyCatch
  unsigned int total_no_of_parameters() const EddyTry { return(no_of_parameters_per_field()*no_of_fields()); } EddyCatch
  unsigned int no_of_b0s() const EddyTry { return(_b0s.size()); } EddyCatch
  unsigned int no_of_dwis() const EddyTry { return(_dwis.size()); } EddyCatch
  unsigned int nvox() EddyTry { if (!_m_utd) recalculate_sum_images(); return(_nvox); } EddyCatch
  double lambda() const { return(_lmbd); }
  double pi() const { return(3.141592653589793); }
  MISCMATHS::BFMatrixPrecisionType hessian_precision() const { return(_hp); }
  NEWMAT::ColumnVector get_field_parameters() const EddyTry { return(_cp); } EddyCatch
  void set_field_parameters(EDDY::ECScanManager&        sm,
			    const NEWMAT::ColumnVector& p);
  const NEWIMAGE::volume<float>& mask() EddyTry { if (!_m_utd) recalculate_sum_images(); return(_mask); } EddyCatch
  const NEWIMAGE::volume<char>& cmask() EddyTry { if (!_m_utd) recalculate_sum_images(); return(_cmask); } EddyCatch
  void resize_containers_for_sum_images_for_grad(const std::vector<std::vector<unsigned int> >& indicies,
						 const std::vector<EDDY::ScanType>&             imtypes);
  void resize_containers_for_sum_images_for_hess(const std::vector<std::vector<unsigned int> >& indicies,
						 const std::vector<EDDY::ScanType>&             imtypes);
  void set_sum_images_to_zero(const NEWIMAGE::volume<float>& ima);
  void recalculate_sum_images();
  void recalculate_sum_so_imas_for_hess(const NEWMAT::ColumnVector&           p,
					const BpeVec&                         bpe,
					const NEWIMAGE::volume<float>&        ima,
					const NEWIMAGE::volume<float>&        deriv);
  void recalculate_sum_cross_imas_for_hess(const NEWMAT::ColumnVector&           p,
					   const BpeVec&                         bpe,
					   const NEWIMAGE::volume<float>&        ima,
					   const NEWIMAGE::volume<float>&        deriv);
  void calculate_first_order_subm(
				  const std::vector<ImageVec>&                                        sop_dfdf,
				  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
				  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
				  const NEWIMAGE::volume<char>&                                       lmask,
				  unsigned int                                                        nvox,
				  
				  std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const;
  void calculate_second_order_subm(
				   const std::vector<ImageVec>&                                        sop_dfdf,
				   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
				   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
				   const NEWIMAGE::volume<char>&                                       lmask,
				   unsigned int                                                        nvox,
				   
				   std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const;
  void calculate_cross_subm(
			    const std::vector<ImageVec>&                                        sop_dfdf,
			    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
			    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
			    const NEWIMAGE::volume<char>&                                       lmask,
			    unsigned int                                                        nvox,
			    
			    std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const;

  NEWMAT::ColumnVector stl2newmat(const std::vector<double>& stl) const;
  NEWMAT::Matrix linspace(const NEWMAT::Matrix& inmat) const;
  boost::shared_ptr<MISCMATHS::BFMatrix> concatenate_subdiag_subm(const std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
								  unsigned int                                                              n) const;
  boost::shared_ptr<MISCMATHS::BFMatrix> concatenate_rect_subm(const std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
							       unsigned int                                                              m,
							       unsigned int                                                              n) const;
  bool is_first_index_pair(unsigned int                                                 i,
			   unsigned int                                                 j,
			   const std::vector<unsigned int>&                             ivec,
			   const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const;
  std::pair<unsigned int,unsigned int> find_first_index_pair(const std::vector<unsigned int>&                             ivec,
							     const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const;
  std::vector<std::vector<std::vector<unsigned int> > > build_second_order_pmap() const;
  std::vector<std::vector<std::vector<unsigned int> > > build_cross_pmap() const;
  std::vector<unsigned int> make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
					      const std::pair<unsigned int,unsigned int>& p2,
					      unsigned int                                np) const;
  std::vector<unsigned int> make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
					      unsigned int                                i,
					      unsigned int                                np) const;
  std::pair<unsigned int,unsigned int> get_second_order_index_pair(unsigned int i,
								   unsigned int np) const;
};

MoveBySuscCF::MoveBySuscCF(EDDY::ECScanManager&                 sm,
			   const EDDY::EddyCommandLineOptions&  clo,
			   const std::vector<unsigned int>&     b0s,
			   const std::vector<unsigned int>&     dwis,
			   const std::vector<unsigned int>&     mps,
			   unsigned int                         order,
			   double                               ksp) EddyTry { _pimpl = new MoveBySuscCFImpl(sm,clo,b0s,dwis,mps,order,true,ksp); } EddyCatch

MoveBySuscCF::~MoveBySuscCF() EddyTry { delete _pimpl; } EddyCatch

double MoveBySuscCF::cf(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->cf(p)); } EddyCatch

NEWMAT::ReturnMatrix MoveBySuscCF::grad(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->grad(p)); } EddyCatch

boost::shared_ptr<BFMatrix> MoveBySuscCF::hess(const NEWMAT::ColumnVector& p,
					       boost::shared_ptr<BFMatrix> iptr) const EddyTry { return(_pimpl->hess(p,iptr)); } EddyCatch

void MoveBySuscCF::SetLambda(double lambda) EddyTry { _pimpl->SetLambda(lambda); } EddyCatch

NEWMAT::ReturnMatrix MoveBySuscCF::Par() const EddyTry { return(_pimpl->Par()); } EddyCatch

unsigned int MoveBySuscCF::NPar() const EddyTry { return(_pimpl->NPar()); } EddyCatch

void MoveBySuscCF::WriteFirstOrderFields(const std::string& fname) const EddyTry { _pimpl->WriteFirstOrderFields(fname); } EddyCatch

void MoveBySuscCF::WriteSecondOrderFields(const std::string& fname) const EddyTry { _pimpl->WriteSecondOrderFields(fname); } EddyCatch

void MoveBySuscCF::ResetCache() EddyTry { _pimpl->ResetCache(); } EddyCatch

MoveBySuscCFImpl::MoveBySuscCFImpl(EDDY::ECScanManager&                 sm,
				   const EDDY::EddyCommandLineOptions&  clo,
				   const std::vector<unsigned int>&     b0s,
				   const std::vector<unsigned int>&     dwis,
				   const std::vector<unsigned int>&     mps,
				   unsigned int                         order,
				   bool                                 ujm,
				   double                               ksp) EddyTry : _sm(sm), _clo(clo), _b0s(b0s), _dwis(dwis), _mps(mps), _order(order), _ujm(ujm), _lmbd(50)
{
  if (order != 1 && order != 2) throw EddyException("MoveBySuscCFImpl::MoveBySuscCFImpl: order must be 1 or 2");
  std::vector<unsigned int> isz(3,0); 
  isz[0] = static_cast<unsigned int>(_sm.Scan(0,ANY).GetIma().xsize());
  isz[1] = static_cast<unsigned int>(_sm.Scan(0,ANY).GetIma().ysize()); 
  isz[2] = static_cast<unsigned int>(_sm.Scan(0,ANY).GetIma().zsize());
  std::vector<double> vxs(3,0); 
  vxs[0] = static_cast<double>(_sm.Scan(0,ANY).GetIma().xdim());
  vxs[1] = static_cast<double>(_sm.Scan(0,ANY).GetIma().ydim());
  vxs[2] = static_cast<double>(_sm.Scan(0,ANY).GetIma().zdim());
  _ksp.resize(3);
  _ksp[0] = static_cast<unsigned int>((ksp / vxs[0]) + 0.5);
  _ksp[1] = static_cast<unsigned int>((ksp / vxs[1]) + 0.5);
  _ksp[2] = static_cast<unsigned int>((ksp / vxs[2]) + 0.5);

  _sp1.resize(_mps.size(),nullptr);
  for (unsigned int i=0; i<_mps.size(); i++) {
#ifdef COMPILE_GPU
    _sp1[i] = std::shared_ptr<CBF::CBFSplineField>(new CBF::CBFSplineField(isz,vxs,_ksp));
#else
    _sp1[i] = std::shared_ptr<BASISFIELD::splinefield>(new BASISFIELD::splinefield(isz,vxs,_ksp));
#endif
  }
  if (order == 2) {
    _sp2.resize(_mps.size());
    for (unsigned int i=0; i<_mps.size(); i++) {
      _sp2[i].resize(i+1,nullptr);
      for (unsigned int j=0; j<(i+1); j++) {
#ifdef COMPILE_GPU
        _sp2[i][j] = std::shared_ptr<CBF::CBFSplineField>(new CBF::CBFSplineField(isz,vxs,_ksp));
#else
        _sp2[i][j] = std::shared_ptr<BASISFIELD::splinefield>(new BASISFIELD::splinefield(isz,vxs,_ksp));
#endif
      }
    }
  }
  _cp.ReSize(this->total_no_of_parameters()); _cp = 0.0;
  char *ont = getenv("OMP_NUM_THREADS");
  if (ont == nullptr) _omp_num_threads = 1;
  else if (sscanf(ont,"%u",&_omp_num_threads) != 1) throw EddyException("MoveBySuscCFImpl::MoveBySuscCFImpl: problem reading environment variable OMP_NUM_THREADS");
} EddyCatch

void MoveBySuscCFImpl::WriteFirstOrderFields(const std::string& fname) const EddyTry
{
  const NEWIMAGE::volume<float>& tmp=_sm.Scan(0,ANY).GetIma();
  NEWIMAGE::volume4D<float> ovol(tmp.xsize(),tmp.ysize(),tmp.zsize(),no_of_first_order_fields());
  NEWIMAGE::copybasicproperties(tmp,ovol);
  NEWIMAGE::volume<float> dfield = tmp;
  for (unsigned int i=0; i<_mps.size(); i++) {
    _sp1[i]->AsVolume(dfield);
    if (_mps[i] > 2 && _mps[i] < 6) { 
      dfield *= static_cast<float>(this->pi() / 180.0);
    }
    ovol[i] = dfield;
  }
  NEWIMAGE::write_volume(ovol,fname);
} EddyCatch

void MoveBySuscCFImpl::WriteSecondOrderFields(const std::string& fname) const EddyTry
{
  const NEWIMAGE::volume<float>& tmp=_sm.Scan(0,ANY).GetIma();
  NEWIMAGE::volume4D<float> ovol(tmp.xsize(),tmp.ysize(),tmp.zsize(),no_of_second_order_fields());
  NEWIMAGE::copybasicproperties(tmp,ovol);
  NEWIMAGE::volume<float> dfield = tmp;
  unsigned int cnt=0;
  for (unsigned int i=0; i<_mps.size(); i++) {
    for (unsigned int j=0; j<(i+1); j++) {
      _sp2[i][j]->AsVolume(dfield);
      ovol[cnt++] = dfield;
    }
  }
  NEWIMAGE::write_volume(ovol,fname);
} EddyCatch

void MoveBySuscCFImpl::set_field_parameters(EDDY::ECScanManager&        sm,
					    const NEWMAT::ColumnVector& p) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::set_field_parameters: mismatch between p and total no of parameters");
  
  if (static_cast<unsigned int>(_cp.Nrows()) == total_no_of_parameters() && _cp == p) return;
  else {
    _m_utd = false;
    _utd = false;
    _cp = p;
    
    unsigned int pindx=1;
    for (unsigned int i=0; i<_mps.size(); i++) {
      _sp1[i]->SetCoef(p.Rows(pindx,pindx+no_of_parameters_per_field()-1));
      NEWIMAGE::volume<float> dfield = sm.Scan(0,ANY).GetIma(); dfield=0.0;
      _sp1[i]->AsVolume(dfield);
      sm.SetDerivSuscField(_mps[i],dfield);
      pindx += no_of_parameters_per_field();
    }
    
    if (_order == 2) {
      for (unsigned int i=0; i<_mps.size(); i++) {
	for (unsigned int j=0; j<(i+1); j++) {
	  _sp2[i][j]->SetCoef(p.Rows(pindx,pindx+no_of_parameters_per_field()-1));
	  NEWIMAGE::volume<float> dfield = sm.Scan(0,ANY).GetIma(); dfield=0.0;
	  _sp2[i][j]->AsVolume(dfield);
	  sm.Set2ndDerivSuscField(_mps[i],_mps[j],dfield);
	  pindx += no_of_parameters_per_field();
	}
      }
    }
  }
} EddyCatch

double MoveBySuscCFImpl::cf(const NEWMAT::ColumnVector&  p) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::cf: mismatch between p and total no of parameters");
  
  
  set_field_parameters(_sm,p);
  recalculate_sum_images();
  
  double ssd = _sum_sqr_diff.sum(mask()) / static_cast<double>(nvox());
  
  double reg = 0;
  for (unsigned int i=0; i<_mps.size(); i++) {
    reg += lambda() * _sp1[i]->BendEnergy() / static_cast<double>(nvox());
  }
  if (_order == 2) {
    for (unsigned int i=0; i<_mps.size(); i++) {
      for (unsigned int j=0; j<(i+1); j++) {
	reg += lambda() * _sp2[i][j]->BendEnergy() / static_cast<double>(nvox());
      }
    }
  }
  reg /= no_of_fields(); 
  ssd += reg;

  return(ssd);
} EddyCatch

NEWMAT::ReturnMatrix MoveBySuscCFImpl::grad(const NEWMAT::ColumnVector& p) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::grad: mismatch between p and total no of parameters");
  
  set_field_parameters(_sm,p);
  recalculate_sum_images();
  
  
  
  NEWMAT::ColumnVector gradient(total_no_of_parameters()); gradient=0.0;
  unsigned int fr = 1;
  unsigned int lr = no_of_parameters_per_field();
  NEWIMAGE::volume<float> ones = _sm.Scan(0,ANY).GetIma(); ones = 1.0;
  for (unsigned int pi=0; pi<_mps.size(); pi++) {
    
    gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp1[pi]->Jte(_sum_fo_deriv_diff[pi],ones,&cmask());
    
    if (_ujm) {
      for (const auto& elem : _sum_fo_ima_diff) { 
	std::vector<unsigned int> dvec = elem.first;
	gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp1[pi]->Jte(dvec,elem.second[pi],ones,&cmask());      
      }
    }
    fr += no_of_parameters_per_field();
    lr += no_of_parameters_per_field();
  }
  if (_order == 2) {
    for (unsigned int pi=0; pi<_mps.size(); pi++) {
      for (unsigned int pj=0; pj<(pi+1); pj++) {
	
	gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp2[pi][pj]->Jte(_sum_so_deriv_diff[pi][pj],ones,&cmask());
	
	if (_ujm) {
	  for (const auto& elem: _sum_so_ima_diff) {
	    std::vector<unsigned int> dvec = elem.first;
	    gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp2[pi][pj]->Jte(dvec,elem.second[pi][pj],ones,&cmask());
	  }
	}
      }
    }
  }
  
  
  fr = 1;
  lr = no_of_parameters_per_field();
  for (unsigned int i=0; i<_mps.size(); i++) {
    gradient.Rows(fr,lr) += (lambda()/static_cast<double>(nvox()*no_of_fields())) * _sp1[i]->BendEnergyGrad();
    fr += no_of_parameters_per_field();
    lr += no_of_parameters_per_field();
  }
  if (_order == 2) {
    for (unsigned int i=0; i<_mps.size(); i++) {
      for (unsigned int j=0; j<(i+1); j++) {
	gradient.Rows(fr,lr) += (lambda()/static_cast<double>(nvox()*no_of_fields())) * _sp2[i][j]->BendEnergyGrad();
	fr += no_of_parameters_per_field();
	lr += no_of_parameters_per_field();
      }
    }
  }

  if (_chgr) { 
    char fname[256]; sprintf(fname,"gradient_%02d.txt",_grad_cnt);
    MISCMATHS::write_ascii_matrix(gradient,fname);
    
    NEWMAT::ColumnVector tmp_p = p;
    tmp_p(10000) += 0.0001;
    double cf0 = this->cf(tmp_p);
    tmp_p(10000) -= 0.0001;
    cf0 = this->cf(tmp_p);
    NEWMAT::ColumnVector numgrad(total_no_of_parameters()); numgrad=0.0;
    unsigned int no_of_values = 20;
    unsigned int step = total_no_of_parameters() / (no_of_values + 1);
    double delta = 0.001;
    for (unsigned int i=step/2; i<total_no_of_parameters(); i+=step) {
      if (i<total_no_of_parameters()) {
	tmp_p(i) += delta;
	double cf1 = this->cf(tmp_p);
	numgrad(i) = (cf1-cf0) / delta;
	tmp_p(i) -= delta;
      }
    }
    sprintf(fname,"numgrad_delta_001_%02d.txt",_grad_cnt);
    MISCMATHS::write_ascii_matrix(numgrad,fname);    
  }
  _grad_cnt++;

  gradient.Release(); return(gradient);
} EddyCatch


boost::shared_ptr<MISCMATHS::BFMatrix> MoveBySuscCFImpl::hess(const NEWMAT::ColumnVector& p,
							      boost::shared_ptr<BFMatrix> iptr) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::hess: mismatch between p and total no of parameters");
  
  set_field_parameters(_sm,p);
  recalculate_sum_images();
  
  if (iptr) iptr->Clear();
  
  std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > > first_order_subm(this->no_of_first_order_fields());
  std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > > second_order_subm(this->no_of_second_order_fields());
  std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > > cross_subm(this->no_of_second_order_fields());
  for (unsigned int i=0; i<no_of_first_order_fields(); i++) {
    first_order_subm[i].resize(i+1); 
  }
  if (_order == 2) { 
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      second_order_subm[i].resize(i+1);
      cross_subm[i].resize(this->no_of_first_order_fields());
    }
  }

  
  calculate_first_order_subm(_sum_fo_deriv_deriv,_sum_fo_ima_ima,_sum_fo_deriv_ima,cmask(),nvox(),first_order_subm);
  if (_order == 2) {
    calculate_second_order_subm(_sum_so_deriv_deriv,_sum_so_ima_ima,_sum_so_deriv_ima,cmask(),nvox(),second_order_subm);
    calculate_cross_subm(_sum_cross_deriv_deriv,_sum_cross_ima_ima,_sum_cross_deriv_ima,cmask(),nvox(),cross_subm);
  }

  
  boost::shared_ptr<MISCMATHS::BFMatrix> reg = _sp1[0]->BendEnergyHess(hessian_precision());
  for (unsigned int i=0; i<this->no_of_first_order_fields(); i++) {
    first_order_subm[i][i]->AddToMe(*reg,lambda()/static_cast<double>(nvox()*no_of_fields()));
  }
  for (unsigned int i=0; i<this->no_of_second_order_fields(); i++) {
    second_order_subm[i][i]->AddToMe(*reg,lambda()/static_cast<double>(nvox()*no_of_fields()));
  }

  
  boost::shared_ptr<MISCMATHS::BFMatrix> first_order_subm_cat = concatenate_subdiag_subm(first_order_subm,no_of_first_order_fields());
  first_order_subm.clear(); 
  boost::shared_ptr<MISCMATHS::BFMatrix> rval = first_order_subm_cat;
  
  
  if (_order == 2) {
    boost::shared_ptr<MISCMATHS::BFMatrix> second_order_subm_cat = concatenate_subdiag_subm(second_order_subm,no_of_second_order_fields());
    second_order_subm.clear(); 
    boost::shared_ptr<MISCMATHS::BFMatrix> cross_subm_cat = concatenate_rect_subm(cross_subm,no_of_second_order_fields(),no_of_first_order_fields());
    cross_subm.clear(); 
    rval->HorConcat2MyRight(*(cross_subm_cat->Transpose()));
    cross_subm_cat->HorConcat2MyRight(*second_order_subm_cat);
    rval->VertConcatBelowMe(*cross_subm_cat);
  }

  if (_chH) { 
    bool old_chgr = false;
    if (_chgr) { _chgr = false; old_chgr = true; }
    char fname[256]; sprintf(fname,"hessian_%02d.txt",_hess_cnt);
    rval->Print(fname);

    NEWMAT::ColumnVector tmp_p = p;
    NEWMAT::ColumnVector grad0 = this->grad(tmp_p);
    unsigned int no_of_values = 20;
    NEWMAT::Matrix numhess(no_of_values*total_no_of_parameters(),3); numhess=0.0;
    unsigned int step = total_no_of_parameters() / (no_of_values + 1);
    double delta = 0.01;
    unsigned int ii = 0;
    for (unsigned int i=step/2; i<total_no_of_parameters(); i+=step) {
      if (ii<no_of_values) {
	tmp_p(i) += delta;
	NEWMAT::ColumnVector grad1 = this->grad(tmp_p);
	unsigned int fr = (ii)*total_no_of_parameters()+1;
	unsigned int lr = (ii+1)*total_no_of_parameters();
	numhess.SubMatrix(fr,lr,3,3) = (grad1 - grad0) / delta;
	numhess.SubMatrix(fr,lr,2,2) = i;
	numhess.SubMatrix(fr,lr,1,1) = this->linspace(numhess.SubMatrix(fr,lr,1,1));
	tmp_p(i) -= delta;
	ii++;
      }
    }
    sprintf(fname,"numhess_delta_01_%02d.txt",_hess_cnt);
    MISCMATHS::write_ascii_matrix(numhess,fname); 
    _chgr = old_chgr;
  }
  _hess_cnt++;

  return(rval);
} EddyCatch

boost::shared_ptr<MISCMATHS::BFMatrix> MoveBySuscCFImpl::concatenate_subdiag_subm(const std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
										  unsigned int                                                              n) const EddyTry
{
  
  std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > tmp(n);
  for (unsigned int i=0; i<n; i++) {
    for (unsigned int j=0; j<n; j++) {
      if (!j) {
	if (_hp == MISCMATHS::BFMatrixFloatPrecision) {
	  const SparseBFMatrix<float>& tmp2 = dynamic_cast<SparseBFMatrix<float>& >(*subm[i][0]);
	  tmp[i] = boost::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<float>(tmp2));
	}
	else {
	  const SparseBFMatrix<double>& tmp2 = dynamic_cast<SparseBFMatrix<double>& >(*subm[i][0]);
	  tmp[i] = boost::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<double>(tmp2));
	}
      }
      else {
	if (j <= i) tmp[i]->HorConcat2MyRight(*subm[i][j]);
	else tmp[i]->HorConcat2MyRight(*(subm[j][i]->Transpose()));
      }
    }
  }
  
  boost::shared_ptr<MISCMATHS::BFMatrix> rval = tmp[0];
  for (unsigned int i=1; i<n; i++) rval->VertConcatBelowMe(*tmp[i]);

  return(rval);
} EddyCatch

boost::shared_ptr<MISCMATHS::BFMatrix> MoveBySuscCFImpl::concatenate_rect_subm(const std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
									       unsigned int                                                              m,
									       unsigned int                                                              n) const EddyTry
{
  
  std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > tmp(m);
  for (unsigned int i=0; i<m; i++) {
    for (unsigned int j=0; j<n; j++) {
      if (!j) {
	if (_hp == MISCMATHS::BFMatrixFloatPrecision) {
	  const SparseBFMatrix<float>& tmp2 = dynamic_cast<SparseBFMatrix<float>& >(*subm[i][0]);
	  tmp[i] = boost::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<float>(tmp2));
	}
	else {
	  const SparseBFMatrix<double>& tmp2 = dynamic_cast<SparseBFMatrix<double>& >(*subm[i][0]);
	  tmp[i] = boost::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<double>(tmp2));
	}
      }
      else tmp[i]->HorConcat2MyRight(*subm[i][j]);
    }
  }
  
  boost::shared_ptr<MISCMATHS::BFMatrix> rval = tmp[0];
  for (unsigned int i=1; i<m; i++) rval->VertConcatBelowMe(*tmp[i]);

  return(rval);
} EddyCatch

void MoveBySuscCFImpl::calculate_first_order_subm(
						  const std::vector<ImageVec>&                                        sop_dfdf,
						  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
						  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
						  const NEWIMAGE::volume<char>&                                       lmask,
						  unsigned int                                                        nvox,
						  
						  std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const EddyTry
{
  std::vector<unsigned int> noderiv(3,0); 
  NEWIMAGE::volume<float> ones = sop_dfdf[0][0]; ones = 1.0;
  for (unsigned int i=0; i<no_of_first_order_fields(); i++) {
    for (unsigned int j=0; j<(i+1); j++) {
      subm[i][j] = _sp1[i]->JtJ(sop_dfdf[i][j],ones,&lmask,hessian_precision());
      subm[i][j]->MulMeByScalar(2.0/static_cast<double>(nvox));
      if (_ujm) {
	for (const auto& elem : sop_ff) {
	  subm[i][j]->AddToMe(*_sp1[i]->JtJ(elem.first,elem.second[i][j],ones,&lmask,hessian_precision()),2.0/static_cast<double>(nvox));
	}
	for (const auto& elem: sop_fdf) {
	  boost::shared_ptr<MISCMATHS::BFMatrix> tmp = _sp1[i]->JtJ(elem.first,elem.second[i][j],noderiv,ones,&lmask,hessian_precision());
	  subm[i][j]->AddToMe(*tmp,2.0/static_cast<double>(nvox));
	  subm[i][j]->AddToMe(*(tmp->Transpose()),2.0/static_cast<double>(nvox));
	}
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::calculate_second_order_subm(
						   const std::vector<ImageVec>&                                        sop_dfdf,
						   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
						   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
						   const NEWIMAGE::volume<char>&                                       lmask,
						   unsigned int                                                        nvox,
						   
						   std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const EddyTry
{
  if (_order == 2)
  {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_second_order_pmap();
    std::vector<unsigned int> noderiv(3,0); 
    NEWIMAGE::volume<float> ones = sop_dfdf[0][0]; ones = 1.0;
    
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<(i+1); j++) {
	std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
	std::vector<unsigned int> pindx = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { 
	  subm[i][j] = _sp2[i][j]->JtJ(sop_dfdf[i][j],ones,&lmask,hessian_precision());
	  subm[i][j]->MulMeByScalar(2.0/static_cast<double>(nvox));
	  if (_ujm) {
	    for (const auto& elem : sop_ff) {
	      subm[i][j]->AddToMe(*_sp2[i][j]->JtJ(elem.first,elem.second[i][j],ones,&lmask,hessian_precision()),(2.0/static_cast<double>(nvox)));
	    }
	    for (const auto& elem : sop_fdf) {
	      boost::shared_ptr<MISCMATHS::BFMatrix> tmp = _sp2[i][j]->JtJ(elem.first,elem.second[i][j],noderiv,ones,&lmask,hessian_precision());
	      subm[i][j]->AddToMe(*tmp,(2.0/static_cast<double>(nvox)));
	      subm[i][j]->AddToMe(*(tmp->Transpose()),(2.0/static_cast<double>(nvox)));
	    }
	  }
	}
      }
    }
    
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<(i+1); j++) {
	std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
	std::vector<unsigned int> pindx = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
	if (!is_first_index_pair(i,j,pindx,pmap)) { 
	  std::pair<unsigned int,unsigned int> iijj = find_first_index_pair(pindx,pmap);
	  if (i==j || iijj.first==iijj.second) { 
	    if (_hp == MISCMATHS::BFMatrixFloatPrecision) {
	      const SparseBFMatrix<float>& tmp = dynamic_cast<SparseBFMatrix<float>& >(*(subm[iijj.first][iijj.second]));
	      subm[i][j] = boost::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<float>(tmp));
	    }
	    else {
	      const SparseBFMatrix<double>& tmp = dynamic_cast<SparseBFMatrix<double>& >(*(subm[iijj.first][iijj.second]));
	      subm[i][j] = boost::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<double>(tmp));
	    }
	  }
	  else subm[i][j] = subm[iijj.first][iijj.second];
	}
      }
    }    
  }
  else throw EddyException("MoveBySuscCFImpl::calculate_second_order_subm: I should not be here.");
  return;
} EddyCatch

void MoveBySuscCFImpl::calculate_cross_subm(
					    const std::vector<ImageVec>&                                        sop_dfdf,
					    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
					    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
					    const NEWIMAGE::volume<char>&                                       lmask,
					    unsigned int                                                        nvox,
					    
					    std::vector<std::vector<boost::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const EddyTry
{
  if (_order == 2)
  {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_cross_pmap();
    std::vector<unsigned int> noderiv(3,0); 
    NEWIMAGE::volume<float> ones = sop_dfdf[0][0]; ones = 1.0;
    
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	std::vector<unsigned int> pindx = make_index_vector(first_pair,j,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { 
	  subm[i][j] = _sp2[i][j]->JtJ(sop_dfdf[i][j],ones,&lmask,hessian_precision());
	  subm[i][j]->MulMeByScalar(2.0/static_cast<double>(nvox));
	  if (_ujm) {
	    for (const auto& elem : sop_ff) {
	      subm[i][j]->AddToMe(*_sp2[i][j]->JtJ(elem.first,elem.second[i][j],ones,&lmask,hessian_precision()),2.0/static_cast<double>(nvox));
	    }
	    for (const auto& elem : sop_fdf) {
	      boost::shared_ptr<MISCMATHS::BFMatrix> tmp = _sp2[i][j]->JtJ(elem.first,elem.second[i][j],noderiv,ones,&lmask,hessian_precision());
	      subm[i][j]->AddToMe(*tmp,2.0/static_cast<double>(nvox));
	      subm[i][j]->AddToMe(*(tmp->Transpose()),2.0/static_cast<double>(nvox));
	    }
	  }
	}
      }
    }
    
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	std::vector<unsigned int> pindx = make_index_vector(first_pair,j,no_of_first_order_fields());
	if (!is_first_index_pair(i,j,pindx,pmap)) { 
	  std::pair<unsigned int,unsigned int> iijj = find_first_index_pair(pindx,pmap);
	  subm[i][j] = subm[iijj.first][iijj.second];
	}
      }
    }    
  }
  else throw EddyException("MoveBySuscCFImpl::calculate_cross_subm: I should not be here.");
  return;
} EddyCatch

bool MoveBySuscCFImpl::is_first_index_pair(unsigned int                                                 i,
					   unsigned int                                                 j,
					   const std::vector<unsigned int>&                             ivec,
					   const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const EddyTry
{
  if (pmap[i][j] != ivec) throw EddyException("MoveBySuscCFImpl::is_first_index_pair: ivec is not the i-j'th member of pmap.");
  std::pair<unsigned int,unsigned int> first = find_first_index_pair(ivec,pmap);
  if (first.first==i && first.second==j) return(true);
  return(false);
} EddyCatch

std::pair<unsigned int,unsigned int> MoveBySuscCFImpl::find_first_index_pair(const std::vector<unsigned int>&                             ivec,
									     const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const EddyTry
{
  std::pair<unsigned int,unsigned int> first(0,0);
  for ( ; first.first<pmap.size(); first.first++) {
    for ( ; first.second<pmap[first.first].size(); first.second++) {
      if (pmap[first.first][first.second] == ivec) return(first);
    }
  }
  throw EddyException("MoveBySuscCFImpl::find_first_index_pair: ivec is not a member of pmap.");
} EddyCatch

std::vector<std::vector<std::vector<unsigned int> > > MoveBySuscCFImpl::build_second_order_pmap() const EddyTry
{  
  std::vector<std::vector<std::vector<unsigned int> > > pmap(no_of_second_order_fields());
  for (unsigned int i=0; i<pmap.size(); i++) {
    std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
    pmap[i].resize(i+1);
    for (unsigned j=0; j<(i+1); j++) {
      std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
      pmap[i][j] = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
    }
  }
  return(pmap);
} EddyCatch

std::vector<std::vector<std::vector<unsigned int> > > MoveBySuscCFImpl::build_cross_pmap() const EddyTry
{  
  std::vector<std::vector<std::vector<unsigned int> > > pmap(no_of_second_order_fields());
  for (unsigned int i=0; i<pmap.size(); i++) {
    std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
    pmap[i].resize(no_of_first_order_fields());
    for (unsigned j=0; j<no_of_first_order_fields(); j++) {
      pmap[i][j] = make_index_vector(first_pair,j,no_of_first_order_fields());
    }
  }
  return(pmap);
} EddyCatch

std::vector<unsigned int> MoveBySuscCFImpl::make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
							      const std::pair<unsigned int,unsigned int>& p2,
							      unsigned int                                np) const EddyTry
{
  std::vector<unsigned int> rval(np,0);
  rval[p1.first]++; rval[p1.second]++; rval[p2.first]++; rval[p2.second]++; 
  return(rval);
} EddyCatch

std::vector<unsigned int> MoveBySuscCFImpl::make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
							      unsigned int                                i,
							      unsigned int                                np) const EddyTry
{
  std::vector<unsigned int> rval(np,0);
  rval[p1.first]++; rval[p1.second]++; rval[i]++;
  return(rval);
} EddyCatch

std::pair<unsigned int,unsigned int> MoveBySuscCFImpl::get_second_order_index_pair(unsigned int i,
										   unsigned int np) const EddyTry
{
  std::pair<unsigned int,unsigned int> indx;
  for (indx.first = 0; indx.first<np; indx.first++) {
    for (indx.second = 0; indx.second<(indx.first+1); indx.second++) {
      if (indx.first + indx.second == i) return(indx);
    }
  }
  throw EddyException("MoveBySuscCFImpl::get_second_order_index_pair: I should not be here");
} EddyCatch

NEWMAT::ColumnVector MoveBySuscCFImpl::stl2newmat(const std::vector<double>& stl) const EddyTry
{
  NEWMAT::ColumnVector nm(stl.size());
  for (unsigned int i=0; i<stl.size(); i++) nm(i+1) = stl[i];
  return(nm);
} EddyCatch   

NEWMAT::Matrix MoveBySuscCFImpl::linspace(const NEWMAT::Matrix& inmat) const EddyTry
{
  NEWMAT::Matrix rval(inmat.Nrows(),inmat.Ncols());
  for (int r=1; r<=inmat.Nrows(); r++) rval(r,1) = r;
  return(rval);
} EddyCatch

void MoveBySuscCFImpl::resize_containers_for_sum_images_for_grad(const std::vector<std::vector<unsigned int> >& indicies,
								 const std::vector<EDDY::ScanType>&             imtypes) EddyTry
{
  _sum_fo_deriv_diff.resize(_mps.size());
  if (_order==2) {
    _sum_so_deriv_diff.resize(_mps.size());
  }
  if (_ujm) {
    for (unsigned int ti=0; ti<indicies.size(); ti++) { 
      for (unsigned int i=0; i<indicies[ti].size(); i++) {
	NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	BpeVec bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
	if (_sum_fo_ima_diff.find(bpe) == _sum_fo_ima_diff.end()) { 
	  _sum_fo_ima_diff[bpe].resize(_mps.size()); 
	  if (_order==2) {
	    _sum_so_ima_diff[bpe].resize(_mps.size()); 
	    for (unsigned int pi=0; pi<_mps.size(); pi++) {
	      _sum_so_ima_diff[bpe][pi].resize(pi+1);
	    }
	  }
	}
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::resize_containers_for_sum_images_for_hess(const std::vector<std::vector<unsigned int> >& indicies,
								 const std::vector<EDDY::ScanType>&             imtypes) EddyTry
{
  
  _sum_fo_deriv_deriv.resize(_mps.size());
  for (unsigned int pi=0; pi<_mps.size(); pi++) _sum_fo_deriv_deriv[pi].resize(pi+1);
  if (_ujm) {
    for (unsigned int ti=0; ti<indicies.size(); ti++) { 
      for (unsigned int i=0; i<indicies[ti].size(); i++) {
	NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	BpeVec bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
	if (_sum_fo_ima_ima.find(bpe) == _sum_fo_ima_ima.end()) { 
	  _sum_fo_ima_ima[bpe].resize(_mps.size());
	  _sum_fo_deriv_ima[bpe].resize(_mps.size());
	  for (unsigned int pi=0; pi<_mps.size(); pi++) {
	    _sum_fo_ima_ima[bpe][pi].resize(pi+1);
	    _sum_fo_deriv_ima[bpe][pi].resize(pi+1);
	  }
	}
      }
    }    
  }
  
  if (_order == 2) {
    _sum_so_deriv_deriv.resize(no_of_second_order_fields());
    _sum_cross_deriv_deriv.resize(no_of_second_order_fields());
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      _sum_so_deriv_deriv[i].resize(i+1);
      _sum_cross_deriv_deriv.resize(no_of_first_order_fields());
    }
    if (_ujm) {
      for (unsigned int ti=0; ti<indicies.size(); ti++) { 
	for (unsigned int i=0; i<indicies[ti].size(); i++) {
	  NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  BpeVec bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
	  if (_sum_so_ima_ima.find(bpe) == _sum_so_ima_ima.end()) { 
	    _sum_so_ima_ima[bpe].resize(no_of_second_order_fields());
	    _sum_so_deriv_ima[bpe].resize(no_of_second_order_fields());
	    _sum_cross_ima_ima[bpe].resize(no_of_second_order_fields());
	    _sum_cross_deriv_ima[bpe].resize(no_of_second_order_fields());
	    for (unsigned j=0; j<no_of_second_order_fields(); j++) {
	      _sum_so_ima_ima[bpe][j].resize(j+1);
	      _sum_so_deriv_ima[bpe][j].resize(j+1);
	      _sum_cross_ima_ima[bpe][j].resize(no_of_first_order_fields());
	      _sum_cross_deriv_ima[bpe][j].resize(no_of_first_order_fields());
	    }
	  }
	}
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::set_sum_images_to_zero(const NEWIMAGE::volume<float>& ima) EddyTry
{
  NEWIMAGE::volume<float> zeros = ima; zeros = 0.0;
  
  _sum_sqr_diff = zeros;
  
  for (unsigned int pi=0; pi<_mps.size(); pi++) {
    _sum_fo_deriv_diff[pi] = zeros;
    for (auto& elem : _sum_fo_ima_diff) elem.second[pi] = zeros;
    if (_order == 2) {
      for (unsigned int pj=0; pj<(pi+1); pj++) {
	_sum_so_deriv_diff[pi][pj] = zeros;
	for (auto& elem : _sum_so_ima_diff) elem.second[pi][pj] = zeros;
      }
    }
  }
  
  for (unsigned int pi=0; pi<_mps.size(); pi++) {
    for (unsigned int pj=0; pj<(pi+1); pj++) {
      _sum_fo_deriv_deriv[pi][pj] = zeros;
      for (auto& elem : _sum_fo_ima_ima) elem.second[pi][pj] = zeros;
      for (auto& elem : _sum_fo_deriv_ima) elem.second[pi][pj] = zeros;
    }
  }
  if (_order == 2) {
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      for (unsigned int j=0; j<(i+1); j++) {
	_sum_so_deriv_deriv[i][j] = zeros;
	for (auto& elem : _sum_so_ima_ima) elem.second[i][j] = zeros;
	for (auto& elem : _sum_so_deriv_ima) elem.second[i][j] = zeros;
      }
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	_sum_cross_deriv_deriv[i][j] = zeros;
	for (auto& elem : _sum_cross_ima_ima) elem.second[i][j] = zeros;
	for (auto& elem : _sum_cross_deriv_ima) elem.second[i][j] = zeros;
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::recalculate_sum_so_imas_for_hess(const NEWMAT::ColumnVector&           p,
							const BpeVec&                         bpe,
							const NEWIMAGE::volume<float>&        ima,
							const NEWIMAGE::volume<float>&        deriv) EddyTry
{
  if (!_utd && _order==2) {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_second_order_pmap();
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<(i+1); j++) {
	std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
	std::vector<unsigned int> pindx = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { 
	  float factor=1.0;
	  for (unsigned int ii=0; ii<pindx.size(); ii++) {
	    for (unsigned int jj=0; jj<pindx[ii]; jj++) factor *= p(_mps[ii]+1);
	  }
	  
	  _sum_so_deriv_deriv[i][j] += factor*deriv*deriv;
	  if (_ujm) {
	    
	    _sum_so_ima_ima[bpe][i][j] += factor*ima*ima;
	    
	    _sum_so_deriv_ima[bpe][i][j] += factor*ima*deriv;
	  }
	}
      }
    }
  }
  else throw EddyException("MoveBySuscCFImpl::recalculate_sum_so_imas_for_hess: I should not be here.");
  return;
} EddyCatch

void MoveBySuscCFImpl::recalculate_sum_cross_imas_for_hess(const NEWMAT::ColumnVector&           p,
							   const BpeVec&                         bpe,
							   const NEWIMAGE::volume<float>&        ima,
							   const NEWIMAGE::volume<float>&        deriv) EddyTry
{
  if (!_utd && _order==2) {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_cross_pmap();
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	std::vector<unsigned int> pindx = make_index_vector(first_pair,j,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { 
	  float factor=1.0;
	  for (unsigned int ii=0; ii<pindx.size(); ii++) {
	    for (unsigned int jj=0; jj<pindx[ii]; jj++) factor *= p(_mps[ii]+1);
	  }
	  
	  _sum_cross_deriv_deriv[i][j] += factor*deriv*deriv;
	  if (_ujm) {
	    
	    _sum_cross_ima_ima[bpe][i][j] += factor*ima*ima;
	    
	    _sum_cross_deriv_ima[bpe][i][j] += factor*ima*deriv;
	  }
	}
      }
    }
  }
  else throw EddyException("MoveBySuscCFImpl::recalculate_sum_cross_imas_for_hess: I should not be here.");
  return;
} EddyCatch

void MoveBySuscCFImpl::recalculate_sum_images() EddyTry
{
  if (!_utd || !_m_utd) {
    
    std::vector<std::vector<unsigned int> > indicies = {_b0s, _dwis};
    std::vector<EDDY::ScanType> imtypes = {B0, DWI};
    std::vector<NEWIMAGE::volume<float> > masks = {_sm.Scan(0,ANY).GetIma(), _sm.Scan(0,ANY).GetIma()};
    std::vector<std::shared_ptr<EDDY::DWIPredictionMaker> > pmps(2,nullptr);
    if (_bno) { 
      resize_containers_for_sum_images_for_grad(indicies,imtypes);
      resize_containers_for_sum_images_for_hess(indicies,imtypes);
      
      if (_clo.NVoxHp() < 10000) _clo.SetNVoxHp(10000); 
#ifdef COMPILE_GPU 
      if (indicies[0].size()) pmps[0] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); 
      if (indicies[1].size()) {  
	pmps[1] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]);
	_hypar = this->stl2newmat(pmps[1]->GetHyperPar());
      }
#else
      if (indicies[0].size()) pmps[0] = EDDY::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); 
      if (indicies[1].size()) {  
	pmps[1] = EDDY::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]);
	_hypar = this->stl2newmat(pmps[1]->GetHyperPar());
      }
#endif
      _bno = false;
    }
    else {
      
      _clo.SetHyperParValues(_hypar); 
#ifdef COMPILE_GPU 
      if (indicies[0].size()) pmps[0] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); 
      if (indicies[1].size()) pmps[1] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]); 
#else
      if (indicies[0].size()) pmps[0] = EDDY::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); 
      if (indicies[1].size()) pmps[1] = EDDY::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]); 
#endif
    }

    
    _mask = masks[0] * masks[1];
    _nvox = static_cast<unsigned int>(std::round(_mask.sum()));
    _cmask.reinitialize(_mask.xsize(),_mask.ysize(),_mask.zsize());
    NEWIMAGE::copybasicproperties(_mask,_cmask);
    std::vector<int> tsz = {static_cast<int>(_mask.xsize()), static_cast<int>(_mask.ysize()), static_cast<int>(_mask.zsize())};
    for (int k=0; k<tsz[2]; k++) for (int j=0; j<tsz[1]; j++) for (int i=0; i<tsz[0]; i++) _cmask(i,j,k) = (_mask(i,j,k) > 0.0) ? 1 : 0;

    
    
    set_sum_images_to_zero(_mask);
    
    
    NEWIMAGE::volume4D<float> deriv(_mask.xsize(),_mask.ysize(),_mask.zsize(),3);
    NEWIMAGE::copybasicproperties(_mask,deriv);
    NEWIMAGE::volume<float> sderiv = _mask;  
    std::vector<double> vxs = { _mask.xdim(), _mask.ydim(), _mask.zdim() };
    for (unsigned int ti=0; ti<indicies.size(); ti++) {  
      for (unsigned int i=0; i<indicies[ti].size(); i++) { 
	NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
        NEWMAT::ColumnVector hz2mm = _sm.Scan(indicies[ti][i],imtypes[ti]).GetHz2mmVector();
	std::vector<unsigned int> bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
#ifdef COMPILE_GPU 
	NEWIMAGE::volume<float> vol = EddyGpuUtils::GetVolumetricUnwarpedScan(_sm.Scan(indicies[ti][i],imtypes[ti]),_sm.GetSuscHzOffResField(indicies[ti][i],imtypes[ti]),false,_sm.GetPolation(),nullptr,&deriv);
#else
	NEWIMAGE::volume<float> vol = _sm.Scan(indicies[ti][i],imtypes[ti]).GetVolumetricUnwarpedIma(_sm.GetSuscHzOffResField(indicies[ti][i],imtypes[ti]),deriv);
#endif
	NEWIMAGE::volume<float> diff = vol - pmps[ti]->Predict(indicies[ti][i]);
	_sum_sqr_diff += diff * diff;
	sderiv = static_cast<float>(hz2mm(1)/vxs[0])*deriv[0] + static_cast<float>(hz2mm(2)/vxs[1])*deriv[1] + static_cast<float>(hz2mm(3)/vxs[2])*deriv[2];
        if (_ujm) vol *= static_cast<float>(hz2mm(1)/vxs[0] + hz2mm(2)/vxs[1] + hz2mm(3)/vxs[2]); 
        
	for (unsigned int pi=0; pi<_mps.size(); pi++) {
	  _sum_fo_deriv_diff[pi] += static_cast<float>(p(_mps[pi]+1))*sderiv*diff;
	  if (_ujm) _sum_fo_ima_diff[bpe][pi] += static_cast<float>(p(_mps[pi]+1))*vol*diff;
	  if (_order == 2) {
	    for (unsigned int pj=0; pj<(pi+1); pj++) {
	      _sum_so_deriv_diff[pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*sderiv*diff;
	      if (_ujm) _sum_so_ima_diff[bpe][pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*vol*diff;
	    }
	  }
	}
	
	for (unsigned int pi=0; pi<_mps.size(); pi++) {
	  for (unsigned int pj=0; pj<(pi+1); pj++) {
	    _sum_fo_deriv_deriv[pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*sderiv*sderiv;
	    if (_ujm) {
	      _sum_fo_ima_ima[bpe][pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*vol*vol;
	      _sum_fo_deriv_ima[bpe][pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*sderiv*vol;
	    }
	  }
	}
	if (_order == 2) { 
	  recalculate_sum_so_imas_for_hess(p,bpe,vol,sderiv);     
	  recalculate_sum_cross_imas_for_hess(p,bpe,vol,sderiv);  
	}
      }
    }
    _m_utd = true;
    _utd = true;
  }

  return;
} EddyCatch

} 

