/*! \file BiasFieldEstimatorImpl.cpp
    \brief Contains one implementation of class for estimation of a bias field

    \author Jesper Andersson
    \version 1.0b, December, 2017.
*/
// 
// BiasFieldEstimatorImpl.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2017 University of Oxford 
//

#ifndef BiasFieldEstimatorImpl_h
#define BiasFieldEstimatorImpl_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include <boost/shared_ptr.hpp>
#include "newmat.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "newimage/newimageall.h"
#include "miscmaths/SpMat.h"
#include "ECScanClasses.h"
#include "BiasFieldEstimator.h"

namespace EDDY {

class BiasFieldEstimatorImpl
{
public:
  BiasFieldEstimatorImpl() : _nima(0) {}
  
  void SetRefScan(const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  
  void AddScan(const NEWIMAGE::volume<float>& predicted, const NEWIMAGE::volume<float>& observed, 
	       const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  
  NEWIMAGE::volume<float> GetField(double ksp, double lambda) const;
  
  MISCMATHS::SpMat<float> GetAtMatrix(const EDDY::ImageCoordinates&  coords, 
				      const NEWIMAGE::volume<float>& predicted, 
				      const NEWIMAGE::volume<float>& mask) const;
  
  MISCMATHS::SpMat<float> GetAtStarMatrix(const EDDY::ImageCoordinates&  coords, const NEWIMAGE::volume<float>& mask) const;
  
  void Write(const std::string& basename) const { 
    MISCMATHS::write_ascii_matrix(_Aty,"BiasFieldEstimatorImpl_" + basename + "_Aty.txt"); 
    _AtA.Print("BiasFieldEstimatorImpl_" + basename + "_AtA.txt");
  }
private:
  unsigned int              _nima;    
  std::vector<unsigned int> _isz;     
  std::vector<double>       _vxs;     
  MISCMATHS::SpMat<float>   _rAt;     
  MISCMATHS::SpMat<float>   _AtA;     
  NEWMAT::ColumnVector      _Aty;     

  unsigned int nvox() const { return(_isz[0]*_isz[1]*_isz[2]); }
  unsigned int ijk2indx(unsigned int i, unsigned int j, unsigned int k, const EDDY::ImageCoordinates& c) const {
    return(k*c.NX()*c.NY() + j*c.NX() + i);
  }
  void get_wgts(const double& dx, const double& dy, const double& dz,
		double& w000, double& w100, double& w010, double& w110,
		double& w001, double& w101, double& w011, double& w111) const
  {
    w000 = w100 = w010 = w110 = 1.0 - dz;
    w001 = w101 = w011 = w111 = dz;
    w000 *= 1.0 - dy; w100 *= 1.0 - dy; w001 *= 1.0 - dy; w101 *= 1.0 - dy;
    w010 *= dy; w110 *= dy; w011 *= dy; w111 *= dy;
    w000 *= 1.0 - dx; w010 *= 1.0 - dx; w001 *= 1.0 - dx; w011 *= 1.0 - dx;
    w100 *= dx; w110 *= dx; w101 *= dx; w111 *= dx;
  }
  unsigned int make_At_star_CSC(
				const EDDY::ImageCoordinates&  coords,
				const NEWIMAGE::volume<float>& mask,
				
				unsigned int*&                 irp,    
				unsigned int*&                 jcp,
				double*&                       sp) const;
  void multiply_At_star_CSC_by_image(const NEWIMAGE::volume<float>& ima,
				     const NEWIMAGE::volume<float>& mask,
				     unsigned int* const            jcp,
				     double* const                  sp) const;

  
};

unsigned int BiasFieldEstimatorImpl::make_At_star_CSC(
						      const EDDY::ImageCoordinates&  coords,
						      const NEWIMAGE::volume<float>& mask,
						      
						      unsigned int*&                 irp,    
						      unsigned int*&                 jcp,
						      double*&                       sp) const
{
  
  irp = new unsigned int[8*coords.N()]; 
  jcp = new unsigned int[coords.N()+1];
  sp = new double[8*coords.N()];
  unsigned int ii = 0; 
  unsigned int ji = 0; 
  NEWMAT::ColumnVector vmask = mask.vec();
  for (unsigned int i=0; i<coords.N(); i++) {
    jcp[ji++] = ii;
    if (vmask(i+1) > 0.0 && coords.IsInBounds(i)) { 
      unsigned int xi = floor(coords.x(i));
      unsigned int yi = floor(coords.y(i));
      unsigned int zi = floor(coords.z(i));
      double dx = coords.x(i) - xi;
      double dy = coords.y(i) - yi;
      double dz = coords.z(i) - zi;
      if (dx < 1e-6 && dy < 1e-6 && dz < 1e-6) { 
	irp[ii] = this->ijk2indx(xi,yi,zi,coords);
	sp[ii++] = 1.0;
      }
      else {
	double w000, w100, w010, w110, w001, w101, w011, w111;
	get_wgts(dx,dy,dz,w000,w100,w010,w110,w001,w101,w011,w111);
	if (w000 > 1e-6) { irp[ii] = this->ijk2indx(xi,yi,zi,coords); sp[ii++] = w000; }
	if (w100 > 1e-6) { irp[ii] = this->ijk2indx(xi+1,yi,zi,coords); sp[ii++] = w100; }
	if (w010 > 1e-6) { irp[ii] = this->ijk2indx(xi,yi+1,zi,coords); sp[ii++] = w010; }
	if (w110 > 1e-6) { irp[ii] = this->ijk2indx(xi+1,yi+1,zi,coords); sp[ii++] = w110; }
	if (w001 > 1e-6) { irp[ii] = this->ijk2indx(xi,yi,zi+1,coords); sp[ii++] = w001; }
	if (w101 > 1e-6) { irp[ii] = this->ijk2indx(xi+1,yi,zi+1,coords); sp[ii++] = w101; }
	if (w011 > 1e-6) { irp[ii] = this->ijk2indx(xi,yi+1,zi+1,coords); sp[ii++] = w011; }
	if (w111 > 1e-6) { irp[ii] = this->ijk2indx(xi+1,yi+1,zi+1,coords); sp[ii++] = w111; }
      }
    }
  }
  jcp[ji] = ii; 
  return(ii);
}

void BiasFieldEstimatorImpl::multiply_At_star_CSC_by_image(const NEWIMAGE::volume<float>& ima,
							   const NEWIMAGE::volume<float>& mask,
							   unsigned int* const            jcp,
							   double* const                  sp) const
{
  unsigned int indx = 0;
  for (int k=0; k<ima.zsize(); k++) {
    for (int j=0; j<ima.ysize(); j++) {
      for (int i=0; i<ima.xsize(); i++) {
	if (mask(i,j,k)) { for (unsigned int ii=jcp[indx]; ii<jcp[indx+1]; ii++) sp[ii] *= ima(i,j,k); }
	indx++;
      }
    }
  }
}

void BiasFieldEstimatorImpl::SetRefScan(const NEWIMAGE::volume<float>& mask,
					const EDDY::ImageCoordinates&  coords)
{
  if (!_rAt.Nrows()) { 
    _isz.resize(3); 
    _isz[0] = static_cast<unsigned int>(mask.xsize());
    _isz[1] = static_cast<unsigned int>(mask.ysize());
    _isz[2] = static_cast<unsigned int>(mask.zsize());
    _vxs.resize(3);
    _vxs[0] = static_cast<double>(mask.xdim());
    _vxs[1] = static_cast<double>(mask.ydim());
    _vxs[2] = static_cast<double>(mask.zdim());
  }
  else { 
    if (static_cast<unsigned int>(mask.xsize()) != _isz[0] || 
	static_cast<unsigned int>(mask.ysize()) != _isz[1] ||
	static_cast<unsigned int>(mask.zsize()) != _isz[2]) {
      throw EddyException("BiasFieldEstimatorImpl::SetRefScan: Size mismatch between new and previously set ref image");
    }
  }
  unsigned int *irp = nullptr; 
  unsigned int *jcp = nullptr; 
  double *sp = nullptr;        
  make_At_star_CSC(coords,mask,irp,jcp,sp); 
  _rAt = MISCMATHS::SpMat<float>(coords.N(),coords.N(),irp,jcp,sp);
  _rAt.Save("Ref_At_matrix.txt");
  delete[] irp; delete[] jcp; delete[] sp;
    
}

void BiasFieldEstimatorImpl::AddScan(const NEWIMAGE::volume<float>& predicted, 
				     const NEWIMAGE::volume<float>& observed,
				     const NEWIMAGE::volume<float>& mask,
				     const EDDY::ImageCoordinates&  coords)
{
  static int cnt = 0;

  if (predicted.xsize() != observed.xsize() ||
      predicted.ysize() != observed.ysize() ||
      predicted.zsize() != observed.zsize()) {
    throw EddyException("BiasFieldEstimatorImpl::AddScan: Size mismatch between predicted and observed image");
  }
  if (!_rAt.Nrows()) { 
    throw EddyException("BiasFieldEstimatorImpl::AddScan: Attempting to add scan before ref scan has been set");
  }
  if (static_cast<unsigned int>(predicted.xsize()) != _isz[0] || 
      static_cast<unsigned int>(predicted.ysize()) != _isz[1] ||
      static_cast<unsigned int>(predicted.zsize()) != _isz[2]) {
    throw EddyException("BiasFieldEstimatorImpl::AddScan: Size mismatch between predicted and previously set ref image");
  }     

  NEWMAT::ColumnVector v_predicted = predicted.vec();
  NEWMAT::ColumnVector v_observed = observed.vec();
  NEWMAT::ColumnVector v_mask = mask.vec();
  _nima++;
  cnt++;
  unsigned int *irp = nullptr; 
  unsigned int *jcp = nullptr; 
  double *sp = nullptr;        
  
  make_At_star_CSC(coords,mask,irp,jcp,sp); 
  
  
  
  
  
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),irp,jcp,sp);

  char fname[256]; sprintf(fname,"At_%02d.txt",cnt); 

  cout << "At -= _rAt;" << endl;
  At -= _rAt;

  

  cout << "At.MultiplyColumns(NEWMAT::SP(v_predicted,v_mask));" << endl;
  At.MultiplyColumns(NEWMAT::SP(v_predicted,v_mask));
  if (!_Aty.Nrows()) _Aty = At * NEWMAT::SP((v_observed - v_predicted),v_mask);
  else _Aty += At * NEWMAT::SP((v_observed - v_predicted),v_mask);
  cout << "AtA = At*At.t()" << endl;
  
  if (!_AtA.Nrows()) _AtA = At*At.t();
  else _AtA += At*At.t();
  delete[] irp; delete[] jcp; delete[] sp;

  
}

NEWIMAGE::volume<float> BiasFieldEstimatorImpl::GetField(double ksp,           
							 double lambda) const  
{
  if (_nima==0) {
    throw EddyException("BiasFieldEstimatorImpl::GetField: The field cannot be estimated until images have been loaded");
  }
  
  std::vector<unsigned int> iksp(3);
  iksp[0] = static_cast<unsigned int>((ksp / _vxs[0]) + 0.5);
  iksp[1] = static_cast<unsigned int>((ksp / _vxs[1]) + 0.5);
  iksp[2] = static_cast<unsigned int>((ksp / _vxs[2]) + 0.5);
  
  BASISFIELD::splinefield spf(_isz,_vxs,iksp);
  
  boost::shared_ptr<MISCMATHS::SpMat<float> > B = spf.J();
  MISCMATHS::SpMat<float> BtAtAB = B->t() * _AtA * *B;
  BtAtAB += (lambda/static_cast<double>(nvox())) * (*spf.BendEnergyHessAsSpMat());
  NEWMAT::ColumnVector BtAtf = B->t() * _Aty;
  
  NEWMAT::ColumnVector b = BtAtAB.SolveForx(BtAtf,SYM_POSDEF,1.0e-6,200);
  
  spf.SetCoef(b);
  
  NEWIMAGE::volume<float> field(static_cast<int>(_isz[0]),static_cast<int>(_isz[1]),static_cast<int>(_isz[2]));
  field.setxdim(_vxs[0]); field.setydim(_vxs[1]); field.setzdim(_vxs[2]);
  spf.AsVolume(field);
  
  return(field);
}

MISCMATHS::SpMat<float> BiasFieldEstimatorImpl::GetAtMatrix(const EDDY::ImageCoordinates&  coords,
							    const NEWIMAGE::volume<float>& predicted,
							    const NEWIMAGE::volume<float>& mask) const
{
  unsigned int *irp = nullptr;
  unsigned int *jcp = nullptr;
  double *sp = nullptr;
  make_At_star_CSC(coords,mask,irp,jcp,sp);
  multiply_At_star_CSC_by_image(predicted,mask,jcp,sp);
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),irp,jcp,sp);
  return(At);
}

MISCMATHS::SpMat<float> BiasFieldEstimatorImpl::GetAtStarMatrix(const EDDY::ImageCoordinates&  coords, 
								const NEWIMAGE::volume<float>& mask) const
{
  unsigned int *irp = nullptr;
  unsigned int *jcp = nullptr;
  double *sp = nullptr;
  make_At_star_CSC(coords,mask,irp,jcp,sp);
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),irp,jcp,sp);
  return(At);
}

BiasFieldEstimator::BiasFieldEstimator()
{
  _pimpl = new BiasFieldEstimatorImpl();
}

BiasFieldEstimator::~BiasFieldEstimator() { delete _pimpl; }

void BiasFieldEstimator::SetRefScan(const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords)
{
  _pimpl->SetRefScan(mask,coords);
}

void BiasFieldEstimator::AddScan(const NEWIMAGE::volume<float>& predicted, const NEWIMAGE::volume<float>& observed, 
				 const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords)
{
  _pimpl->AddScan(predicted,observed,mask,coords);
}

NEWIMAGE::volume<float> BiasFieldEstimator::GetField(double ksp, double lambda) const { return(_pimpl->GetField(ksp,lambda)); } 

MISCMATHS::SpMat<float> BiasFieldEstimator::GetAtMatrix(const EDDY::ImageCoordinates&  coords, 
							const NEWIMAGE::volume<float>& predicted,
							const NEWIMAGE::volume<float>& mask) const
{
  return(_pimpl->GetAtMatrix(coords,predicted,mask));
}

MISCMATHS::SpMat<float> BiasFieldEstimator::GetAtStarMatrix(const EDDY::ImageCoordinates&  coords, 
							    const NEWIMAGE::volume<float>& mask) const
{
  return(_pimpl->GetAtStarMatrix(coords,mask));
}

void BiasFieldEstimator::Write(const std::string& basename) const { _pimpl->Write(basename); }

} 

#endif 

