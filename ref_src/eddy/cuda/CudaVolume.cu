#include <cstdlib>
#include <string>
#include <sys/time.h>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#include "newmat.h"
#include "miscmaths/miscmaths.h"

#ifndef EXPOSE_TREACHEROUS
#define I_CUDAVOLUME_H_DEFINED_ET
#define EXPOSE_TREACHEROUS           
#endif

#include "newimage/newimageall.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "EddyKernels.h"
#include "EddyFunctors.h"
#include "CudaVolume.h"

using namespace EDDY;
using namespace EddyKernels;

void CudaVolume::SetHdr(const CudaVolume4D& cv) EddyTry
{
  _spv=false; _sz=cv._sz; _hdr=cv._hdr; 
  try {
    _devec.resize(cv.Size()); _spcoef.clear();
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::SetHdr with message: " << e.what() << std::endl;
    throw;
  }
} EddyCatch  

void CudaVolume::Sample(const EDDY::CudaImageCoordinates& coord,
			CudaVolume&                       smpl) const EddyTry
{
  if (Interp()!=NEWIMAGE::spline && Interp()!=NEWIMAGE::trilinear) throw EddyException("CudaVolume::Sample: Invalid interpolation option");
  if (Extrap()!=NEWIMAGE::extraslice && Extrap()!=NEWIMAGE::periodic && Extrap()!=NEWIMAGE::mirror) throw EddyException("CudaVolume::Sample: Invalid extrapolation option");
  if (smpl!=*this) throw EddyException("CudaVolume::Sample: Dimension mismatch");

  if (Interp()==NEWIMAGE::spline && !_spv) {
    if (_spcoef.size() != _devec.size()) {
      try {
	_spcoef.resize(_devec.size());
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in CudaVolume::Sample_1 after call to resize with message: " << e.what() << std::endl;
	throw;
      }
    }
    calculate_spline_coefs(_sz,_devec,_spcoef);
    _spv = true;
  }

  int tpb = threads_per_block_interpolate; 
  int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::ExtrapType ep = EddyKernels::PERIODIC;
  if (Extrap()==NEWIMAGE::extraslice) ep = EddyKernels::CONSTANT;
  else if (Extrap()==NEWIMAGE::mirror) ep = EddyKernels::MIRROR;

  if (Interp()==NEWIMAGE::spline) {
    EddyKernels::spline_interpolate<<<nblocks,tpb>>>(Size(0),Size(1),Size(2),sp_ptr(),coord.XPtr(),
						     coord.YPtr(),coord.ZPtr(),nthreads,ep,smpl.GetPtr());
    EddyKernels::CudaSync("EddyKernels::spline_interpolate");
  }  
  else { 
    EddyKernels::linear_interpolate<<<nblocks,tpb>>>(Size(0),Size(1),Size(2),GetPtr(),coord.XPtr(),
						     coord.YPtr(),coord.ZPtr(),nthreads,ep,smpl.GetPtr());
    EddyKernels::CudaSync("EddyKernels::linear_interpolate");
  }
} EddyCatch

void CudaVolume::Sample(const EDDY::CudaImageCoordinates& coord,
			CudaVolume&                       smpl,
			CudaVolume4D&                     dsmpl) const EddyTry
{
  if (Interp()!=NEWIMAGE::spline && Interp()!=NEWIMAGE::trilinear) throw EddyException("CudaVolume::Sample: Invalid interpolation option");
  if (Extrap()!=NEWIMAGE::extraslice && Extrap()!=NEWIMAGE::periodic && Extrap()!=NEWIMAGE::mirror) throw EddyException("CudaVolume::Sample: Invalid extrapolation option");
  if (smpl!=(*this) || dsmpl!=(*this)) throw EddyException("CudaVolume::Sample: Dimension mismatch");
  if (dsmpl.Size(3)!=3) throw EddyException("CudaVolume::Sample: dsmpl.Size(3) must be 3");

  if (Interp()==NEWIMAGE::spline && !_spv) {
    if (_spcoef.size() != _devec.size()) {
      try {
	_spcoef.resize(_devec.size());
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in CudaVolume::Sample_2 after call to resize with message: " << e.what() << std::endl;
	throw;
      }
    }
    calculate_spline_coefs(_sz,_devec,_spcoef);
    _spv = true;
  }

  int tpb = threads_per_block_interpolate; 
  int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::ExtrapType ep = EddyKernels::PERIODIC;
  if (Extrap()==NEWIMAGE::extraslice) ep = EddyKernels::CONSTANT;
  else if (Extrap()==NEWIMAGE::mirror) ep = EddyKernels::MIRROR;

  if (Interp()==NEWIMAGE::spline) {
    EddyKernels::spline_interpolate<<<nblocks,tpb>>>(Size(0),Size(1),Size(2),sp_ptr(),coord.XPtr(),
						     coord.YPtr(),coord.ZPtr(),nthreads,ep,smpl.GetPtr(),
						     dsmpl.GetPtr(0),dsmpl.GetPtr(1),dsmpl.GetPtr(2));
    EddyKernels::CudaSync("EddyKernels::spline_interpolate");
  }
  else {
    EddyKernels::linear_interpolate<<<nblocks,tpb>>>(Size(0),Size(1),Size(2),GetPtr(),coord.XPtr(),
						     coord.YPtr(),coord.ZPtr(),nthreads,ep,smpl.GetPtr(),
						     dsmpl.GetPtr(0),dsmpl.GetPtr(1),dsmpl.GetPtr(2));
    EddyKernels::CudaSync("EddyKernels::linear_interpolate");
  }
} EddyCatch

void CudaVolume::ValidMask(const EDDY::CudaImageCoordinates& coord, CudaVolume& mask) const EddyTry
{
  int tpb = threads_per_block_interpolate; 
  int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;

  std::vector<bool> epval = ExtrapValid();
  EddyKernels::valid_voxels<<<nblocks,tpb>>>(Size(0),Size(1),Size(2),epval[0],epval[1],epval[2],
					     coord.XPtr(),coord.YPtr(),coord.ZPtr(),nthreads,mask.GetPtr());
  EddyKernels::CudaSync("EddyKernels::valid_voxels");
} EddyCatch

bool CudaVolume::operator==(const CudaVolume4D& rhs) const EddyTry {
  return(this->_sz[0]==rhs.Size(0) && this->_sz[1]==rhs.Size(1) && this->_sz[2]==rhs.Size(2) &&
	 fabs(this->_hdr.xdim()-rhs.Vxs(0))<1e-6 && fabs(this->_hdr.ydim()-rhs.Vxs(1))<1e-6 && fabs(this->_hdr.zdim()-rhs.Vxs(2))<1e-6);
} EddyCatch

CudaVolume& CudaVolume::operator+=(const CudaVolume& cv) EddyTry
{
  if (*this != cv) throw EddyException("CudaVolume::operator+=: Mismatched volumes");
  if (!this->Size()) throw EddyException("CudaVolume::operator+=: Empty volume");
  try {
    thrust::transform(_devec.begin(),_devec.end(),cv._devec.begin(),_devec.begin(),thrust::plus<float>());
    if (_spv && cv._spv) {
      thrust::transform(_spcoef.begin(),_spcoef.end(),cv._spcoef.begin(),_spcoef.begin(),thrust::plus<float>());
    }
    else _spv=false;
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::operator+= with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

CudaVolume& CudaVolume::operator-=(const CudaVolume& cv) EddyTry
{
  if (*this != cv) throw EddyException("CudaVolume::operator-=: Mismatched volumes");
  if (!this->Size()) throw EddyException("CudaVolume::operator-=: Empty volume");
  try {
    thrust::transform(_devec.begin(),_devec.end(),cv._devec.begin(),_devec.begin(),thrust::minus<float>());
    if (_spv && cv._spv) {
      thrust::transform(_spcoef.begin(),_spcoef.end(),cv._spcoef.begin(),_spcoef.begin(),thrust::minus<float>());
    }
    else _spv=false;
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::operator-= with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

CudaVolume& CudaVolume::operator*=(const CudaVolume& cv) EddyTry
{
  if (*this != cv) throw EddyException("CudaVolume::operator*=: Mismatched volumes");
  if (!this->Size()) throw EddyException("CudaVolume::operator*=: Empty volume");
  try {
    thrust::transform(_devec.begin(),_devec.end(),cv._devec.begin(),_devec.begin(),thrust::multiplies<float>());
    _spv=false;
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::operator*= with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

CudaVolume& CudaVolume::operator/=(float a) EddyTry
{
  if (!a) throw EddyException("CudaVolume::operator/=: Division by zero");
  try {
    thrust::transform(_devec.begin(),_devec.end(),_devec.begin(),EDDY::MulByScalar<float>(1.0/a));
    if (_spv) thrust::transform(_spcoef.begin(),_spcoef.end(),_spcoef.begin(),EDDY::MulByScalar<float>(1.0/a));
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::operator/= with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

void CudaVolume::Smooth(float fwhm,                      
			const CudaVolume& mask) EddyTry  
{
  CudaVolume smask=mask;
  *this *= mask;      
  this->Smooth(fwhm); 
  smask.Smooth(fwhm); 
  this->DivideWithinMask(smask,mask);
  *this *= mask;
} EddyCatch

void CudaVolume::MultiplyAndAddToMe(const CudaVolume& pv, float a) EddyTry
{
  if (pv!=*this) throw EddyException("CudaVolume::MultiplyAndAddToMe: Dimension mismatch");
  try {
    thrust::transform(_devec.begin(),_devec.end(),pv._devec.begin(),_devec.begin(),EDDY::MulAndAdd<float>(a));
    if (_spv) {
      if (pv._spv) thrust::transform(_spcoef.begin(),_spcoef.end(),pv._spcoef.begin(),_spcoef.begin(),EDDY::MulAndAdd<float>(a));
      else { _spcoef.clear(); _spv=false; };
    }
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::MultiplyAndAddToMe: with message: " << e.what() << std::endl;
    throw;
  }
} EddyCatch

void CudaVolume::SubtractMultiplyAndAddToMe(const CudaVolume& pv, const CudaVolume& nv, float a) EddyTry
{
  if (pv!=*this || nv!=*this) throw EddyException("CudaVolume::SubtractMultiplyAndAddToMe: Dimension mismatch");
  int tpb = threads_per_block_smaatm;
  int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;

  EddyKernels::subtract_multiply_and_add_to_me<<<nblocks,tpb>>>(pv.GetPtr(),nv.GetPtr(),a,nthreads,GetPtr());
  EddyKernels::CudaSync("EddyKernels::subtract_multiply_and_add_to_me");

  if (_spv) { _spcoef.clear(); _spv=false; } 
  
  return;
} EddyCatch

void CudaVolume::SubtractSquareAndAddToMe(const CudaVolume& pv, const CudaVolume& nv) EddyTry
{
  if (pv!=*this || nv!=*this) throw EddyException("CudaVolume::SubtractSquareAndAddToMe: Dimension mismatch");
  int tpb = threads_per_block_ssaatm;
  int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;

  EddyKernels::subtract_square_and_add_to_me<<<nblocks,tpb>>>(pv.GetPtr(),nv.GetPtr(),nthreads,GetPtr());
  EddyKernels::CudaSync("EddyKernels::subtract_square_and_add_to_me");

  if (_spv) { _spcoef.clear(); _spv=false; }
} EddyCatch

void CudaVolume::DivideWithinMask(const CudaVolume& divisor, const CudaVolume& mask) EddyTry
{
  if (divisor!=*this || mask!=*this) throw EddyException("CudaVolume::DivideWithinMask: Dimension mismatch");
  cuda_volume_utils::divide_within_mask(divisor._devec,mask._devec,_devec);
  if (_spv) { _spcoef.clear(); _spv=false; }
} EddyCatch

CudaVolume& CudaVolume::Binarise(float tv) EddyTry
{
  try {
    thrust::transform(_devec.begin(),_devec.end(),_devec.begin(),EDDY::Binarise<float>(tv));
    if (_spv) { _spcoef.clear(); _spv=false; }
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::Binarise_1: with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

CudaVolume& CudaVolume::Binarise(float ll, float ul) EddyTry
{
  try {
    thrust::transform(_devec.begin(),_devec.end(),_devec.begin(),EDDY::Binarise<float>(ll,ul));
    if (_spv) { _spcoef.clear(); _spv=false; }
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::Binarise_2: with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

CudaVolume& CudaVolume::MakeNormRand(float mu, float sigma) EddyTry
{
  try {
    thrust::counting_iterator<unsigned int> index_seq_begin(0);
    thrust::transform(index_seq_begin,index_seq_begin+_devec.size(),_devec.begin(),EDDY::MakeNormRand<float>(mu,sigma));
    if (_spv) { _spcoef.clear(); _spv=false; }
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::MakeRandom: with message: " << e.what() << std::endl;
    throw;
  }
  return(*this);
} EddyCatch

double CudaVolume::Sum(const CudaVolume& mask) const EddyTry
{
  double sum = 0.0;
  if (mask.Size()) {
    if (mask != *this) throw EddyException("CudaVolume::Sum: Mismatched volumes");
    try {
      sum = thrust::inner_product(_devec.begin(),_devec.end(),mask._devec.begin(),sum,
				  thrust::plus<double>(),EDDY::Product<float,double>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume::Sum in call inner_product: with message: " << e.what() << std::endl;
      throw;
    }
  }
  else {
    try {
      sum = thrust::reduce(_devec.begin(),_devec.end(),sum,EDDY::Sum<float,double>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume::Sum in call reduce: with message: " << e.what() << std::endl;
      throw;
    }
  }
  return(sum);
} EddyCatch

double CudaVolume::SumOfSquares(const CudaVolume& mask) const EddyTry
{
  double sos = 0.0;
  if (mask.Size()) {
    if (mask != *this) throw EddyException("CudaVolume::SumOfSquares: Mismatched volumes");
    try {
      sos = thrust::inner_product(_devec.begin(),_devec.end(),mask._devec.begin(),sos,
				  thrust::plus<double>(),EDDY::MaskedSquare<float,double>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume::SumOfSquares in call inner_product: with message: " << e.what() << std::endl;
      throw;
    }
  }
  else {
    try {
      sos = thrust::reduce(_devec.begin(),_devec.end(),sos,EDDY::SumSquare<float,double>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume::SumOfSquares in call reduce: with message: " << e.what() << std::endl;
      throw;
    }
  }
  return(sos);
} EddyCatch
 
CudaVolume& CudaVolume::operator=(float val) EddyTry
{
  try {
    thrust::fill(_devec.begin(), _devec.end(), val);
    if (_spcoef.size()) { thrust::fill(_spcoef.begin(), _spcoef.end(), val); _spv=true; }
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::operator= with message: " << e.what() << std::endl;
    throw;
  }  
  return(*this);
} EddyCatch

unsigned int CudaVolume::Size(unsigned int indx) const EddyTry
{
  if (indx > 2) throw EddyException("CudaVolume::Size: Index out of range");
  return(_sz[indx]);
} EddyCatch

float CudaVolume::Vxs(unsigned int indx) const EddyTry
{
  if (indx > 2) throw EddyException("CudaVolume::Vxs: Index out of range");
  float vxs = (!indx) ? _hdr.xdim() : ((indx==1) ? _hdr.ydim() : _hdr.zdim());
  return(vxs);
} EddyCatch

NEWMAT::Matrix CudaVolume::Ima2WorldMatrix() const EddyTry { return(_hdr.sampling_mat()); } EddyCatch

NEWMAT::Matrix CudaVolume::World2ImaMatrix() const EddyTry { return(_hdr.sampling_mat().i()); } EddyCatch

void CudaVolume::GetVolume(NEWIMAGE::volume<float>& ovol) const EddyTry
{
  thrust::host_vector<float> on_host;
  try {
    on_host = _devec;   
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::GetVolume with message: " << e.what() << std::endl;
    throw;
  }
  ovol.reinitialize(_sz[0],_sz[1],_sz[2]);
  NEWIMAGE::copybasicproperties(_hdr,ovol);  
  unsigned int indx=0;
  for (int k=0; k<ovol.zsize(); k++) { for (int j=0; j<ovol.ysize(); j++) { for (int i=0; i<ovol.xsize(); i++) {
	ovol(i,j,k) = on_host[indx++];                        
  } } }
  return;  
} EddyCatch

void CudaVolume::GetSplineCoefs(NEWIMAGE::volume<float>& ovol) const EddyTry
{
  if (!_spv) throw EddyException("CudaVolume::GetSplineCoefs: Attempt to obtain invalid spline coefficients");

  thrust::host_vector<float> on_host;
  try {
    on_host = _spcoef;   
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::GetSplineCoefs with message: " << e.what() << std::endl;
    throw;
  }
  ovol.reinitialize(_sz[0],_sz[1],_sz[2]);
  NEWIMAGE::copybasicproperties(_hdr,ovol);  
  unsigned int indx=0;
  for (int k=0; k<ovol.zsize(); k++) { for (int j=0; j<ovol.ysize(); j++) { for (int i=0; i<ovol.xsize(); i++) {
	ovol(i,j,k) = on_host[indx++];                        
  } } }
  return;  
} EddyCatch

void CudaVolume::common_assignment_from_newimage_vol(const NEWIMAGE::volume<float>& vol,
						     bool                           ifvol) EddyTry
{
  if (ifvol) { 
    thrust::host_vector<float> hvec(vol.xsize()*vol.ysize()*vol.zsize());
    unsigned int i=0;
    for (NEWIMAGE::volume<float>::fast_const_iterator it=vol.fbegin(); it!=vol.fend(); it++, i++) {
      hvec[i] = *it; 
    }
    try {
      _devec = hvec; 
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume::common_assignment_from_newimage_vol after transfer with message: " << e.what() << std::endl;
      throw;
    }
  }
  else { 
    try {
      _devec.resize(vol.xsize()*vol.ysize()*vol.zsize());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume::common_assignment_from_newimage_vol after resize() with message: " << e.what() << std::endl;
      throw;
    }
  }
  _sz[0] = vol.xsize(); _sz[1] = vol.ysize(); _sz[2] = vol.zsize();
  try {
    _spcoef.clear(); 
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::common_assignment_from_newimage_vol after clear() with message: " << e.what() << std::endl;
    throw;
  }
  _spv = false;
  _hdr.reinitialize(1,1,1);
  NEWIMAGE::copybasicproperties(vol,_hdr);
} EddyCatch

void CudaVolume::calculate_spline_coefs(const std::vector<unsigned int>&     sz,
					const thrust::device_vector<float>&  ima,
					thrust::device_vector<float>&        coef) const EddyTry
{
  if (ima.size() != coef.size()) throw EddyException("CudaVolume::calculate_spline_coefs: Mismatched ima and coef");
  try {
    thrust::copy(ima.begin(),ima.end(),coef.begin());          
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::calculate_spline_coefs after copy() with message: " << e.what() << std::endl;
    throw;
  }
  float *cptr = NULL;
  try {
    cptr = thrust::raw_pointer_cast(coef.data());
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume::calculate_spline_coefs after raw_pointer_cast() with message: " << e.what() << std::endl;
    throw;
  }

  float z = -0.267949192431123f;                         
  unsigned int nburn = ((log(1e-8)/log(abs(z))) + 1.5);  
  std::vector<unsigned int> initn(3); 

  
  for (unsigned int i=0; i<3; i++) initn[i] = (nburn > sz[i]) ? sz[i] : nburn;

  int tpb = threads_per_block_deconv; 

  EddyKernels::ExtrapType ep = EddyKernels::PERIODIC;
  if (Extrap()==NEWIMAGE::extraslice) ep = EddyKernels::CONSTANT;
  for (unsigned int dir=0; dir<3; dir++) {
    int nthreads = 1;
    for (int i=0; i<3; i++) if (i!=dir) nthreads *= sz[i];
    int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
    EddyKernels::cubic_spline_deconvolution<<<nblocks,tpb>>> (cptr,sz[0],sz[1],sz[2],dir,
							      initn[dir],ep,nthreads);
    EddyKernels::CudaSync("EddyKernels::cubic_spline_deconvolution");
  }

  return;
} EddyCatch

CudaVolume3D_2_4D_Helper CudaVolume4D::operator[](unsigned int indx) EddyTry
{
  if (indx >= _sz[3]) throw EddyException("CudaVolume4D::operator[]: indx out of range");
  CudaVolume3D_2_4D_Helper hlp(*this,indx);
  return(hlp);
} EddyCatch


void CudaVolume4D::SetVolume(unsigned int indx, const CudaVolume& vol) EddyTry
{
  if (indx >= _sz[3]) throw EddyException("CudaVolume4D::SetVolume: indx out of range");
  for (unsigned int i=0; i<3; i++) if (_sz[i] != vol._sz[i]) throw EddyException("CudaVolume4D::SetVolume: Mismatched volumes");
  if (!NEWIMAGE::samedim(_hdr,vol._hdr,3)) throw EddyException("CudaVolume4D::SetVolume: Mismatched volumes");
  _devecs[indx] = vol._devec;
} EddyCatch
  
CudaVolume4D& CudaVolume4D::operator+=(const CudaVolume4D& cv) EddyTry
{
  if (*this != cv) throw EddyException("CudaVolume4D::operator+=: Mismatched volumes");
  if (!this->Size()) throw EddyException("CudaVolume4D::operator+=: Empty volume");
  for (unsigned int i=0; i<_devecs.size(); i++) {
    try {
      thrust::transform(_devecs[i].begin(),_devecs[i].end(),cv._devecs[i].begin(),_devecs[i].begin(),thrust::plus<float>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume4D::operator+= with index: " << i << ", and message: " << e.what() << std::endl;
      throw;
    }  
  }
  return(*this);
} EddyCatch

CudaVolume4D& CudaVolume4D::operator*=(const CudaVolume& cv) EddyTry
{
  if (*this != cv) throw EddyException("CudaVolume4D::operator*=: Mismatched volumes");  
  if (!this->Size()) throw EddyException("CudaVolume4D::operator*=: Empty volume");
  for (unsigned int i=0; i<_devecs.size(); i++) {
    try {
      thrust::transform(_devecs[i].begin(),_devecs[i].end(),cv._devec.begin(),_devecs[i].begin(),thrust::multiplies<float>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume4D::operator*= with index: " << i << ", and message: " << e.what() << std::endl;
      throw;
    }  
  }
  return(*this);
} EddyCatch

void CudaVolume4D::DivideWithinMask(const CudaVolume& divisor, const CudaVolume& mask) EddyTry
{
  if (divisor!=*this || mask!=*this) throw EddyException("CudaVolume::DivideWithinMask: Dimension mismatch");
  if (!this->Size()) throw EddyException("CudaVolume4D::DivideWithinMask: Empty volume");
  for (unsigned int i=0; i<_devecs.size(); i++) cuda_volume_utils::divide_within_mask(divisor._devec,mask._devec,_devecs[i]);
} EddyCatch

void CudaVolume4D::Smooth(float fwhm, const CudaVolume& mask) EddyTry
{
  *this *= mask;
  for (unsigned int i=0; i<_devecs.size(); i++) {
    cuda_volume_utils::smooth(fwhm,_sz,_hdr,_devecs[i]);
  }
  CudaVolume smask=mask;
  smask.Smooth(fwhm);
  DivideWithinMask(smask,mask);
  *this *= mask;  
} EddyCatch
 
CudaVolume4D& CudaVolume4D::operator=(float val) EddyTry
{
  for (unsigned int i=0; i<_devecs.size(); i++) {
    try {
      thrust::fill(_devecs[i].begin(), _devecs[i].end(), val);
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume4D::operator= with index: " << i << ", and message: " << e.what() << std::endl;
      throw;
    }  
  }
  return(*this);
} EddyCatch

unsigned int CudaVolume4D::Size(unsigned int indx) const EddyTry
{
  if (indx > 3) throw EddyException("CudaVolume4D::Size: Index out of range");
  return(_sz[indx]);
} EddyCatch

float CudaVolume4D::Vxs(unsigned int indx) const EddyTry
{
  if (indx > 2) throw EddyException("CudaVolume4D::Vxs: Index out of range");
  float vxs = (!indx) ? _hdr.xdim() : ((indx==1) ? _hdr.ydim() : _hdr.zdim());
  return(vxs);
} EddyCatch

void CudaVolume4D::GetVolume(NEWIMAGE::volume4D<float>& ovol) const EddyTry
{
  ovol.reinitialize(_sz[0],_sz[1],_sz[2],_sz[3]);
  NEWIMAGE::copybasicproperties(_hdr,ovol);  
  unsigned int volsize = _sz[0]*_sz[1]*_sz[2];
  NEWIMAGE::volume<float>::nonsafe_fast_iterator it = ovol.nsfbegin();
  for (unsigned int v=0; v<_devecs.size(); v++) {
    thrust::host_vector<float> on_host;
    try {
      on_host = _devecs[v];       
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaVolume4D::GetVolume_1: with message: " << e.what() << std::endl;
      throw;
    }
    for (unsigned int i=0; i<volsize; i++, it++) *it = on_host[i]; 
  }
  return;

  
    
} EddyCatch

void CudaVolume4D::GetVolume(unsigned int indx, NEWIMAGE::volume<float>& ovol) const EddyTry
{
  if (indx >= _sz[3]) throw EddyException("CudaVolume4D::GetVolume(indx,ovol): indx out of range");
  ovol.reinitialize(_sz[0],_sz[1],_sz[2]);
  NEWIMAGE::copybasicproperties(_hdr,ovol);  
  thrust::host_vector<float> on_host;
  try {
    on_host = _devecs[indx];       
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in CudaVolume4D::GetVolume_2: with message: " << e.what() << std::endl;
    throw;
  }
  unsigned int volsize = _sz[0]*_sz[1]*_sz[2];
  NEWIMAGE::volume<float>::nonsafe_fast_iterator it = ovol.nsfbegin();
  for (unsigned int i=0; i<volsize; i++, it++) *it = on_host[i];

  return;
  
  
    
} EddyCatch







void CudaVolume4D::common_assignment_from_newimage_vol(const NEWIMAGE::volume<float>& vol,
						       bool                           ifvol) EddyTry
{
  _devecs.resize(vol.tsize());
  unsigned int volsize = static_cast<unsigned int>(vol.xsize()*vol.ysize()*vol.zsize());
  if (ifvol) { 
    thrust::host_vector<float> hvec(volsize);
    for (unsigned int i=0; i<_devecs.size(); i++) {
      NEWIMAGE::volume<float>::fast_const_iterator it=vol.fbegin(i);
      for (unsigned int j=0; j<volsize; j++, it++) {
	hvec[j] = *it; 
      }
      try {
	_devecs[i] = hvec;
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in CudaVolume4D::common_assignment_from_newimage_vol after transfer with index: " << i << ", with message: " << e.what() << std::endl;
	throw;
      }
    }    
  }
  else { 
    int i;
    for (i=0; i<vol.tsize(); i++) {
      try {
	_devecs[i].resize(volsize);
      }
      catch(thrust::system_error &e) {
	std::cerr << "thrust::system_error thrown in CudaVolume4D::common_assignment_from_newimage_vol after resize() with index: " << i << ", with message: " << e.what() << std::endl;
	throw;
      }
    }
  }
  _sz[0] = vol.xsize(); _sz[1] = vol.ysize(); _sz[2] = vol.zsize(); _sz[3] = vol.tsize();
  _hdr.reinitialize(1,1,1);
  NEWIMAGE::copybasicproperties(vol,_hdr);
} EddyCatch


void cuda_volume_utils::smooth(float                            fwhm, 
			       const std::vector<unsigned int>& sz, 
			       const NEWIMAGE::volume<float>&   hdr, 
			       thrust::device_vector<float>&    ima) EddyTry
{
  
  thrust::device_vector<float> xk = cuda_volume_utils::gaussian_1D_kernel(fwhm/hdr.xdim());
  thrust::device_vector<float> yk = cuda_volume_utils::gaussian_1D_kernel(fwhm/hdr.ydim());
  thrust::device_vector<float> zk = cuda_volume_utils::gaussian_1D_kernel(fwhm/hdr.zdim());
  
  thrust::device_vector<float> sv(sz[0]*sz[1]*sz[2]);
  
  int tpb = threads_per_block_convolve_1D;
  int nthreads = sz[0]*sz[1]*sz[2];
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  
  EddyKernels::convolve_1D<<<nblocks,tpb>>>(sz[0],sz[1],sz[2],thrust::raw_pointer_cast(ima.data()),thrust::raw_pointer_cast(xk.data()),
					    xk.size(),0,nthreads,thrust::raw_pointer_cast(sv.data()));
  EddyKernels::convolve_1D<<<nblocks,tpb>>>(sz[0],sz[1],sz[2],thrust::raw_pointer_cast(sv.data()),thrust::raw_pointer_cast(yk.data()),
					    yk.size(),1,nthreads,thrust::raw_pointer_cast(ima.data()));
  EddyKernels::convolve_1D<<<nblocks,tpb>>>(sz[0],sz[1],sz[2],thrust::raw_pointer_cast(ima.data()),thrust::raw_pointer_cast(zk.data()),
					    zk.size(),2,nthreads,thrust::raw_pointer_cast(sv.data()));
  ima = sv;
} EddyCatch

thrust::host_vector<float> cuda_volume_utils::gaussian_1D_kernel(float fwhm) EddyTry 
{
  float s = fwhm/std::sqrt(8.0*std::log(2.0));
  unsigned int sz = 6*s + 0.5;
  sz = 2*sz+1;
  thrust::host_vector<float> rval(sz);
  double sum=0.0;
  for (unsigned int i=0; i<sz; i++) {
    rval[i] = exp(-sqr(int(i)-int(sz)/2)/(2.0*sqr(s)));
    sum += rval[i];
  }
  for (unsigned int i=0; i<sz; i++) rval[i] /= sum;
  return(rval);
} EddyCatch

void cuda_volume_utils::divide_within_mask(const thrust::device_vector<float>& divisor,
					   const thrust::device_vector<float>& mask,
					   thrust::device_vector<float>&       ima) EddyTry
{
  try {
    thrust::transform_if(ima.begin(),ima.end(),divisor.begin(),mask.begin(),ima.begin(),
			 thrust::divides<float>(),thrust::identity<float>());
  }
  catch(thrust::system_error &e) {
    std::cerr << "thrust::system_error thrown in cuda_volume_utils::divide_within_mask: with message: " << e.what() << std::endl;
    throw;
  }
} EddyCatch


void CudaVolume3D_2_4D_Helper::operator=(const CudaVolume& threed) EddyTry
{
  for (unsigned int i=0; i<3; i++) if (_fourd._sz[i] != threed._sz[i]) throw EddyException("CudaVolume4D::operator=(CudaVolume): Mismatched 3D volume");
  if (!NEWIMAGE::samedim(_fourd._hdr,threed._hdr,3)) throw EddyException("CudaVolume4D::operator=(CudaVolume): Mismatched 3D volume");
  _fourd._devecs[_indx] = threed._devec;
} EddyCatch

void CudaImageCoordinates::Transform(const NEWMAT::Matrix& A) EddyTry
{
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::affine_transform_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,A(1,1),A(1,2),A(1,3),A(1,4),A(2,1),
							     A(2,2),A(2,3),A(2,4),A(3,1),A(3,2),A(3,3),A(3,4),
							     XPtr(),YPtr(),ZPtr(),_init,nthreads);
  EddyKernels::CudaSync("EddyKernels::affine_transform_coordinates");
  _init=true;
  return;
} EddyCatch

void CudaImageCoordinates::Transform(const std::vector<NEWMAT::Matrix>& A) EddyTry
{
  if (A.size() != this->Size(2)) throw EddyException("CudaImageCoordinates::Transform: Mismatched vector of matrices A");
  thrust::device_vector<float> dA = this->repack_vector_of_matrices(A);
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::slice_wise_affine_transform_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,thrust::raw_pointer_cast(dA.data()),
									XPtr(),YPtr(),ZPtr(),_init,nthreads);
  EddyKernels::CudaSync("EddyKernels::slice_wise_affine_transform_coordinates");
  _init=true;
  return;
} EddyCatch

void CudaImageCoordinates::Transform(const NEWMAT::Matrix&            A, 
				     const EDDY::CudaVolume4D&        dfield, 
				     const NEWMAT::Matrix&            B) EddyTry
{
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::general_transform_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,dfield.GetPtr(0),dfield.GetPtr(1),
							      dfield.GetPtr(2),A(1,1),A(1,2),A(1,3),A(1,4),
							      A(2,1),A(2,2),A(2,3),A(2,4),A(3,1),A(3,2),
							      A(3,3),A(3,4),B(1,1),B(1,2),B(1,3),B(1,4),
							      B(2,1),B(2,2),B(2,3),B(2,4),B(3,1),B(3,2),B(3,3),
							      B(3,4),XPtr(),YPtr(),ZPtr(),_init,nthreads);
  EddyKernels::CudaSync("EddyKernels::general_transform_coordinates");
  _init=true;
  return;  
} EddyCatch

void CudaImageCoordinates::Transform(const std::vector<NEWMAT::Matrix>&  A, 
				     const EDDY::CudaVolume4D&           dfield, 
				     const std::vector<NEWMAT::Matrix>&  B) EddyTry
{
  if (A.size() != this->Size(2)) throw EddyException("CudaImageCoordinates::Transform: Mismatched vector of matrices A");
  if (B.size() != this->Size(2)) throw EddyException("CudaImageCoordinates::Transform: Mismatched vector of matrices B");
  thrust::device_vector<float> dA = this->repack_vector_of_matrices(A);
  thrust::device_vector<float> dB = this->repack_vector_of_matrices(B);
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::slice_wise_general_transform_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,dfield.GetPtr(0),dfield.GetPtr(1),
									 dfield.GetPtr(2),thrust::raw_pointer_cast(dA.data()),
									 thrust::raw_pointer_cast(dB.data()),
									 XPtr(),YPtr(),ZPtr(),_init,nthreads);
  EddyKernels::CudaSync("EddyKernels::slice_wise_general_transform_coordinates");
  _init=true;
  return;  
} EddyCatch

void CudaImageCoordinates::GetSliceToVolXYZCoord(const NEWMAT::Matrix&               M1, 
						 const std::vector<NEWMAT::Matrix>&  R, 
						 const EDDY::CudaVolume4D&           dfield, 
						 const NEWMAT::Matrix&               M2,
						 EDDY::CudaVolume&                   zcoord) EddyTry
{
  if (R.size() != this->Size(2)) throw EddyException("CudaImageCoordinates::GetSliceToVolXYZCoord: Mismatched vector of matrices R");
  if (M1(1,2) != 0.0 || M1(1,3) != 0.0 || M1(2,1) != 0.0 || M1(2,3) != 0.0 || M1(3,1) != 0.0 || M1(3,2) != 0.0) {
    EddyException("CudaImageCoordinates::GetSliceToVolXYZCoord: Invalid M1 matrix");
  }
  if (M2(1,2) != 0.0 || M2(1,3) != 0.0 || M2(2,1) != 0.0 || M2(2,3) != 0.0 || M2(3,1) != 0.0 || M2(3,2) != 0.0) {
    EddyException("CudaImageCoordinates::GetSliceToVolXYZCoord: Invalid M2 matrix");
  }
  thrust::device_vector<float> dM1 = this->repack_matrix(M1);
  thrust::device_vector<float> dR = this->repack_vector_of_matrices(R);
  thrust::device_vector<float> dM2 = this->repack_matrix(M2);
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::slice_to_vol_xyz_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,dfield.GetPtr(0),dfield.GetPtr(1),
							     dfield.GetPtr(2),thrust::raw_pointer_cast(dM1.data()),
							     thrust::raw_pointer_cast(dR.data()),thrust::raw_pointer_cast(dM2.data()),
							     XPtr(),YPtr(),ZPtr(),zcoord.GetPtr(),_init,nthreads);
  EddyKernels::CudaSync("EddyKernels::slice_to_vol_xyz_coordinates");
  _init=true;
  return;  
} EddyCatch

void CudaImageCoordinates::GetSliceToVolZCoord(const NEWMAT::Matrix&               M1, 
					       const std::vector<NEWMAT::Matrix>&  R, 
					       const EDDY::CudaVolume4D&           dfield, 
					       const NEWMAT::Matrix&               M2) EddyTry
{
  if (R.size() != this->Size(2)) throw EddyException("CudaImageCoordinates::GetSliceToVolZCoord: Mismatched vector of matrices R");
  if (M1(1,2) != 0.0 || M1(1,3) != 0.0 || M1(2,1) != 0.0 || M1(2,3) != 0.0 || M1(3,1) != 0.0 || M1(3,2) != 0.0) {
    EddyException("CudaImageCoordinates::GetSliceToVolZCoord: Invalid M1 matrix");
  }
  if (M2(1,2) != 0.0 || M2(1,3) != 0.0 || M2(2,1) != 0.0 || M2(2,3) != 0.0 || M2(3,1) != 0.0 || M2(3,2) != 0.0) {
    EddyException("CudaImageCoordinates::GetSliceToVolZCoord: Invalid M2 matrix");
  }
  thrust::device_vector<float> dM1 = this->repack_matrix(M1);
  thrust::device_vector<float> dR = this->repack_vector_of_matrices(R);
  thrust::device_vector<float> dM2 = this->repack_matrix(M2);
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::slice_to_vol_z_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,dfield.GetPtr(0),dfield.GetPtr(1),
							     dfield.GetPtr(2),thrust::raw_pointer_cast(dM1.data()),
							     thrust::raw_pointer_cast(dR.data()),thrust::raw_pointer_cast(dM2.data()),
							     XPtr(),YPtr(),ZPtr(),_init,nthreads);
  EddyKernels::CudaSync("EddyKernels::slice_to_vol_z_coordinates");
  _init=true;
  return;  
} EddyCatch


CudaImageCoordinates& CudaImageCoordinates::operator-=(const CudaImageCoordinates& rhs) EddyTry
{
  if (this->Size() != rhs.Size()) throw EddyException("CudaImageCoordinates::operator-=: Size mismatch.");
  if (!_init) init_coord();
  if (!rhs._init) {
    int tpb = threads_per_block;
    unsigned int nthreads = Size();
    int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
    EddyKernels::implicit_coord_sub<<<nblocks,tpb>>>(_xn,_yn,_zn,XPtr(),YPtr(),ZPtr(),nthreads);
  }
  else {
    try {
      thrust::transform(_x.begin(),_x.end(),rhs._x.begin(),_x.begin(),thrust::minus<float>());
      thrust::transform(_y.begin(),_y.end(),rhs._y.begin(),_y.begin(),thrust::minus<float>());
      thrust::transform(_z.begin(),_z.end(),rhs._z.begin(),_z.begin(),thrust::minus<float>());
    }
    catch(thrust::system_error &e) {
      std::cerr << "thrust::system_error thrown in CudaImageCoordinates::::operator-= with message: " << e.what() << std::endl;
      throw;
    }
  }
  return(*this);
} EddyCatch

NEWMAT::Matrix CudaImageCoordinates::AsMatrix() const EddyTry
{
  NEWMAT::Matrix rval(Size(),3);
  thrust::host_vector<float> x = _x;
  thrust::host_vector<float> y = _y;
  thrust::host_vector<float> z = _z;
  for (unsigned int i=0; i<Size(); i++) {
    rval(i+1,1) = x[i];
    rval(i+1,2) = y[i];
    rval(i+1,3) = z[i];
  }
  return(rval);
} EddyCatch

void CudaImageCoordinates::Write(const std::string& fname, 
				 unsigned int       n) const EddyTry
{
  NEWMAT::Matrix coord = AsMatrix();
  if (n && n<Size()) MISCMATHS::write_ascii_matrix(fname,coord.Rows(1,n));
  else MISCMATHS::write_ascii_matrix(fname,coord);
} EddyCatch

void CudaImageCoordinates::init_coord() EddyTry
{
  int tpb = threads_per_block;
  unsigned int nthreads = Size();
  int nblocks = (nthreads % tpb) ? nthreads / tpb + 1 : nthreads / tpb;
  EddyKernels::make_coordinates<<<nblocks,tpb>>>(_xn,_yn,_zn,XPtr(),YPtr(),ZPtr(),nthreads);
  EddyKernels::CudaSync("EddyKernels::make_coordinates");
  _init = true;
  return;
} EddyCatch

thrust::device_vector<float> CudaImageCoordinates::repack_matrix(const NEWMAT::Matrix& A) EddyTry
{
  thrust::host_vector<float> hA(12);
  hA[0] = A(1,1); hA[1] = A(1,2); hA[2] = A(1,3); hA[3] = A(1,4); 
  hA[4] = A(2,1); hA[5] = A(2,2); hA[6] = A(2,3); hA[7] = A(2,4); 
  hA[8] = A(3,1); hA[9] = A(3,2); hA[10] = A(3,3); hA[11] = A(3,4); 
  return(hA); 
} EddyCatch

thrust::device_vector<float> CudaImageCoordinates::repack_vector_of_matrices(const std::vector<NEWMAT::Matrix>& A) EddyTry
{
  thrust::host_vector<float> hA(12*A.size());
  for (unsigned int i=0; i<A.size(); i++) {
    unsigned int offs = 12*i;
    hA[offs] = A[i](1,1); hA[offs+1] = A[i](1,2); hA[offs+2] = A[i](1,3); hA[offs+3] = A[i](1,4); 
    hA[offs+4] = A[i](2,1); hA[offs+5] = A[i](2,2); hA[offs+6] = A[i](2,3); hA[offs+7] = A[i](2,4); 
    hA[offs+8] = A[i](3,1); hA[offs+9] = A[i](3,2); hA[offs+10] = A[i](3,3); hA[offs+11] = A[i](3,4); 
  }
  return(hA); 
} EddyCatch

