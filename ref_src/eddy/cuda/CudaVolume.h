/////////////////////////////////////////////////////////////////////
///
/// \file CudaVolume.h
/// \brief Declarations of static class with collection of GPU routines used in the eddy project
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///


#ifndef CudaVolume_h
#define CudaVolume_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "EddyKernels.h"

namespace EDDY {

class CudaImageCoordinates;
class CudaVolume4D; 
class CudaVolume3D_2_4D_Helper;
class CudaVolume;

 
class cuda_volume_utils
{
private:
  friend class CudaVolume;
  friend class CudaVolume4D;
  static const int threads_per_block_convolve_1D = 128;
  static float sqr(float a) { return(a*a); }
  static thrust::host_vector<float> gaussian_1D_kernel(float fwhm); 
  static void smooth(float                            fwhm, 
		     const std::vector<unsigned int>& sz, 
		     const NEWIMAGE::volume<float>&   hdr, 
		     thrust::device_vector<float>&    ima);
  static void divide_within_mask(const thrust::device_vector<float>& divisor,
				 const thrust::device_vector<float>& mask,
				 thrust::device_vector<float>&       ima);

};







class CudaVolume
{
public:
  
  CudaVolume() EddyTry : _spv(false), _sz(3,0) {} EddyCatch
  
  CudaVolume(const CudaVolume& cv, bool ifcv=true) EddyTry : _spv(false), _hdr(cv._hdr), _sz(cv._sz) {
    if (ifcv) {_devec=cv._devec; _spcoef=cv._spcoef; _spv=cv._spv; } else _devec.resize(cv.Size());
  } EddyCatch
  
  CudaVolume(const NEWIMAGE::volume<float>& vol, bool ifvol=true) EddyTry : _spv(false), _sz(3,0) {
    common_assignment_from_newimage_vol(vol,ifvol);
  } EddyCatch
  
  void SetHdr(const CudaVolume& cv) EddyTry {
    if (this != &cv) { _spv=false; _sz=cv._sz; _hdr=cv._hdr; _devec.resize(cv.Size()); _spcoef.clear(); }
  } EddyCatch
  
  void SetHdr(const CudaVolume4D& cv); 
  
  void SetHdr(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,false);
  } EddyCatch
  
  CudaVolume& operator=(const CudaVolume& cv) EddyTry {
    if (this != &cv) { _sz=cv._sz; _hdr=cv._hdr; _devec=cv._devec; _spcoef=cv._spcoef; _spv=cv._spv; } return(*this);
  } EddyCatch
  
  CudaVolume& operator=(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,true); return(*this);
  } EddyCatch
  
  void Sample(const EDDY::CudaImageCoordinates& coord, CudaVolume& smpl) const;
  
  void Sample(const EDDY::CudaImageCoordinates& coord, CudaVolume& smpl, CudaVolume4D& dsmpl) const;
  
  void ValidMask(const EDDY::CudaImageCoordinates& coord, CudaVolume& mask) const;
  
  void ResampleStack(const CudaVolume& zcoord, const CudaVolume& inmask, CudaVolume oima) const;
  
  CudaVolume& operator+=(const CudaVolume& rhs);
  
  CudaVolume& operator-=(const CudaVolume& rhs);
  
  CudaVolume& operator*=(const CudaVolume& rhs);
  
  CudaVolume& operator/=(float a);
  
  const CudaVolume operator+(const CudaVolume& rhs) const EddyTry { return(CudaVolume(*this) += rhs); } EddyCatch
  
  const CudaVolume operator-(const CudaVolume& rhs) const EddyTry { return(CudaVolume(*this) -= rhs); } EddyCatch
  
  const CudaVolume operator*(const CudaVolume& rhs) const EddyTry { return(CudaVolume(*this) *= rhs); } EddyCatch
  
  const CudaVolume operator/(float a) const EddyTry { return(CudaVolume(*this) /= a); } EddyCatch
  
  void Smooth(float fwhm) EddyTry { cuda_volume_utils::smooth(fwhm,_sz,_hdr,_devec); if (_spv) { _spcoef.clear(); _spv=false; } } EddyCatch
  
  void Smooth(float fwhm, const CudaVolume& mask);
  
  void MultiplyAndAddToMe(const CudaVolume& pv, float a);
  
  void SubtractMultiplyAndAddToMe(const CudaVolume& pv, const CudaVolume& nv, float a);
  
  void SubtractSquareAndAddToMe(const CudaVolume& pv, const CudaVolume& nv);
  
  void DivideWithinMask(const CudaVolume& divisor, const CudaVolume& mask);
  
  CudaVolume& Binarise(float tv);
  
  CudaVolume& Binarise(float ll, float ul);
  
  CudaVolume& MakeNormRand(float mu, float sigma);
  
  double Sum(const CudaVolume& mask) const;
  
  double Sum() const EddyTry { CudaVolume skrutt; return(Sum(skrutt)); } EddyCatch
  
  double SumOfSquares(const CudaVolume& mask) const;
  
  double SumOfSquares() const EddyTry { CudaVolume skrutt; return(SumOfSquares(skrutt)); } EddyCatch
  
  CudaVolume& operator=(float val);
  
  bool operator==(const CudaVolume& rhs) const EddyTry {
    return(_sz[0]==rhs._sz[0] && _sz[1]==rhs._sz[1] && _sz[2]==rhs._sz[2] &&
	   fabs(_hdr.xdim()-rhs._hdr.xdim())<1e-6 && fabs(_hdr.ydim()-rhs._hdr.ydim())<1e-6 && fabs(_hdr.zdim()-rhs._hdr.zdim())<1e-6);
  } EddyCatch
  
  
  bool operator!=(const CudaVolume& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  bool operator==(const NEWIMAGE::volume<float>& rhs) const EddyTry {
    return(int(_sz[0])==rhs.xsize() && int(_sz[1])==rhs.ysize() && int(_sz[2])==rhs.zsize() &&
	   fabs(_hdr.xdim()-rhs.xdim())<1e-6 && fabs(_hdr.ydim()-rhs.ydim())<1e-6 && fabs(_hdr.zdim()-rhs.zdim())<1e-6);
  } EddyCatch
  
  bool operator!=(const NEWIMAGE::volume<float>& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  bool operator==(const CudaVolume4D& rhs) const;
  
  bool operator!=(const CudaVolume4D& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  friend std::ostream& operator<<(std::ostream& out, const CudaVolume& cv) EddyTry {
    out << "Matrix size: " << cv._sz[0] << ", " << cv._sz[1] << ", " << cv._sz[2] << std::endl;
    out << "Voxel size: " << cv._hdr.xdim() << "mm, " << cv._hdr.ydim() << "mm, " << cv._hdr.zdim() << "mm" << std::endl;
    out << "_devec.size() = " << cv._devec.size() << ", _spv = " << cv._spv << ", _spcoef.size() = " << cv._spcoef.size(); 
    return(out); 
  } EddyCatch
  
  float *GetPtr() EddyTry { _spv=false; return((Size()) ? thrust::raw_pointer_cast(_devec.data()) : 0); } EddyCatch
  
  const float *GetPtr() const EddyTry { return((Size()) ? thrust::raw_pointer_cast(_devec.data()) : 0); } EddyCatch
  
  thrust::device_vector<float>::iterator Begin() { _spv=false; return(_devec.begin()); } 
  
  thrust::device_vector<float>::iterator End() { _spv=false; return(_devec.end()); }
  
  thrust::device_vector<float>::const_iterator Begin() const { return(_devec.begin()); } 
  
  thrust::device_vector<float>::const_iterator End() const { return(_devec.end()); } 
  
  unsigned int Size() const { return(_sz[0]*_sz[1]*_sz[2]); }
  
  unsigned int Size(unsigned int indx) const;
  
  float Vxs(unsigned int indx) const;
  
  NEWMAT::Matrix Ima2WorldMatrix() const; 
  
  NEWMAT::Matrix World2ImaMatrix() const; 
  
  NEWIMAGE::interpolation Interp() const EddyTry { return(_hdr.getinterpolationmethod()); } EddyCatch
  
  NEWIMAGE::extrapolation Extrap() const EddyTry { return(_hdr.getextrapolationmethod()); } EddyCatch
  
  std::vector<bool> ExtrapValid() const EddyTry { return(_hdr.getextrapolationvalidity()); } EddyCatch
  
  void SetInterp(NEWIMAGE::interpolation im) EddyTry { _hdr.setinterpolationmethod(im); } EddyCatch
  
  void SetExtrap(NEWIMAGE::extrapolation im) EddyTry { _hdr.setextrapolationmethod(im); } EddyCatch
  
  void GetVolume(NEWIMAGE::volume<float>& ovol) const;
  
  NEWIMAGE::volume<float> GetVolume() const EddyTry { NEWIMAGE::volume<float> ovol; GetVolume(ovol); return(ovol); } EddyCatch
  
  void Write(const std::string& fname) const EddyTry { NEWIMAGE::write_volume(GetVolume(),fname); } EddyCatch
  
  void GetSplineCoefs(NEWIMAGE::volume<float>& ovol) const;
  
  NEWIMAGE::volume<float> GetSplineCoefs() const EddyTry { NEWIMAGE::volume<float> ovol; GetSplineCoefs(ovol); return(ovol); } EddyCatch
  
  void WriteSplineCoefs(const std::string& fname) const EddyTry { NEWIMAGE::write_volume(GetSplineCoefs(),fname); } EddyCatch
  friend class CudaVolume4D;              
  friend class CudaVolume3D_2_4D_Helper;  
private:
  static const int                       threads_per_block_interpolate = 128;
  static const int                       threads_per_block_deconv = 128;
  static const int                       threads_per_block_smaatm = 128;
  static const int                       threads_per_block_ssaatm = 128;

  thrust::device_vector<float>           _devec;
  mutable thrust::device_vector<float>   _spcoef;     
  mutable bool                           _spv;        
  NEWIMAGE::volume<float>                _hdr;
  std::vector<unsigned int>              _sz;

  void common_assignment_from_newimage_vol(const NEWIMAGE::volume<float>& vol,
					   bool                           ifvol);
  const float *sp_ptr() const EddyTry { return(thrust::raw_pointer_cast(_spcoef.data())); } EddyCatch
  void calculate_spline_coefs(const std::vector<unsigned int>&     sz,
			      const thrust::device_vector<float>&  ima,
			      thrust::device_vector<float>&        coef) const;
  thrust::host_vector<float> gaussian_1D_kernel(float fwhm) const;
  void smooth(float                            fwhm, 
	      const std::vector<unsigned int>& sz, 
	      const NEWIMAGE::volume<float>&   hdr, 
	      thrust::device_vector<float>&    ima);
  void divide_within_mask(const thrust::device_vector<float>& divisor,
			  const thrust::device_vector<float>& mask,
			  thrust::device_vector<float>&       ima);
};







class CudaVolume4D
{
public:
  CudaVolume4D() EddyTry : _sz(4,0) {} EddyCatch
  CudaVolume4D(const CudaVolume4D& cv, bool ifcv=true) EddyTry : _sz(cv._sz), _hdr(cv._hdr), _devecs(cv._devecs.size()) {
    if (ifcv) _devecs = cv._devecs; 
    else { for (int i=0; i<_devecs.size(); i++) _devecs[i].resize(cv._devecs[i].size()); }
  } EddyCatch
  CudaVolume4D(const CudaVolume& cv, unsigned int nv, bool ifcv=true) EddyTry : _sz(4,0), _hdr(cv._hdr), _devecs(nv) {
    _sz[0]=cv._sz[0]; _sz[1]=cv._sz[1]; _sz[2]=cv._sz[2]; _sz[3]=nv;
    if (ifcv) { for (int i=0; i<_devecs.size(); i++) _devecs[i] = cv._devec; }
    else { for (int i=0; i<_devecs.size(); i++) _devecs[i].resize(cv._devec.size()); }
  } EddyCatch
  CudaVolume4D(const NEWIMAGE::volume<float>& vol, bool ifvol=true) EddyTry : _sz(4,0) {
    common_assignment_from_newimage_vol(vol,ifvol);
  } EddyCatch
  
  
  
  void SetHdr(const CudaVolume4D& cv) EddyTry {
    if (this != &cv) { 
      _sz=cv._sz; _hdr=cv._hdr; _devecs.resize(cv._devecs.size()); 
      for (int i=0; i<_devecs.size(); i++) _devecs[i].resize(cv._devecs[i].size()); 
    }
  } EddyCatch
  
  void SetHdr(const CudaVolume& cv, unsigned int nv) EddyTry {
    _sz[0]=cv._sz[0]; _sz[0]=cv._sz[1]; _sz[2]=cv._sz[0]; _sz[3]=nv; _hdr=cv._hdr; _devecs.resize(nv);
    for (int i=0; i<_devecs.size(); i++) _devecs[i].resize(cv._devec.size()); 
  } EddyCatch
  
  void SetHdr(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,false);
  } EddyCatch
  
  
  
  CudaVolume4D& operator=(const CudaVolume4D& cv) EddyTry {
    if (this != &cv) { _sz=cv._sz; _hdr=cv._hdr; _devecs=cv._devecs; } return(*this);
  } EddyCatch
  
  CudaVolume4D& operator=(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,true); return(*this);
  } EddyCatch
  
  
  
  CudaVolume3D_2_4D_Helper operator[](unsigned int indx);
  
  void SetVolume(unsigned int i, const CudaVolume& vol);
  
  CudaVolume4D& operator+=(const CudaVolume4D& cv);
  
  CudaVolume4D& operator*=(const CudaVolume& cv);
  
  void DivideWithinMask(const CudaVolume& divisor, const CudaVolume& mask);
  
  void Smooth(float fwhm) EddyTry { for (unsigned int i=0; i<_devecs.size(); i++) cuda_volume_utils::smooth(fwhm,_sz,_hdr,_devecs[i]); } EddyCatch
  
  void Smooth(float fwhm, const CudaVolume& mask);
    
  CudaVolume4D& operator=(float val);
  
  bool operator==(const CudaVolume4D& rhs) const EddyTry {
    return(this->_sz[0]==rhs._sz[0] && this->_sz[1]==rhs._sz[1] && this->_sz[2]==rhs._sz[2] && this->_sz[3]==rhs._sz[3] &&
	   fabs(this->_hdr.xdim()-rhs._hdr.xdim())<1e-6 && fabs(this->_hdr.ydim()-rhs._hdr.ydim())<1e-6 && fabs(this->_hdr.zdim()-rhs._hdr.zdim())<1e-6);
  } EddyCatch
  
  bool operator!=(const CudaVolume4D& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  bool operator==(const CudaVolume& rhs) const EddyTry {
    return(this->_sz[0]==rhs.Size(0) && this->_sz[1]==rhs.Size(1) && this->_sz[2]==rhs.Size(2) &&
	   fabs(this->_hdr.xdim()-rhs.Vxs(0))<1e-6 && fabs(this->_hdr.ydim()-rhs.Vxs(1))<1e-6 && fabs(this->_hdr.zdim()-rhs.Vxs(2))<1e-6);
  } EddyCatch
  
  bool operator!=(const CudaVolume& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  bool operator==(const NEWIMAGE::volume<float>& rhs) const EddyTry {
    return(int(_sz[0])==rhs.xsize() && int(_sz[1])==rhs.ysize() && int(_sz[2])==rhs.zsize() &&
	   fabs(_hdr.xdim()-rhs.xdim())<1e-6 && fabs(_hdr.ydim()-rhs.ydim())<1e-6 && fabs(_hdr.zdim()-rhs.zdim())<1e-6);
  } EddyCatch
  
  bool operator!=(const NEWIMAGE::volume<float>& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  float *GetPtr(unsigned int i) EddyTry { 
    if (i>=_devecs.size()) throw EddyException("CudaVolume4D::GetPtr: index out of range");
    return(thrust::raw_pointer_cast(_devecs[i].data())); 
  } EddyCatch
  
  const float *GetPtr(unsigned int i) const EddyTry { 
    if (i>=_devecs.size()) throw EddyException("CudaVolume4D::GetPtr: index out of range");
    return(thrust::raw_pointer_cast(_devecs[i].data())); 
  } EddyCatch
  
  thrust::device_vector<float>::iterator Begin(unsigned int i) EddyTry { 
    if (i>=_devecs.size()) throw EddyException("CudaVolume4D::Begin: index out of range");
    return(_devecs[i].begin()); 
  } EddyCatch
  
  thrust::device_vector<float>::iterator End(unsigned int i) EddyTry { 
    if (i>=_devecs.size()) throw EddyException("CudaVolume4D::End: index out of range");
    return(_devecs[i].end()); 
  } EddyCatch
  
  thrust::device_vector<float>::const_iterator Begin(unsigned int i) const EddyTry { 
    if (i>=_devecs.size()) throw EddyException("CudaVolume4D::Begin:const: index out of range");
    return(_devecs[i].begin()); 
  } EddyCatch
  
  thrust::device_vector<float>::const_iterator End(unsigned int i) const EddyTry { 
    if (i>=_devecs.size()) throw EddyException("CudaVolume4D::End:const: index out of range");
    return(_devecs[i].end()); 
  } EddyCatch
  
  unsigned int Size() const { return(_sz[0]*_sz[1]*_sz[2]); }
  
  unsigned int Size(unsigned int indx) const;
  
  float Vxs(unsigned int indx) const;
  NEWIMAGE::interpolation Interp() const EddyTry { return(_hdr.getinterpolationmethod()); } EddyCatch
  NEWIMAGE::extrapolation Extrap() const EddyTry { return(_hdr.getextrapolationmethod()); } EddyCatch
  std::vector<bool> ExtrapValid() const EddyTry { return(_hdr.getextrapolationvalidity()); } EddyCatch
  
  void SetInterp(NEWIMAGE::interpolation im) EddyTry { _hdr.setinterpolationmethod(im); } EddyCatch
  
  NEWIMAGE::volume4D<float> GetVolume() const EddyTry { NEWIMAGE::volume4D<float> ovol; GetVolume(ovol); return(ovol); } EddyCatch
  
  void GetVolume(NEWIMAGE::volume4D<float>& ovol) const;
  
  NEWIMAGE::volume<float> GetVolume(unsigned int indx) const EddyTry { NEWIMAGE::volume<float> ovol; GetVolume(indx,ovol); return(ovol); } EddyCatch
  
  void GetVolume(unsigned int indx, NEWIMAGE::volume<float>& ovol) const;
  
  void Write(const std::string& fname) const EddyTry { NEWIMAGE::write_volume4D(GetVolume(),fname); } EddyCatch
  
  void Write(unsigned int indx, const std::string& fname) const EddyTry { NEWIMAGE::write_volume(GetVolume(indx),fname); } EddyCatch
  friend class CudaVolume;                
  friend class CudaVolume3D_2_4D_Helper;  
private:
  std::vector<unsigned int>                    _sz;
  NEWIMAGE::volume<float>                      _hdr;
  std::vector<thrust::device_vector<float> >   _devecs;

  void common_assignment_from_newimage_vol(const NEWIMAGE::volume<float>& vol,
					   bool                           ifvol);
  
  
};

 
class CudaVolume3D_2_4D_Helper
{
public:
  void operator=(const CudaVolume& threed);
  friend class CudaVolume4D; 
private:
  CudaVolume3D_2_4D_Helper(CudaVolume4D& fourd, unsigned int indx) EddyTry : _fourd(fourd), _indx(indx) {} EddyCatch 
  CudaVolume4D& _fourd;
  unsigned int _indx;
};

 
class CudaImageCoordinates
{
public:
  CudaImageCoordinates() EddyTry : _xn(0), _yn(0), _zn(0), _init(false) {} EddyCatch
  CudaImageCoordinates(unsigned int xn, unsigned int yn, unsigned int zn, bool init=false) EddyTry 
    : _xn(xn), _yn(yn), _zn(zn), _x(xn*yn*zn), _y(xn*yn*zn), _z(xn*yn*zn), _init(init) { if (init) init_coord(); } EddyCatch
  void Resize(unsigned int xn, unsigned int yn, unsigned int zn, bool init=false) EddyTry {
    _xn=xn; _yn=yn; _zn=zn;
    _x.resize(xn*yn*zn); _y.resize(xn*yn*zn); _y.resize(xn*yn*zn); _init=false;
    if (init) init_coord();
  } EddyCatch
  
  void Transform(const NEWMAT::Matrix& A);
  
  void Transform(const std::vector<NEWMAT::Matrix>& A);
  
  void Transform(const NEWMAT::Matrix& A, const EDDY::CudaVolume4D& dfield, const NEWMAT::Matrix& B);
  
  void Transform(const std::vector<NEWMAT::Matrix>& A, const EDDY::CudaVolume4D& dfield, const std::vector<NEWMAT::Matrix>& B);
  
  void GetSliceToVolXYZCoord(const NEWMAT::Matrix& M1, const std::vector<NEWMAT::Matrix>& R, const EDDY::CudaVolume4D& dfield, const NEWMAT::Matrix& M2, EDDY::CudaVolume& zcoord); 
  
  void GetSliceToVolZCoord(const NEWMAT::Matrix& M1, const std::vector<NEWMAT::Matrix>& R, const EDDY::CudaVolume4D& dfield, const NEWMAT::Matrix& M2); 
  unsigned int Size() const { return(_xn*_yn*_zn); }
  unsigned int Size(unsigned int indx) const EddyTry {
    if (indx>2) throw EddyException("CudaImageCoordinates::Size: Index out of range.");
    return((!indx) ? _xn : ((indx==1) ? _yn : _zn)); 
  } EddyCatch
  CudaImageCoordinates& operator-=(const CudaImageCoordinates& rhs);
  const float *XPtr() const EddyTry { return(thrust::raw_pointer_cast(_x.data())); } EddyCatch
  const float *YPtr() const EddyTry { return(thrust::raw_pointer_cast(_y.data())); } EddyCatch
  const float *ZPtr() const EddyTry { return(thrust::raw_pointer_cast(_z.data())); } EddyCatch
  
  NEWMAT::Matrix AsMatrix() const;
  
  void Write(const std::string& fname, unsigned int n=0) const;
private:
  float *XPtr() EddyTry { return(thrust::raw_pointer_cast(_x.data())); } EddyCatch
  float *YPtr() EddyTry { return(thrust::raw_pointer_cast(_y.data())); } EddyCatch
  float *ZPtr() EddyTry { return(thrust::raw_pointer_cast(_z.data())); } EddyCatch
  static const int             threads_per_block = 128;

  unsigned int                 _xn;
  unsigned int                 _yn;
  unsigned int                 _zn;
  thrust::device_vector<float> _x;
  thrust::device_vector<float> _y;
  thrust::device_vector<float> _z;
  bool                         _init;

  void init_coord();
  thrust::device_vector<float> repack_matrix(const NEWMAT::Matrix& A);
  thrust::device_vector<float> repack_vector_of_matrices(const std::vector<NEWMAT::Matrix>& A);
};


} 

#ifdef I_CUDAVOLUME_H_DEFINED_ET
#undef I_CUDAVOLUME_H_DEFINED_ET
#undef EXPOSE_TREACHEROUS   
#endif

#endif 

