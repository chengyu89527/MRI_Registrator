/*! \file LSResampler.cu
    \brief Contains definition of CUDA implementation of a class for least-squares resampling of pairs of images

    \author Jesper Andersson
    \version 1.0b, August, 2013.
*/
//
// LSResampler.cu
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford 
//


#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
#include <cuda.h>
#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#pragma diag_suppress = expr_has_no_effect  
#include "newimage/newimageall.h"
#pragma pop
#include "topup/topup_file_io.h"
#include "topup/displacement_vector.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "ECScanClasses.h"
#include "LSResampler.h"
#include "EddyKernels.h"
#include "EddyMatrixKernels.h"
#include "CudaVolume.h"
#include "EddyInternalGpuUtils.h"
#include "EddyGpuUtils.h"

namespace EDDY {

class LSResamplerImpl
{
public:
  LSResamplerImpl(const EDDY::ECScan&                             s1, 
		  const EDDY::ECScan&                             s2,
		  std::shared_ptr<const NEWIMAGE::volume<float> > hzfield,
		  double                                          lambda);
  const NEWIMAGE::volume<float>& GetResampledVolume() const EddyTry { return(_rvol); } EddyCatch
  const NEWIMAGE::volume<float>& GetMask() const EddyTry { return(_omask); } EddyCatch
private:
  NEWIMAGE::volume<float>  _rvol;  
  NEWIMAGE::volume<float>  _omask; 
  
  static const int         _threads_per_block_QR = 128;
  static const int         _threads_per_block_Solve = 128;
  static const int         _threads_per_block_Kty = 128;
  static const dim3        _threads_per_block_KtK;

  template<typename T>
  T sqr(const T& v) const { return(v*v); }

  unsigned int make_k_matrices(
			       const NEWIMAGE::volume4D<float>& field1,  
			       const NEWIMAGE::volume4D<float>& field2,  
			       const NEWIMAGE::volume<float>&   mask,    
			       unsigned int                     sl,      
			       bool                             pex,     
			       
			       thrust::device_vector<float>&    kmats,   
			       std::vector<bool>&               isok);   

  void solve_for_y_hat(
		       const thrust::device_vector<float>& ktkmats, 
		       const thrust::device_vector<float>& ktyvecs, 
		       unsigned int                        n,       
		       unsigned int                        nmat,    
		       bool                                sync,    
		       
		       thrust::device_vector<float>&       yhat);   

  void make_kty_vectors(
			const thrust::device_vector<float>& kmats, 
			const thrust::device_vector<float>& yvecs, 
			unsigned int                        m,     
			unsigned int                        n,     
			unsigned int                        nmat,  
			bool                                sync,  
			
			thrust::device_vector<float>&       kty);  

  void make_ktk_matrices(
			 const thrust::device_vector<float>& kmats, 
			 unsigned int                        m,     
			 unsigned int                        n,     
			 unsigned int                        nmat,  
			 const thrust::device_vector<float>& sts,   
			 bool                                sync,  
			 
			 thrust::device_vector<float>&       ktk);  

  void transfer_y_hat_vectors(
			      const thrust::device_vector<float>& yhatvecs, 
			      const std::vector<bool>&            isok,     
			      unsigned int                        sl,       
			      bool                                pex,      
			      
			      NEWIMAGE::volume<float>&            ima,      
			      NEWIMAGE::volume<float>&            mask);    

  void transfer_y_vectors(
			  const NEWIMAGE::volume<float>& ima1,   
			  const NEWIMAGE::volume<float>& ima2,   
			  const std::vector<bool>&       isok,   
			  unsigned int                   sl,     
			  bool                           pex,    
			  
			  thrust::device_vector<float>&  yvecs); 

  void copy_row_first_to_matrix(const float      *rf_fptr,
				NEWMAT::Matrix&  M);

  void copy_to_matrix(const float      *fptr,  
		      NEWMAT::Matrix&  M);     

  void copy_matrix(const NEWMAT::Matrix& M,      
		   float                 *fptr); 

  void get_sts(unsigned int                  sz,      
	       double                        lambda,  
	       thrust::device_vector<float>& dStS);   

  void dump_matrices(const std::string&                  fname,      
		     const thrust::device_vector<float>& mats,       
		     unsigned int                        m,          
		     unsigned int                        n,          
		     unsigned int                        nmat,       
		     bool                                row_first); 
};
const dim3 LSResamplerImpl::_threads_per_block_KtK = dim3(16,16);

LSResampler::LSResampler(const EDDY::ECScan&                             s1, 
			 const EDDY::ECScan&                             s2,
			 std::shared_ptr<const NEWIMAGE::volume<float> > hzfield,
			 double                                          lambda) EddyTry
{
  _pimpl = new LSResamplerImpl(s1,s2,hzfield,lambda);
} EddyCatch

LSResampler::~LSResampler() EddyTry
{
  delete _pimpl;
} EddyCatch

const NEWIMAGE::volume<float>& LSResampler::GetResampledVolume() const EddyTry
{
  return(_pimpl->GetResampledVolume());
} EddyCatch

const NEWIMAGE::volume<float>& LSResampler::GetMask() const EddyTry
{
  return(_pimpl->GetMask());
} EddyCatch

 
LSResamplerImpl::LSResamplerImpl(const EDDY::ECScan&                             s1, 
				 const EDDY::ECScan&                             s2,
				 std::shared_ptr<const NEWIMAGE::volume<float> > hzfield,
				 double                                          lambda) EddyTry
{
  if (!EddyUtils::AreMatchingPair(s1,s2)) throw EddyException("LSResampler::LSResampler:: Mismatched pair");
  EddyGpuUtils::InitGpu();
  
  NEWIMAGE::volume<float> ima1;
  NEWIMAGE::volume<float> mask;
  EddyGpuUtils::GetMotionCorrectedScan(s1,false,ima1,&mask);
  NEWIMAGE::volume<float> ima2;
  NEWIMAGE::volume<float> mask2;
  EddyGpuUtils::GetMotionCorrectedScan(s2,false,ima2,&mask2);
  mask *= mask2;

  _omask.reinitialize(mask.xsize(),mask.ysize(),mask.zsize());
  _omask = 1.0;
  _rvol.reinitialize(ima1.xsize(),ima1.ysize(),ima1.zsize());
  NEWIMAGE::copybasicproperties(ima1,_rvol);
  
  NEWIMAGE::volume4D<float> field1 = s1.FieldForScanToModelTransform(hzfield); 
  NEWIMAGE::volume4D<float> field2 = s2.FieldForScanToModelTransform(hzfield); 

  
  bool pex = false;
  unsigned int matsz = field1[0].ysize();
  if (s1.GetAcqPara().PhaseEncodeVector()(1)) { pex = true; matsz = field1[0].xsize(); }
  unsigned int nK = matsz; unsigned int mK = 2*nK;
  unsigned int nmat_per_sl = (pex) ? ima1.ysize() : ima1.xsize();

  
  thrust::device_vector<float> gpu_K_matrices(nmat_per_sl*2*sqr(matsz),0.0);
  thrust::device_vector<float> gpu_KtK_matrices(nmat_per_sl*sqr(matsz),0.0);
  thrust::device_vector<float> gpu_StS(sqr(matsz),0.0);
  thrust::device_vector<float> gpu_y_vectors(nmat_per_sl*2*matsz,0.0);
  thrust::device_vector<float> gpu_Kty_vectors(nmat_per_sl*matsz,0.0);
  thrust::device_vector<float> gpu_solution_vectors(nmat_per_sl*matsz,0.0);
  
  std::vector<bool> isok(nmat_per_sl);
  
  get_sts(matsz,lambda,gpu_StS);

  for (int k=0; k<ima1.zsize(); k++) {
    
    unsigned int nmat = make_k_matrices(field1,field2,mask,k,pex,gpu_K_matrices,isok);
    
    transfer_y_vectors(ima1,ima2,isok,k,pex,gpu_y_vectors);
    
    make_ktk_matrices(gpu_K_matrices,mK,nK,nmat,gpu_StS,true,gpu_KtK_matrices);
    
    make_kty_vectors(gpu_K_matrices,gpu_y_vectors,mK,nK,nmat,true,gpu_Kty_vectors);
    
    solve_for_y_hat(gpu_KtK_matrices,gpu_Kty_vectors,nK,nmat,true,gpu_solution_vectors);
    
    transfer_y_hat_vectors(gpu_solution_vectors,isok,k,pex,_rvol,_omask);
  }  
  
  return;
} EddyCatch

unsigned int LSResamplerImpl::make_k_matrices(
					      const NEWIMAGE::volume4D<float>& field1,  
					      const NEWIMAGE::volume4D<float>& field2,  
					      const NEWIMAGE::volume<float>&   mask,    
					      unsigned int                     sl,      
					      bool                             pex,     
					      
					      thrust::device_vector<float>&    kmats,   
					      std::vector<bool>&               isok) EddyTry 
{
  unsigned int sz = (pex) ? field1.xsize() : field1.ysize();
  unsigned int nmat_per_sl = (pex) ? field1.ysize() : field1.xsize();
  unsigned int matsize = 2*sz*sz;
  TOPUP::DispVec dv1(sz), dv2(sz);
  thrust::host_vector<float> host_kmats(nmat_per_sl*matsize,0.0);

  double sf1, sf2; 
  if (pex) { sf1 = 1.0/field1.xdim(); sf2 = 1.0/field2.xdim(); }
  else { sf1 = 1.0/field1.ydim(); sf2 = 1.0/field2.ydim(); }

  float *kptr = thrust::raw_pointer_cast(host_kmats.data());
  unsigned int nvalid = 0; 
  for (int i=0; i<nmat_per_sl; i++) {
    bool row_col_is_ok = true;
    if (pex) {
      if (!dv1.RowIsAlright(mask,sl,i)) row_col_is_ok = false;
      else {
	dv1.SetFromRow(field1[0],sl,i);
	dv2.SetFromRow(field2[0],sl,i);
      }
    }
    else {
      if (!dv1.ColumnIsAlright(mask,sl,i)) row_col_is_ok = false;
      else {
	dv1.SetFromColumn(field1[1],sl,i);
	dv2.SetFromColumn(field2[1],sl,i);
      }
    }
    if (row_col_is_ok) {
      isok[i] = true;
      NEWMAT::Matrix K = dv1.GetK_Matrix(sf1) & dv2.GetK_Matrix(sf2);
      copy_matrix(K,kptr);
      kptr+=matsize;
      nvalid++;
    }
    else isok[i] = false;
  }
  kmats = host_kmats; 
  return(nvalid);
} EddyCatch

void LSResamplerImpl::solve_for_y_hat(
				      const thrust::device_vector<float>& ktkmats, 
				      const thrust::device_vector<float>& ktyvecs, 
				      unsigned int                        n,       
				      unsigned int                        nmat,    
				      bool                                sync,    
				      
				      thrust::device_vector<float>&       yhat) EddyTry 
{
  if (nmat) {
    
    thrust::device_vector<float> Qt(nmat*n*n,0.0);
    thrust::device_vector<float> R(nmat*n*n,0.0);
    
    size_t sh_mem_sz = 2*n*sizeof(float);
    int tpb = _threads_per_block_QR;
    EddyMatrixKernels::QR<<<nmat,tpb,sh_mem_sz>>>(thrust::raw_pointer_cast(ktkmats.data()),n,n,nmat,
						  thrust::raw_pointer_cast(Qt.data()),
						  thrust::raw_pointer_cast(R.data()));
    if (sync) EddyKernels::CudaSync("QR_Kernels::QR");
    tpb = _threads_per_block_Solve;
    EddyMatrixKernels::Solve<<<nmat,tpb>>>(thrust::raw_pointer_cast(Qt.data()),
					   thrust::raw_pointer_cast(R.data()),
					   thrust::raw_pointer_cast(ktyvecs.data()),n,n,nmat,
					   thrust::raw_pointer_cast(yhat.data()));
    if (sync) EddyKernels::CudaSync("QR_Kernels::Solve");
  }
  else {
    thrust::fill(yhat.begin(),yhat.end(),static_cast<float>(0.0));
  }
  return;
} EddyCatch

void LSResamplerImpl::dump_matrices(const std::string&                  fname,      
				    const thrust::device_vector<float>& mats,       
				    unsigned int                        m,          
				    unsigned int                        n,          
				    unsigned int                        nmat,       
				    bool                                row_first) EddyTry 
{
  thrust::host_vector<float> h_mats = mats; 
  NEWMAT::Matrix big_m(m,n);
  NEWMAT::Matrix little_m(m,n);
  if (row_first) copy_row_first_to_matrix(thrust::raw_pointer_cast(h_mats.data()),big_m);
  else copy_to_matrix(thrust::raw_pointer_cast(h_mats.data()),big_m);
  for (unsigned int mat=1; mat<nmat; mat++) {
    if (row_first) copy_row_first_to_matrix(thrust::raw_pointer_cast(h_mats.data())+mat*m*n,little_m);
    else copy_to_matrix(thrust::raw_pointer_cast(h_mats.data())+mat*m*n,little_m);
    big_m &= little_m;
  }
  MISCMATHS::write_ascii_matrix(fname,big_m);
} EddyCatch

void LSResamplerImpl::make_kty_vectors(
				       const thrust::device_vector<float>& kmats, 
				       const thrust::device_vector<float>& yvecs, 
				       unsigned int                        m,     
				       unsigned int                        n,     
				       unsigned int                        nmat,  
				       bool                                sync,  
				       
				       thrust::device_vector<float>&       kty) EddyTry 
{
  if (nmat) {
    int tpb = _threads_per_block_Kty;
    EddyMatrixKernels::Kty<<<nmat,tpb>>>(thrust::raw_pointer_cast(kmats.data()),
					 thrust::raw_pointer_cast(yvecs.data()),m,n,nmat,
					 thrust::raw_pointer_cast(kty.data()));
    if (sync) EddyKernels::CudaSync("KtK_Kernels::Kty");
  }
  return;				 
} EddyCatch

void LSResamplerImpl::make_ktk_matrices(
					const thrust::device_vector<float>& kmats, 
					unsigned int                        m,     
					unsigned int                        n,     
					unsigned int                        nmat,  
					const thrust::device_vector<float>& sts,   
					bool                                sync,  
					
					thrust::device_vector<float>&       ktk) EddyTry 
{
  if (nmat) {
    dim3 block = _threads_per_block_KtK; 
    EddyMatrixKernels::KtK<<<nmat,block>>>(thrust::raw_pointer_cast(kmats.data()),m,n,nmat,
					   thrust::raw_pointer_cast(sts.data()),1.0,false,
					   thrust::raw_pointer_cast(ktk.data()));
    if (sync) EddyKernels::CudaSync("KtK_Kernels::KtK");
  }
  return;
} EddyCatch

void LSResamplerImpl::transfer_y_hat_vectors(
					     const thrust::device_vector<float>& yhatvecs, 
					     const std::vector<bool>&            isok,     
					     unsigned int                        sl,       
					     bool                                pex,      
					     
					     NEWIMAGE::volume<float>&            ima,      
					     NEWIMAGE::volume<float>&            mask) EddyTry 
{
  thrust::host_vector<float> host_yhatvecs = yhatvecs; 
  unsigned int vecsize = (pex) ? ima.xsize() : ima.ysize();
  unsigned int nmat = (pex) ? ima.ysize() : ima.xsize();
  NEWMAT::ColumnVector yhat(vecsize);
  NEWMAT::ColumnVector zeros(vecsize); zeros=0.0;
  float *yhatptr = thrust::raw_pointer_cast(host_yhatvecs.data());
  for (unsigned int i=0; i<nmat; i++) {
    if (isok[i]) {
      copy_to_matrix(yhatptr,yhat);
      if (pex) ima.SetRow(i,sl,yhat); else ima.SetColumn(i,sl,yhat);
      yhatptr+=vecsize; 
    }
    else {
      if (pex) { ima.SetRow(i,sl,zeros); mask.SetRow(i,sl,zeros); }
      else { ima.SetColumn(i,sl,zeros); mask.SetColumn(i,sl,zeros); }
    }
  }
} EddyCatch

void LSResamplerImpl::transfer_y_vectors(
					 const NEWIMAGE::volume<float>& ima1,   
					 const NEWIMAGE::volume<float>& ima2,   
					 const std::vector<bool>&       isok,   
					 unsigned int                   sl,     
					 bool                           pex,    
					 
					 thrust::device_vector<float>&  yvecs) EddyTry 
{
  unsigned int nvecs = (pex) ? ima1.ysize() : ima1.xsize();
  unsigned int vecsize = (pex) ? ima1.xsize() : ima1.ysize();
  thrust::host_vector<float> hyvecs(2*nvecs*vecsize,0.0);
  float *vecptr = thrust::raw_pointer_cast(hyvecs.data());
  NEWMAT::ColumnVector y(2*vecsize);
  for (int ij=0; ij<nvecs; ij++) {
    if (isok[ij]) {
      if (pex) y = ima1.ExtractRow(ij,sl) & ima2.ExtractRow(ij,sl);
      else y = ima1.ExtractColumn(ij,sl) & ima2.ExtractColumn(ij,sl);
      copy_matrix(y,vecptr);
      vecptr+=2*vecsize;
    }
  }
  yvecs = hyvecs; 
  return;
} EddyCatch

void LSResamplerImpl::copy_row_first_to_matrix(const float      *rf_fptr,
					       NEWMAT::Matrix&  M) EddyTry
{
  for (unsigned int r=0; r<M.Nrows(); r++) {
    for (unsigned int c=0; c<M.Ncols(); c++) {
      M(r+1,c+1) = rf_fptr[c*M.Nrows()+r];
    }
  }
} EddyCatch

void LSResamplerImpl::copy_to_matrix(const float      *fptr,     
				     NEWMAT::Matrix&  M) EddyTry 
{
  for (unsigned int r=0; r<M.Nrows(); r++) {
    for (unsigned int c=0; c<M.Ncols(); c++) {
      M(r+1,c+1) = fptr[r*M.Ncols()+c];
    }
  }
  return;
  
  
} EddyCatch

void LSResamplerImpl::copy_matrix(const NEWMAT::Matrix& M,             
				  float                 *fptr) EddyTry 
{
  for (unsigned int r=0; r<M.Nrows(); r++) {
    for (unsigned int c=0; c<M.Ncols(); c++) {
      fptr[r*M.Ncols()+c] = static_cast<float>(M(r+1,c+1)); 
    }
  }
  return;
  
  
} EddyCatch

void LSResamplerImpl::get_sts(unsigned int                  sz,
			      double                        lambda,
			      thrust::device_vector<float>& dStS) EddyTry
{
  TOPUP::DispVec dv(sz);
  NEWMAT::Matrix StS = dv.GetS_Matrix(false);
  StS = lambda*(StS.t()*StS);
  thrust::host_vector<float> hStS(sqr(sz),0.0);
  copy_matrix(StS,thrust::raw_pointer_cast(hStS.data()));
  dStS = hStS;
} EddyCatch

} 


