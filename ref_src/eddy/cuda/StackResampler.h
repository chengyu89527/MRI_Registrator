/*! \file StackResampler.h
    \brief Contains declaration of CUDA implementation of a class for spline resampling of irregularly sampled columns in the z-direction

    \author Jesper Andersson
    \version 1.0b, March, 2016.
*/
//
// StackResampler.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford 
//

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
#include "newimage/newimage.h"
#include "EddyHelperClasses.h"
#include "CudaVolume.h"
#pragma pop

namespace EDDY {

class StackResampler
{
public:
  

  
  StackResampler(const EDDY::CudaVolume&  stack,
		 const EDDY::CudaVolume&  zcoord,
		 const EDDY::CudaVolume&  pred,
		 const EDDY::CudaVolume&  mask,
		 double                   lambda=0.005);

  
  StackResampler(const EDDY::CudaVolume&  stack,
		 const EDDY::CudaVolume&  zcoord,
		 const EDDY::CudaVolume&  mask,
		 NEWIMAGE::interpolation  interp=NEWIMAGE::spline,
		 double                   lambda=0.005);
  ~StackResampler() EddyTry {} EddyCatch
  
  const EDDY::CudaVolume& GetResampledIma() const EddyTry { return(_resvol); } EddyCatch
  NEWIMAGE::volume<float> GetResampledImaAsNEWIMAGE() const EddyTry { return(_resvol.GetVolume()); } EddyCatch
  
  const EDDY::CudaVolume& GetMask() const EddyTry { return(_mask); } EddyCatch
private:
  static const int         _threads_per_block_QR = 128;
  static const int         _threads_per_block_Solve = 128;
  static const int         _threads_per_block_Wirty = 128;
  static const int         _threads_per_block_Wir = 128;
  static const int         _threads_per_block_yhat = 128;
  static const int         _threads_per_block_transfer = 128;
  static const dim3        _threads_per_block_WtW_StS;
  EDDY::CudaVolume         _resvol;
  EDDY::CudaVolume         _mask;

  
  void get_StS(unsigned int sz, double lambda, thrust::device_vector<float>& StS) const;
  
  void get_regular_W(unsigned int sz, thrust::device_vector<float>& W) const;
  
  void make_mask(const EDDY::CudaVolume&   inmask,
		 const EDDY::CudaVolume&   zcoord,
		 bool                      zync,
		 EDDY::CudaVolume&         omask);
  
  void sort_zcoords(const EDDY::CudaVolume&        zcoord,
		    bool                           sync,
		    thrust::device_vector<float>&  szcoord) const;
  
  void make_weight_vectors(const thrust::device_vector<float>&  zcoord,
			   unsigned int                         xsz,
			   unsigned int                         zsz,
			   unsigned int                         xzp,
			   bool                                 sync,
			   thrust::device_vector<float>&        weights) const;
  
  void insert_weights(const thrust::device_vector<float>&  wvec,
		      unsigned int                         j,
		      bool                                 sync,
		      EDDY::CudaVolume&                    wvol) const;
  
  void make_diagw_p_vectors(const EDDY::CudaVolume&               pred,
			    const thrust::device_vector<float>&   wgts,
			    unsigned int                          xzp,
			    bool                                  sync,
			    thrust::device_vector<float>&         wp) const;
  
  void make_diagw_W_matrices(const thrust::device_vector<float>&   w,
			     const thrust::device_vector<float>&   W,
			     unsigned int                          matsz,
			     unsigned int                          nmat,
			     bool                                  sync,
			     thrust::device_vector<float>&         diagwW) const;
  
  void make_dwWt_dwp_vectors(const thrust::device_vector<float>& dW,
			     const thrust::device_vector<float>& dp,
			     unsigned int                        matsz,
			     unsigned int                        nmat,
			     bool                                sync,
			     thrust::device_vector<float>&       dWtdp) const;
  
  void make_Wir_matrices(const EDDY::CudaVolume&       zcoord, 
			 unsigned int                  xzp,
			 bool                          sync,
			 thrust::device_vector<float>& Wir) const;
  
  void make_Wir_t_y_vectors(const EDDY::CudaVolume&                 y,
			    const thrust::device_vector<float>&     Wir,
			    unsigned int                            xzp,
			    bool                                    sync,
			    thrust::device_vector<float>&           Wirty) const;
  
  void make_WtW_StS_matrices(const thrust::device_vector<float>&  Wir,
			     unsigned int                         mn,
			     unsigned int                         nmat,
			     const thrust::device_vector<float>&  StS,      
			     bool                                 sync,
			     thrust::device_vector<float>&        WtW) const;
  
  void solve_for_c_hat(
		       const thrust::device_vector<float>& WtW,         
		       const thrust::device_vector<float>& Wty,         
		       unsigned int                        n,           
		       unsigned int                        nmat,        
		       bool                                sync,        
		       
		       thrust::device_vector<float>&       chat) const; 
  
  void make_y_hat_vectors(
			  const thrust::device_vector<float>& W,
			  const thrust::device_vector<float>& chat,
			  unsigned int                        mn,
			  unsigned int                        nvec,
			  bool                                sync,
			  
			  thrust::device_vector<float>&       yhat) const;
  
  void transfer_y_hat_to_volume(
				const thrust::device_vector<float>& yhat,
				unsigned int                        xzp,
				bool                                sync,
				
				EDDY::CudaVolume&                   ovol) const;
  
  void sort_zcoords_and_intensities(const EDDY::CudaVolume&        zcoord,
				    const EDDY::CudaVolume&        data,
				    bool                           sync,
				    thrust::device_vector<float>&  szcoord,
				    thrust::device_vector<float>&  sdata) const;
  
  void linear_interpolate_columns(const thrust::device_vector<float>&  zcoord,
				  const thrust::device_vector<float>&  val,
				  unsigned int                         xsz,
				  unsigned int                         ysz,
				  unsigned int                         zsz,
				  bool                                 sync,
				  thrust::device_vector<float>&        ival) const;
  void transfer_interpolated_columns_to_volume(const thrust::device_vector<float>&  zcols,
					       bool                                 sync,
					       EDDY::CudaVolume&                    vol);
  
  unsigned int rfindx(unsigned int i, unsigned int j, unsigned int mn) const { return(i+j*mn); }
  
  unsigned int cfindx(unsigned int i, unsigned int j, unsigned int mn) const { return(i*mn+j); }
  
  template <typename T> T sqr(T a) const { return(a*a); }
  
  void write_matrix(const thrust::device_vector<float>& mats,
		    unsigned int                        offs,
		    unsigned int                        m,
		    unsigned int                        n,
		    const std::string&                  fname) const;
  
  void write_matrices(const thrust::device_vector<float>& mats,
		      unsigned int                        nmat,
		      unsigned int                        m,
		      unsigned int                        n,
		      const std::string&                  basefname) const;
  
  void write_debug_info_for_pred_resampling(unsigned int                         x,
					    unsigned int                         y,
					    const std::string&                   bfname,
					    const EDDY::CudaVolume&              z,
					    const EDDY::CudaVolume&              g,
					    const EDDY::CudaVolume&              p,
					    const thrust::device_vector<float>&  sz,
					    const thrust::device_vector<float>&  W,
					    const thrust::device_vector<float>&  Wir,
					    const thrust::device_vector<float>&  w,
					    const thrust::device_vector<float>&  wp,
					    const thrust::device_vector<float>&  wW,
					    const thrust::device_vector<float>&  Wirtg,
					    const thrust::device_vector<float>&  wWtwp,
					    const thrust::device_vector<float>&  WirtWir,
					    const thrust::device_vector<float>&  wWtwW,
					    const thrust::device_vector<float>&  sum_vec,
					    const thrust::device_vector<float>&  sum_mat,
					    const thrust::device_vector<float>&  c_hat,
					    const thrust::device_vector<float>&  y_hat) const;

};

} 

