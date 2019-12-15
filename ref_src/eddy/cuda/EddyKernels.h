/////////////////////////////////////////////////////////////////////
///
/// \file EddyKernels.h
/// \brief Declarations of 
///
/// \author Jesper Andersson
/// \version 1.0b, Feb., 2013.
/// \Copyright (C) 2012 University of Oxford 
///


#ifndef EddyKernels_h
#define EddyKernels_h

#include <string>
#include <cuda.h>

namespace EddyKernels {

#define MAX_IMA_SIZE 512
enum ExtrapType { CONSTANT = 0, PERIODIC = 1, MIRROR = 2};


void CudaSync(std::string msg);


__global__ void linear_ec_field(float *ec_field, 
				int xsz, int ysz, int zsz, 
				float xvxs, float yvxs, float zvxs,
				const float *ep, int npar,
				int max_id);

__global__ void quadratic_ec_field(float *ec_field, 
				   int xsz, int ysz, int zsz, 
				   float xvxs, float yvxs, float zvxs,
				   const float *ep, int npar,
				   int max_id);

__global__ void cubic_ec_field(float *ec_field, 
			       int xsz, int ysz, int zsz, 
			       float xvxs, float yvxs, float zvxs,
			       const float *ep, int npar,
			       int max_id);

__global__ void make_coordinates(int xsz, int ysz, int zsz, 
				 float *xcoord, float *ycoord, float *zcoord, 
				 int max_id);

__global__ void affine_transform_coordinates(int   xsz, int   ysz, int   zsz,
					     float A11, float A12, float A13, float A14,
					     float A21, float A22, float A23, float A24,
					     float A31, float A32, float A33, float A34,
					     float *xcoord, float *ycoord, float *zcoord, 
					     bool tec, int   max_id);

__global__ void slice_wise_affine_transform_coordinates(int   xsz, int   ysz, int   zsz, const float *A,
							float *xcoord, float *ycoord, float *zcoord, 
							bool tec, int   max_id);

__global__ void general_transform_coordinates(int   xsz, int   ysz, int   zsz,
					      const float *xfield, const float *yfield, const float *zfield, 
					      float A11, float A12, float A13, float A14,
					      float A21, float A22, float A23, float A24,
					      float A31, float A32, float A33, float A34,
					      float M11, float M12, float M13, float M14, 
					      float M21, float M22, float M23, float M24, 
					      float M31, float M32, float M33, float M34, 
					      float *xcoord, float *ycoord, float *zcoord, 
					      bool tec, int   max_id);

__global__ void slice_wise_general_transform_coordinates(int   xsz, int   ysz, int   zsz,
							 const float *xfield, const float *yfield, 
							 const float *zfield, const float *A,
							 const float *M, float *xcoord, 
							 float *ycoord, float *zcoord, 
							 bool tec, int   max_id);

__global__ void slice_to_vol_xyz_coordinates(int   xsz, int   ysz, int   zsz,
					     const float *xfield, const float *yfield, 
					     const float *zfield, const float *M1,
					     const float *R, const float *M2, 
					     float *xcoord, float *ycoord, float *zcoord, 
					     float *zvolume, bool tec, int max_id);

__global__ void slice_to_vol_z_coordinates(int   xsz, int   ysz, int   zsz,
					   const float *xfield, const float *yfield, 
					   const float *zfield, const float *M1,
					   const float *R, const float *M2, 
					   float *xcoord, float *ycoord, float *zcoord, 
					   bool tec, int max_id);

__global__ void get_mask(int xsz, int ysz, int zsz,
			 int epvx, int epvy, int epvz, 
			 const float *xcoord, const float *ycoord, const float *zcoord, 
			 float *mask, 
			 int max_id);

__global__ void make_deriv(int xsz, int ysz, int zsz,
			   const float *xcoord, const float *ycoord, const float *zcoord, 
			   const float *xgrad, const float *ygrad, const float *zgrad,
			   const float *base, const float *jac, const float *basejac,
			   float dstep,
			   float *deriv, 
			   int max_id);

__global__ void spline_interpolate(int         xsz,
				   int         ysz,
				   int         zsz,
				   const float *spcoef,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *ima);

__global__ void spline_interpolate(int   xsz,
				   int   ysz,
				   int   zsz,
				   const float *inima,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *oima,
				   float       *xd,
				   float       *yd,
				   float       *zd);

__global__ void linear_interpolate(int   xsz,
				   int   ysz,
				   int   zsz,
				   const float *inima,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *oima);

__global__ void linear_interpolate(int   xsz,
				   int   ysz,
				   int   zsz,
				   const float *inima,
				   const float *xcoord, 
				   const float *ycoord, 
				   const float *zcoord, 
				   int         sizeT,
				   int         bc,
				   float       *oima,
				   float       *xd,
				   float       *yd,
				   float       *zd);


__global__ void cubic_spline_deconvolution(float        *data, 
					   unsigned int xsize, 
					   unsigned int ysize,
					   unsigned int zsize,
					   unsigned int dir,
					   unsigned int initn,
					   int          bc,
					   int          max_id);

__global__ void convolve_1D(
			    unsigned int xsz,
			    unsigned int ysz,
			    unsigned int zsz,
			    const float  *ima,
			    const float  *krnl,
			    unsigned int krnsz,
			    unsigned int dir,
			    int          max_id,
			    
			    float        *cima);

__global__ void invert_displacement_field(const float  *dfield, 
					  const float  *inmask,
					  unsigned int xsize, 
					  unsigned int ysize,
					  unsigned int zsize,
					  unsigned int dir,
					  float        *idfield,
					  float        *omask,
					  int          max_id);

__global__ void valid_voxels(unsigned int xsize,
			     unsigned int ysize,
			     unsigned int zsize,
			     bool         xval,
			     bool         yval,
			     bool         zval,
			     const float  *x,
			     const float  *y,
			     const float  *z,
			     int          max_id,
			     float        *mask);

__global__ void implicit_coord_sub(unsigned int xs,
				   unsigned int ys,
				   unsigned int zs,
				   float        *x,
				   float        *y,
				   float        *z,
				   int          max_id);

__global__ void subtract_multiply_and_add_to_me(const float  *pv,
						const float  *nv,
						float        a,
						int          max_id,
						float        *out);

__global__ void subtract_square_and_add_to_me(const float  *pv,
					      const float  *nv,
					      int          max_id,
					      float        *out);

__global__ void make_deriv_first_part(int xsz, int ysz, int zsz,
				      const float *xcoord, const float *ycoord, const float *zcoord, 
				      const float *xgrad, const float *ygrad, const float *zgrad,
				      const float *base, const float *jac, const float *basejac,
				      float dstep,
				      float *deriv, 
				      int max_id);

__global__ void make_deriv_second_part(int xsz, int ysz, int zsz,
				       const float *xcoord, const float *ycoord, const float *zcoord, 
				       const float *xgrad, const float *ygrad, const float *zgrad,
				       const float *base, const float *jac, const float *basejac,
				       float dstep,
				       float *deriv, 
				       int max_id);

__global__ void make_mask_from_stack(const float   *inmask,
				     const float   *zcoord,
				     unsigned int  xsz,
				     unsigned int  ysz,
				     unsigned int  zsz,
				     float         *omask);

__global__ void transfer_y_hat_to_volume(const float   *yhat,
					 unsigned int  xsz,
					 unsigned int  ysz,
					 unsigned int  zsz,
					 unsigned int  y,
					 float         *vol);

__global__ void TransferAndCheckSorting(const float  *origz,
					unsigned int xsz,
					unsigned int ysz,
					unsigned int zsz,
					float        *sortz,
					unsigned int *flags);

__global__ void TransferVolumeToVectors(const float  *orig,
					unsigned int xsz,
					unsigned int ysz,
					unsigned int zsz,
					float        *trgt);

__global__ void SortVectors(const unsigned int  *indx,
			    unsigned int        nindx,
			    unsigned int        zsz,
			    float               *key,
			    float               *vec2);

__global__ void LinearInterpolate(const float    *zcoord,
				  const float    *val,
				  unsigned int   zsz,
				  float          *ival);

__global__ void TransferColumnsToVolume(const float    *zcols,
					unsigned int   xsz,
					unsigned int   ysz,
					unsigned int   zsz,
					float          *vol);

__global__ void MakeWeights(const float  *zcoord,
			    unsigned int xsz,
			    unsigned int zsz,
			    unsigned int j,
			    float        *weight);

__global__ void InsertWeights(const float  *wvec,
			      unsigned int j,
			      unsigned int xsz,
			      unsigned int ysz,
			      unsigned int zsz,
			      float        *wvol);

__global__ void MakeDiagwpVecs(const float *pred,
			       const float *wgts,
			       unsigned int xsz,
			       unsigned int ysz,
			       unsigned int zsz,
			       unsigned int j,
			       float        *diagwp);

} 





#endif 


