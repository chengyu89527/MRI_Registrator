//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Utility functions for CUDA code
/// \details Mostly a number of functions for converting between linear and volumetric
///          indices.
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CBF_KERNEL_HELPERS_CUH
#define CBF_KERNEL_HELPERS_CUH

#include <vector>
#include <cuda.h>

namespace CBF
{
    
    __host__ __device__ void index_vol_to_lin(
                                
                                
                                unsigned int xind, unsigned int yind, unsigned int zind,
                                
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                
                                
                                unsigned int *lind);

    
    __host__ __device__ void index_lin_to_vol(
                                
                                
                                unsigned int lind,
                                
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                 
                                unsigned int *xind, unsigned int *yind, unsigned int *zind);

    
    __host__ __device__ void index_lin_to_lin(
                                
                                
                                unsigned int lind_in,
                                
                                unsigned int lv_szx, unsigned int lv_szy, unsigned int lv_szz,
                                
                                unsigned int vl_szx, unsigned int vl_szy, unsigned int vl_szz,
                                
                                unsigned int *lind_out);

    
    __host__ __device__ void identify_diagonal(
                                
                                
                                unsigned int diag_ind,
                                
                                unsigned int rep_x, unsigned int rep_y, unsigned int rep_z,
                                
                                unsigned int spl_x, unsigned int spl_y, unsigned int spl_z,
                                
                                unsigned int *first_row, unsigned int *last_row,
                                unsigned int *first_column, unsigned int *last_column);
    
    
    __host__ std::vector<unsigned int> get_spl_coef_dim(const std::vector<unsigned int>& ksp,
                                                        const std::vector<unsigned int>& isz);
} 
#endif 

