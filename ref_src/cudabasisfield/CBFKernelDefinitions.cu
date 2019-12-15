//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Implementation of cuda-specific code
/// \details Specifically CBFKernelHelpers and CBFKernels functions
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#include "CBFKernelHelpers.cuh"
#include "CBFKernels.cuh"

#include <cuda.h>

#include <vector>


namespace CBF
{
    
    __host__ __device__ void index_vol_to_lin(
                                
                                
                                unsigned int xind, unsigned int yind, unsigned int zind,
                                
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                
                                
                                unsigned int *lind)
    {
        *lind = zind*(szx*szy) + yind*(szx) + xind;
    }

    
    __host__ __device__ void index_lin_to_vol(
                                
                                
                                unsigned int lind,
                                
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                 
                                unsigned int *xind, unsigned int *yind, unsigned int *zind)
    {
        *zind = lind/(szx*szy);
        *yind = (lind - *zind*(szx*szy))/szx;
        *xind = lind - *zind*(szx*szy) - *yind*(szx);
    }

    
    __host__ __device__ void index_lin_to_lin(
                                
                                
                                unsigned int lind_in,
                                
                                unsigned int lv_szx, unsigned int lv_szy, unsigned int lv_szz,
                                
                                unsigned int vl_szx, unsigned int vl_szy, unsigned int vl_szz,
                                
                                unsigned int *lind_out)
    {
        
        unsigned int xind = 0;
        unsigned int yind = 0;
        unsigned int zind = 0;

        
        index_lin_to_vol(lind_in,lv_szx,lv_szy,lv_szz,&xind,&yind,&zind);
        index_vol_to_lin(xind,yind,zind,vl_szx,vl_szy,vl_szz,lind_out);
    }

    
    __host__ __device__ void identify_diagonal(
                                
                                
                                unsigned int diag_ind,
                                
                                unsigned int rep_x, unsigned int rep_y, unsigned int rep_z,
                                
                                unsigned int spl_x, unsigned int spl_y, unsigned int spl_z,
                                
                                unsigned int *first_row, unsigned int *last_row,
                                unsigned int *first_column, unsigned int *last_column)
    {
        
        unsigned int hess_side_length = spl_x*spl_y*spl_z;

        
        unsigned int main_diag_lind = (rep_x*rep_y*rep_z-1)/2;
        unsigned int hess_main_diag_lind = 0;
        unsigned int hess_lind = 0;
        
        index_lin_to_lin(main_diag_lind,rep_x,rep_y,rep_z,spl_x,spl_y,spl_z,&hess_main_diag_lind);
        index_lin_to_lin(diag_ind,rep_x,rep_y,rep_z,spl_x,spl_y,spl_z,&hess_lind);
        
        if (diag_ind < main_diag_lind)
        {
            *first_row = hess_main_diag_lind - hess_lind;
            *last_row = hess_side_length - 1;
            *first_column = 0;
            *last_column = hess_side_length - *first_row - 1;
        }

        
        else if (diag_ind >= main_diag_lind)
        {
            *first_column = hess_lind - hess_main_diag_lind;
            *last_column = hess_side_length - 1;
            *first_row = 0;
            *last_row = hess_side_length - *first_column - 1;
        }
    }

    
    
    __host__ std::vector<unsigned int> get_spl_coef_dim(const std::vector<unsigned int>& ksp,
                                                        const std::vector<unsigned int>& isz)
    {
      std::vector<unsigned int> rval(ksp.size());
      for (unsigned int i=0; i<ksp.size(); i++)
      {
          rval[i] = static_cast<unsigned int>(std::ceil(float(isz[i]+1) / float(ksp[i]))) + 2;
      }
      return(rval);
    }
} 




namespace CBF
{
  __device__ void calculate_diagonal_range(
      
      const int offset,
      const unsigned int n_rows,
      const unsigned int n_cols,
      
      unsigned int *first_row,
      unsigned int *last_row,
      unsigned int *first_col,
      unsigned int *last_col)
  {
    
    if (offset < 0){
      *first_row = -offset;
      *last_row = n_rows -1;
      *first_col = 0;
      *last_col = n_cols + offset -1;
    }
    
    else{
      *first_row = 0;
      *last_row = n_rows - offset -1;
      *first_col = offset;
      *last_col = n_cols -1;
    }
  }

  __device__ void calculate_overlap(
      
      const int diff_mid,
      const unsigned int ksp,
      const int spl_order,
      
      unsigned int *spl1_start,
      unsigned int *spl1_end,
      unsigned int *spl2_start)
  {
    
    if (diff_mid < 0){
      *spl1_start = static_cast<unsigned int>(-diff_mid)*ksp;
      *spl1_end = (ksp * (spl_order + 1)) - 2;
      *spl2_start = 0;
    }
    
    else{
      *spl1_start = 0;
      *spl1_end = (ksp * (spl_order + 1)) - 2 - (diff_mid * ksp);
      *spl2_start = diff_mid * ksp;
    }
  } 

  __device__ bool is_valid_index(
      const int index,
      const unsigned int n_vals)
  {
    if (index < 0) return false;
    else if (static_cast<unsigned int>(index) >= n_vals) return false;
    else return true;
  } 
} 




namespace CBF
{
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  __global__ void kernel_make_jtj_symmetrical(
      
      unsigned int ima_sz_x,
      unsigned int ima_sz_y,
      unsigned int ima_sz_z,
      cudaTextureObject_t ima,
      const float* __restrict__ spl_x,
      const float* __restrict__ spl_y,
      const float* __restrict__ spl_z,
      unsigned int spl_ksp_x,
      unsigned int spl_ksp_y,
      unsigned int spl_ksp_z,
      unsigned int param_sz_x,
      unsigned int param_sz_y,
      unsigned int param_sz_z,
      const int* __restrict__ jtj_offsets,
      
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; 
    const auto jtj_sz_diag = param_sz_x * param_sz_y * param_sz_z;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x = all_splines;
    float *shared_spline_y = &all_splines[spl_ksp_x * (spl_order + 1) - 1];
    float *shared_spline_z = &shared_spline_y[spl_ksp_y * (spl_order + 1) - 1];
    
    
    
    __shared__ unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    __shared__ unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    __shared__ unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;

    
    
    
    
    
    if (threadIdx.x == 0){
      
      auto this_offset = jtj_offsets[blockIdx.x];
      CBF::calculate_diagonal_range(
          
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      
      
      
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      CBF::index_lin_to_vol(
          
          diag_first_row,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      CBF::index_lin_to_vol(
          
          diag_first_col,
          param_sz_x,
          param_sz_y,
          param_sz_z,
          
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
      
      
      auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
      auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
      auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
      CBF::calculate_overlap(
          
          spl_diff_mid_x,
          spl_ksp_x,
          spl_order,
          
          &spl1_xind_start,
          &spl1_xind_end,
          &spl2_xind_start);
      CBF::calculate_overlap(
          
          spl_diff_mid_y,
          spl_ksp_y,
          spl_order,
          
          &spl1_yind_start,
          &spl1_yind_end,
          &spl2_yind_start);
      CBF::calculate_overlap(
          
          spl_diff_mid_z,
          spl_ksp_z,
          spl_order,
          
          &spl1_zind_start,
          &spl1_zind_end,
          &spl2_zind_start);
    }
    
    __syncthreads();
    
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x) - 1; ++i){
        shared_spline_x[i*blockDim.x + threadIdx.x] = spl_x[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y) - 1; ++i){
        shared_spline_y[i*blockDim.x + threadIdx.x] = spl_y[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z) - 1; ++i){
        shared_spline_z[i*blockDim.x + threadIdx.x] = spl_z[i*blockDim.x + threadIdx.x];
    }
    
    __syncthreads();
    
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - diag_first_row;

    
    if (!is_valid_index(this_row, jtj_sz_diag)) return;
    else if (!is_valid_index(this_col, jtj_sz_diag)) return;
    
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    CBF::index_lin_to_vol(
        
        this_row,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    CBF::index_lin_to_vol(
        
        this_col,
        param_sz_x,
        param_sz_y,
        param_sz_z,
        
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    
    
    
    
    
    
    if (!CBF::is_valid_index(spl_diff_mid_x, spl_order + 1)) return;
    else if (!CBF::is_valid_index(spl_diff_mid_y, spl_order + 1)) return;
    else if (!CBF::is_valid_index(spl_diff_mid_z, spl_order + 1)) return;
    
    
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x) 
      - static_cast<int>(spl_order*spl_ksp_x - 1) 
      + static_cast<int>(spl1_xind_start); 
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y) 
      - static_cast<int>(spl_order*spl_ksp_y - 1) 
      + static_cast<int>(spl1_yind_start); 
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z) 
      - static_cast<int>(spl_order*spl_ksp_z - 1) 
      + static_cast<int>(spl1_zind_start); 
    
    
    
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + static_cast<int>(k);
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + static_cast<int>(j);
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + static_cast<int>(i);
          
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          CBF::index_vol_to_lin(
              
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              
              &vol_lind);
          
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x[spl1_xind]
              * shared_spline_y[spl1_yind]
              * shared_spline_z[spl1_zind]
              * shared_spline_x[spl2_xind]
              * shared_spline_y[spl2_yind]
              * shared_spline_z[spl2_zind];
        }
      }
    }
    
    unsigned int symm_1, symm_2, symm_3;
    CBF::index_lin_to_vol(
        
        blockIdx.x,
        2*spl_order + 1,
        2*spl_order + 1,
        2*spl_order + 1,
        
        &symm_1,
        &symm_2,
        &symm_3);
    int previous_diag_idx = blockIdx.x;
    int previous_row = this_row;
    
    for (int k = 0; k <= 1; ++k){
      unsigned int symm_k;
      
      if (symm_3 == spl_order) ++k;
      if (k == 0) symm_k = symm_3;
      else symm_k = 2*spl_order - symm_3;
      for (int j = 0; j <= 1; ++j){
        unsigned int symm_j;
        
        if (symm_2 == spl_order) ++j;
        if (j == 0) symm_j = symm_2;
        else symm_j = 2*spl_order - symm_2;
        for (int i = 0; i <= 1; ++i){
          unsigned int symm_i;
          
          if (symm_1 == spl_order) ++i;
          if (i == 0) symm_i = symm_1;
          else symm_i = 2*spl_order - symm_1;
          unsigned int inner_diag_idx;
          CBF::index_vol_to_lin(
              
              symm_i,
              symm_j,
              symm_k,
              2*spl_order + 1,
              2*spl_order + 1,
              2*spl_order + 1,
              
              &inner_diag_idx);
          int diag_diff = jtj_offsets[inner_diag_idx] - jtj_offsets[previous_diag_idx];
          int inner_row = previous_row - diag_diff/2;
          previous_diag_idx = inner_diag_idx;
          previous_row = inner_row;
          
          jtj_values[inner_diag_idx*jtj_sz_diag + inner_row] = jtj_val;
        }
      }
    }
  } 

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  __global__ void kernel_make_jtj_non_symmetrical(
      
      unsigned int ima_sz_x,
      unsigned int ima_sz_y,
      unsigned int ima_sz_z,
      cudaTextureObject_t ima,
      const float* __restrict__ spl_x_1,
      const float* __restrict__ spl_y_1,
      const float* __restrict__ spl_z_1,
      const float* __restrict__ spl_x_2,
      const float* __restrict__ spl_y_2,
      const float* __restrict__ spl_z_2,
      unsigned int spl_ksp_x_1,
      unsigned int spl_ksp_y_1,
      unsigned int spl_ksp_z_1,
      unsigned int spl_ksp_x_2,
      unsigned int spl_ksp_y_2,
      unsigned int spl_ksp_z_2,
      unsigned int param_sz_x_1,
      unsigned int param_sz_y_1,
      unsigned int param_sz_z_1,
      unsigned int param_sz_x_2,
      unsigned int param_sz_y_2,
      unsigned int param_sz_z_2,
      const int* __restrict__ jtj_offsets,
      
      float* __restrict__ jtj_values)
  {
    __shared__ unsigned int diag_first_row;
    __shared__ unsigned int diag_last_row;
    __shared__ unsigned int diag_first_col;
    __shared__ unsigned int diag_last_col;
    extern __shared__ float all_splines[];
    const int spl_order = 3; 
    const auto jtj_sz_diag = param_sz_x_1 * param_sz_y_1 * param_sz_z_1;
    const auto jtj_sz_row = jtj_sz_diag;
    const auto jtj_sz_col = param_sz_x_2 * param_sz_y_2 * param_sz_z_2;
    const auto offset_into_diag = blockIdx.y*blockDim.x + threadIdx.x;
    float *shared_spline_x_1 = all_splines;
    float *shared_spline_y_1 = &shared_spline_x_1[spl_ksp_x_1 * (spl_order + 1) - 1];
    float *shared_spline_z_1 = &shared_spline_y_1[spl_ksp_y_1 * (spl_order + 1) - 1];
    float *shared_spline_x_2 = &shared_spline_z_1[spl_ksp_z_1 * (spl_order + 1) - 1];
    float *shared_spline_y_2 = &shared_spline_x_2[spl_ksp_x_2 * (spl_order + 1) - 1];
    float *shared_spline_z_2 = &shared_spline_y_2[spl_ksp_y_2 * (spl_order + 1) - 1];

    
    
    
    
    
    if (threadIdx.x == 0){
      
      auto this_offset = jtj_offsets[blockIdx.x];
      CBF::calculate_diagonal_range(
          
          this_offset,
          jtj_sz_diag,
          jtj_sz_diag,
          
          &diag_first_row,
          &diag_last_row,
          &diag_first_col,
          &diag_last_col);
      
      
      
      unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
      CBF::index_lin_to_vol(
          
          diag_first_row,
          param_sz_x_1,
          param_sz_y_1,
          param_sz_z_1,
          
          &spl1_mid_x,
          &spl1_mid_y,
          &spl1_mid_z);
      CBF::index_lin_to_vol(
          
          diag_first_col,
          param_sz_x_2,
          param_sz_y_2,
          param_sz_z_2,
          
          &spl2_mid_x,
          &spl2_mid_y,
          &spl2_mid_z);
    }
    
    __syncthreads();
    
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_1) - 1; ++i){
        shared_spline_x_1[i*blockDim.x + threadIdx.x] = spl_x_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_1) - 1; ++i){
        shared_spline_y_1[i*blockDim.x + threadIdx.x] = spl_y_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_1) - 1; ++i){
        shared_spline_z_1[i*blockDim.x + threadIdx.x] = spl_z_1[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_x_2) - 1; ++i){
        shared_spline_x_2[i*blockDim.x + threadIdx.x] = spl_x_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_y_2) - 1; ++i){
        shared_spline_y_2[i*blockDim.x + threadIdx.x] = spl_y_2[i*blockDim.x + threadIdx.x];
    }
    for (int i = 0; i*blockDim.x + threadIdx.x < ((spl_order + 1) * spl_ksp_z_2) - 1; ++i){
        shared_spline_z_2[i*blockDim.x + threadIdx.x] = spl_z_2[i*blockDim.x + threadIdx.x];
    }
    
    __syncthreads();
    
    int this_row = offset_into_diag;
    int this_col = offset_into_diag + diag_first_col - static_cast<int>(diag_first_row);
    
    if (!is_valid_index(this_row, jtj_sz_row)) return;
    else if (!is_valid_index(this_col, jtj_sz_col)) return;
    
    unsigned int spl1_mid_x, spl1_mid_y, spl1_mid_z, spl2_mid_x, spl2_mid_y, spl2_mid_z;
    CBF::index_lin_to_vol(
        
        this_row,
        param_sz_x_1,
        param_sz_y_1,
        param_sz_z_1,
        
        &spl1_mid_x,
        &spl1_mid_y,
        &spl1_mid_z);
    CBF::index_lin_to_vol(
        
        this_col,
        param_sz_x_2,
        param_sz_y_2,
        param_sz_z_2,
        
        &spl2_mid_x,
        &spl2_mid_y,
        &spl2_mid_z);
    
    
    auto spl_diff_mid_x = static_cast<int>(spl1_mid_x) - static_cast<int>(spl2_mid_x);
    auto spl_diff_mid_y = static_cast<int>(spl1_mid_y) - static_cast<int>(spl2_mid_y);
    auto spl_diff_mid_z = static_cast<int>(spl1_mid_z) - static_cast<int>(spl2_mid_z);
    
    
    
    
    
    
    if (!CBF::is_valid_index(spl_diff_mid_x + spl_order, 2*spl_order + 1)) return;
    else if (!CBF::is_valid_index(spl_diff_mid_y + spl_order, 2*spl_order + 1)) return;
    else if (!CBF::is_valid_index(spl_diff_mid_z + spl_order, 2*spl_order + 1)) return;
    
    
    
    unsigned int spl1_xind_start, spl1_xind_end, spl2_xind_start;
    unsigned int spl1_yind_start, spl1_yind_end, spl2_yind_start;
    unsigned int spl1_zind_start, spl1_zind_end, spl2_zind_start;
    
    CBF::calculate_overlap(
        
        spl_diff_mid_x,
        spl_ksp_x_1,
        spl_order,
        
        &spl1_xind_start,
        &spl1_xind_end,
        &spl2_xind_start);
    CBF::calculate_overlap(
        
        spl_diff_mid_y,
        spl_ksp_y_1,
        spl_order,
        
        &spl1_yind_start,
        &spl1_yind_end,
        &spl2_yind_start);
    CBF::calculate_overlap(
        
        spl_diff_mid_z,
        spl_ksp_z_1,
        spl_order,
        
        &spl1_zind_start,
        &spl1_zind_end,
        &spl2_zind_start);
    
    
    int vol_xind_start =
      static_cast<int>(spl1_mid_x*spl_ksp_x_1) 
      - static_cast<int>(spl_order*spl_ksp_x_1 - 1) 
      + static_cast<int>(spl1_xind_start); 
    int vol_yind_start =
      static_cast<int>(spl1_mid_y*spl_ksp_y_1) 
      - static_cast<int>(spl_order*spl_ksp_y_1 - 1) 
      + static_cast<int>(spl1_yind_start); 
    int vol_zind_start =
      static_cast<int>(spl1_mid_z*spl_ksp_z_1) 
      - static_cast<int>(spl_order*spl_ksp_z_1 - 1) 
      + static_cast<int>(spl1_zind_start); 
    
    
    
    int i_start = 0;
    int i_end = spl1_xind_end - spl1_xind_start;
    if (vol_xind_start < 0) i_start = -vol_xind_start;
    if (vol_xind_start + i_end >= ima_sz_x) i_end = ima_sz_x - vol_xind_start - 1;
    int j_start = 0;
    int j_end = spl1_yind_end - spl1_yind_start;
    if (vol_yind_start < 0) j_start = -vol_yind_start;
    if (vol_yind_start + j_end >= ima_sz_y) j_end = ima_sz_y - vol_yind_start - 1;
    int k_start = 0;
    int k_end = spl1_zind_end - spl1_zind_start;
    if (vol_zind_start < 0) k_start = -vol_zind_start;
    if (vol_zind_start + k_end >= ima_sz_z) k_end = ima_sz_z - vol_zind_start - 1;
    float jtj_val = 0.0;
    
    for (int k = k_start
        ; k <= k_end
        ; ++k)
    {
      int vol_zind = vol_zind_start + k;
      for (int j = j_start
          ; j <= j_end
          ; ++j)
      {
        int vol_yind = vol_yind_start + j;
        for (int i = i_start
            ; i <= i_end
            ; ++i)
        {
          int vol_xind = vol_xind_start + i;
          
          unsigned int vol_lind = 0;
          unsigned int spl1_zind = spl1_zind_start + k;
          unsigned int spl2_zind = spl2_zind_start + k;
          unsigned int spl1_yind = spl1_yind_start + j;
          unsigned int spl2_yind = spl2_yind_start + j;
          unsigned int spl1_xind = spl1_xind_start + i;
          unsigned int spl2_xind = spl2_xind_start + i;
          CBF::index_vol_to_lin(
              
              vol_xind,
              vol_yind,
              vol_zind,
              ima_sz_x,
              ima_sz_y,
              ima_sz_z,
              
              &vol_lind);
          
          jtj_val += tex1Dfetch<float>(ima,vol_lind)
              * shared_spline_x_1[spl1_xind]
              * shared_spline_y_1[spl1_yind]
              * shared_spline_z_1[spl1_zind]
              * shared_spline_x_2[spl2_xind]
              * shared_spline_y_2[spl2_yind]
              * shared_spline_z_2[spl2_zind];
        }
      }
    }
    
    int diag_idx = blockIdx.x;
    jtj_values[diag_idx*jtj_sz_diag + this_row] = jtj_val;
  } 
} 

