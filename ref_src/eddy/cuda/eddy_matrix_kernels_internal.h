

#ifndef eddy_matrix_kernels_internal_h
#define eddy_matrix_kernels_internal_h

#include <cuda.h>

namespace EMKI {

__device__ void qr_single(
			  const float *K,       
			  unsigned int m,
			  unsigned int n,
			  
			  float       *v,       
			  float       *w,       
			  
			  unsigned int id,       
			  unsigned int nt,       
			  
			  float       *Qt,      
			  float       *R);      

__device__ void M_eq_M(float       *dM, 
                       const float *sM, 
		       unsigned int m,   
		       unsigned int n,   
		       unsigned int nt,  
		       unsigned int id); 

__device__ void set_to_identity(float        *M,   
                                unsigned int  m,   
				unsigned int  nt,  
				unsigned int  id); 

__device__ float get_alfa(const float  *M,   
			  unsigned int m,    
			  unsigned int n,    
			  unsigned int j,    
			  unsigned int nt,   
			  unsigned int id,   
			  float        *scr);

__device__ void get_v(const float  *M,  
		      unsigned int m,   
		      unsigned int n,   
		      unsigned int j,   
		      float        alfa,
		      unsigned int nt,  
		      unsigned int id,  
		      float        *v); 

__device__ void two_x_vt_x_R(const float  *R,      
			     unsigned int m,       
			     unsigned int n,       
			     unsigned int j,       
                             unsigned int nt,      
                             unsigned int id,      
			     const float  *v,      
			     float        *twovtR);

__device__ void R_minus_v_x_wt(unsigned int m,  
			       unsigned int n,  
			       unsigned int j,  
			       unsigned int nt, 
			       unsigned int id, 
			       const float  *v, 
			       const float  *w, 
			       float        *R);

__device__ void two_x_vt_x_Qt(const float  *Qt,      
			      unsigned int m,        
			      unsigned int n,        
			      unsigned int j,        
			      unsigned int nt,       
			      unsigned int id,       
			      const float  *v,       
			      float        *twovtQt);

__device__ void Qt_minus_v_x_wt(unsigned int m,   
			        unsigned int n,   
			        unsigned int j,   
			        unsigned int nt,  
			        unsigned int id,  
			        const float  *v,  
			        const float  *w,  
			        float        *Qt);

__device__ bool is_pow_of_two(unsigned int n);
__device__ unsigned int next_pow_of_two(unsigned int n);
__device__ unsigned int rf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n);
__device__ unsigned int cf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n);
template <typename T>
__device__ T sqr(const T& v) { return(v*v); }
template <typename T>
__device__ T min(const T& p1, const T& p2) { return((p1<p2) ? p1 : p2); }
template <typename T>
__device__ T max(const T& p1, const T& p2) { return((p1<p2) ? p2 : p1); }
__device__ float wgt_at(float x);



__device__ void solve_single(
			     const float *Qt,     
			     const float *R,      
			     const float *y,      
			     unsigned int m,
			     unsigned int n,
			     
			     float       *y_hat); 

__device__ void back_substitute(const float  *R,
				unsigned int  m,
				unsigned int  n,
				float        *v);

__device__ void M_times_v(const float *M,
			  const float *v,
			  unsigned int m,
			  unsigned int n,
			  float       *Mv);

__device__ void cf_KtK_one_mat(const float  *K,   
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, 
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK);

__device__ void rf_KtK_one_mat(const float  *K,   
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, 
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK);

__device__ void Ab_one_mat(
			   const float   *A,  
			   const float   *b,
			   unsigned int  m,
			   unsigned int  n,
			   unsigned int  id,
			   unsigned int  ntr,
			   
			   float         *Ab);

__device__ void Atb_one_mat(
			    const float   *A,  
			    const float   *b,
			    unsigned int  m,
			    unsigned int  n,
			    unsigned int  id,
			    unsigned int  ntr,
			    
			    float         *Atb);

__device__ void Kty_one_mat(
			    const float     *K,
			    const float     *y,
			    unsigned int    m,
			    unsigned int    n,
			    unsigned int    id,
			    unsigned int    ntr,
			    
			    float           *Kty);

__device__ void Wir_one_mat(
			    const float   *zcoord,
			    unsigned int  zstep,
			    unsigned int  mn,
			    unsigned int  id,
			    unsigned int  ntr,
			    
			    float         *Wir);

__device__ void Wirty_one_mat(
			      const float   *y,
			      const float   *Wir,  
			      unsigned int  zstep,
			      unsigned int  mn,
			      unsigned int  id,
			      unsigned int  ntr,
			      
			      float         *Wirty);

} 

#endif 

