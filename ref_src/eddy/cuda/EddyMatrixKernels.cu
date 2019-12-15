

#include <stdio.h>
#include <cuda.h>
#include <math_constants.h>
#include <math_functions.h>
#include "EddyKernels.h"
#include "eddy_matrix_kernels_internal.h"
#include "EddyMatrixKernels.h"



namespace EddyMatrixKernels {

using namespace EMKI;

					
__global__ void QR(
		   const float  *K,     
		   unsigned int m,      
		   unsigned int n,      
		   unsigned int nmat,   
		   
		   float        *Qt,    
		   float        *R)     
{
  extern __shared__ float scratch[];

  if (blockIdx.x < nmat && threadIdx.x < m) {
    unsigned int id = threadIdx.x;
    unsigned int ntpm = min(m,blockDim.x); 
    float *v = scratch;
    float *w = &scratch[m];
    const float *lK = &K[blockIdx.x*m*n];
    float *lQt = &Qt[blockIdx.x*m*m];
    float *lR = &R[blockIdx.x*m*n];
    qr_single(lK,m,n,v,w,id,ntpm,lQt,lR);
  }
  return;
}

					
__global__ void Solve(
		      const float *Qt,   
		      const float *R,    
		      const float *y,    
		      unsigned int m,    
		      unsigned int n,    
		      unsigned int nmat, 
		      
		      float       *y_hat)
{
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= nmat) return;
  const float *lQt = &Qt[id*m*m];
  const float *lR = &R[id*m*n];
  const float *ly = &y[id*m];
  float *ly_hat = &y_hat[id*n];
  solve_single(lQt,lR,ly,m,n,ly_hat);
  return;
}

					
__global__ void KtK(
		    const float  *K,     
		    unsigned int m,      
		    unsigned int n,      
		    unsigned int nmat,   
		    const float  *StS,   
		    float        lambda, 
		    bool         rf,     
		    
		    float        *KtK)   
{
  if (blockIdx.x < nmat) {
    const float    *lK = &K[blockIdx.x*m*n];
    float          *lKtK = &KtK[blockIdx.x*n*n];
    if (rf) rf_KtK_one_mat(lK,m,n,StS,lambda,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,lKtK);
    else cf_KtK_one_mat(lK,m,n,StS,lambda,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,lKtK);
  }
  return;
}

					
__global__ void Kty(
                    const float   *K,      
		    const float   *y,      
		    unsigned int  m,       
		    unsigned int  n,       
		    unsigned int  nmat,    
		    
		    float         *Kty)    
{
  if (blockIdx.x < nmat) {
    const float *lK = &K[blockIdx.x*m*n];
    const float *ly = &y[blockIdx.x*m];
    float *lKty = &Kty[blockIdx.x*n];
    Kty_one_mat(lK,ly,m,n,threadIdx.x,blockDim.x,lKty);
  }
  return;
}

					
__global__ void Wir(
                    const float   *zcoord,  
		    unsigned int  xsz,      
		    unsigned int  ysz,      
		    unsigned int  zsz,      
		    unsigned int  nmat,     
		    unsigned int  xzp,      
		    
		    float         *Wir)     
{
  if (blockIdx.x < nmat) {
    float *lWir = &Wir[blockIdx.x*zsz*zsz];
    const float *lzcoord = &zcoord[xzp*xsz+blockIdx.x];
    unsigned int zstep = xsz*ysz;
    Wir_one_mat(lzcoord,zstep,zsz,threadIdx.x,blockDim.x,lWir);
  }
  return;
}

					
__global__ void Wirty(
		      const float   *y,       
		      const float   *Wir,     
		      unsigned int  xsz,      
		      unsigned int  ysz,      
		      unsigned int  zsz,      
		      unsigned int  nmat,     
		      unsigned int  xzp,      
		      
		      float         *Wirty)   
{
  if (blockIdx.x < nmat) {
    const float *ly = &y[xzp*xsz+blockIdx.x];
    const float *lWir = &Wir[blockIdx.x*zsz*zsz];
    unsigned int zstep = xsz*ysz;
    float *lWirty = &Wirty[blockIdx.x*zsz];
    Wirty_one_mat(ly,lWir,zstep,zsz,threadIdx.x,blockDim.x,lWirty);
  }
  return;
}

	
__global__ void Atb(
		    const float   *A,      
		    const float   *b,      
		    unsigned int  m,       
		    unsigned int  n,       
		    unsigned int  nmat,    
		    unsigned int  nvec,    
		    
		    float         *Atb)     
{
  if (blockIdx.x < nvec) {
    const float *lA;
    if (nmat==1) lA = A;
    else if (nmat==nvec) lA = &A[blockIdx.x*m*n];
    else *(int*)0 = 0; 
    const float *lb = &b[blockIdx.x*n];
    float *lAtb = &Atb[blockIdx.x*m];
    Atb_one_mat(lA,lb,m,n,threadIdx.x,blockDim.x,lAtb);
  }
  return;
}
					
__global__ void Ab(
		   const float   *A,      
		   const float   *b,      
		   unsigned int  m,       
		   unsigned int  n,       
		   unsigned int  nmat,    
		   unsigned int  nvec,    
		   
		   float         *Ab)     
{
  if (blockIdx.x < nvec) {
    const float *lA;
    if (nmat==1) lA = A;
    else if (nmat==nvec) lA = &A[blockIdx.x*m*n];
    else *(int*)0 = 0; 
    const float *lb = &b[blockIdx.x*n];
    float *lAb = &Ab[blockIdx.x*m];
    Ab_one_mat(lA,lb,m,n,threadIdx.x,blockDim.x,lAb);
  }
  return;
}

					
__global__ void DiagwA(const float *w,    
		       const float *A,    
		       unsigned int m,    
		       unsigned int n,    
		       unsigned int nvec, 
		       float        *wA)  
{
  unsigned int mat=blockIdx.x;
  unsigned int row=threadIdx.x;
  if (mat<nvec && row<m) {
    float wgt = w[mat*m+row];
    float *wAp = wA + mat*m*n;
    for (unsigned int c=0; c<n; c++) {
      wAp[rf_indx(row,c,m,n)] = A[rf_indx(row,c,m,n)] * wgt;
    }
  }
}

} 


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
			  float       *R)       
{
  
  M_eq_M(R,K,m,n,nt,id);
  
  set_to_identity(Qt,m,nt,id);
  
  for (int j=0; j<min(m-1,n); j++) {
    
    float alfa = get_alfa(R,m,n,j,nt,id,v);
    
    get_v(R,m,n,j,alfa,nt,id,v);
    
    two_x_vt_x_R(R,m,n,j,nt,id,v,w);
    
    R_minus_v_x_wt(m,n,j,nt,id,v,w,R);
    
    two_x_vt_x_Qt(Qt,m,m,j,nt,id,v,w);
    
    Qt_minus_v_x_wt(m,m,j,nt,id,v,w,Qt);
  }
  return;
}

__device__ void M_eq_M(float       *dM, 
                       const float *sM, 
		       unsigned int m,   
		       unsigned int n,   
		       unsigned int nt,  
		       unsigned int id)  
{
  unsigned int rpt = m/nt; rpt += (m%nt) ? 1 : 0;
  for (int r=0; r<rpt; r++) {
    unsigned int i = id + r*nt;
    if (i < m) {
      for (int j=0; j<n; j++) {
        unsigned int ii = rf_indx(i,j,m,n);
	dM[ii] = sM[ii];
      }
    }
  }
  return;
}

__device__ void set_to_identity(float        *M,   
                                unsigned int  m,   
				unsigned int  nt,  
				unsigned int  id)  
{
  unsigned int rpt = m/nt; rpt += (m%nt) ? 1 : 0;
  for (int r=0; r<rpt; r++) {
    unsigned int i = id + r*nt;
    if (i < m) {
      for (int j=0; j<m; j++) {
	unsigned int ii = rf_indx(i,j,m,m);
	if (i==j) M[ii] = 1.0;
	else M[ii] = 0.0;
      }
    }
  }
  return;
}

__device__ float get_alfa(const float  *M,   
			  unsigned int m,    
			  unsigned int n,    
			  unsigned int j,    
			  unsigned int nt,   
			  unsigned int id,   
			  float        *scr) 
{
  unsigned int ept = (m-j)/nt; ept += ((m-j)%nt) ? 1 : 0; 
  scr[id] = 0.0;
  for (int r=0; r<ept; r++) {
    unsigned int i = j + id + nt*r;
    if (i<m) scr[id] += sqr(M[rf_indx(i,j,m,n)]);
  }
  __syncthreads();
  unsigned int npt = nt;
  if (!is_pow_of_two(npt)) npt = next_pow_of_two(nt);
  unsigned int s = npt>>1;
  if (id<s && id+s<nt) scr[id] += scr[id+s];
  __syncthreads();
  for (s>>=1; s>0; s>>=1) {
    if (id<s) scr[id] += scr[id+s];
    __syncthreads();
  }
  float alfa = sqrt((float) scr[0]);
  alfa = (M[rf_indx(j,j,m,n)] > 0) ? alfa : -alfa;

  return(alfa);
}

__device__ void get_v(const float  *M,  
		      unsigned int m,   
		      unsigned int n,   
		      unsigned int j,   
		      float        alfa,
		      unsigned int nt,  
		      unsigned int id,  
		      float        *v)  
{
  unsigned int ept = (m-j)/nt; ept += ((m-j)%nt) ? 1 : 0; 
  
  for (int e=0; e<ept; e++) {
    unsigned int i = j + id + e*nt;
    if (i<m) v[i] = M[rf_indx(i,j,m,n)];
  }
  __syncthreads();
  
  float norm_v = sqrt(2.0*sqr(alfa) + 2.0*v[j]*alfa);
  
  for (int e=0; e<ept; e++) {
    unsigned int i = j + id + e*nt;
    if (i==j) v[i] += alfa;
    if (i<m) v[i] /= norm_v;
  }
  __syncthreads();
  return;
}

__device__ void two_x_vt_x_R(const float  *R,      
			     unsigned int m,       
			     unsigned int n,       
			     unsigned int j,       
                             unsigned int nt,      
                             unsigned int id,      
			     const float  *v,      
			     float        *twovtR) 
{
  
  unsigned int ept = j/nt; ept += (j%nt) ? 1 : 0; 
  for (int e=0; e<ept; e++) {
    unsigned int i = id + e*nt;
    if (i<j) twovtR[i] = 0.0;
  }
  __syncthreads();
  
  ept = (n-j)/nt; ept += ((n-j)%nt) ? 1 : 0; 
  
  for (int e=0; e<ept; e++) {
    unsigned int i = j + id + e*nt;
    if (i<n) {
      twovtR[i] = 0.0;
      for (int ii=j; ii<m; ii++) twovtR[i] += v[ii]*R[rf_indx(ii,i,m,n)];
      twovtR[i] *= 2.0;
      
    }
  }  
  __syncthreads();
  return;
}

__device__ void R_minus_v_x_wt(unsigned int m,  
			       unsigned int n,  
			       unsigned int j,  
			       unsigned int nt, 
			       unsigned int id, 
			       const float  *v, 
			       const float  *w, 
			       float        *R) 
{
  unsigned int rpt = (m-j)/nt; rpt += ((m-j)%nt) ? 1 : 0; 
  for (unsigned int r=0; r<rpt; r++) {
    unsigned int i = j + id + nt*r;
    if (i<m) {
      for (int jj=j; jj<n; jj++) R[rf_indx(i,jj,m,n)] -= v[i]*w[jj];
    }
  }
  __syncthreads();
  return;
}

__device__ void two_x_vt_x_Qt(const float  *Qt,     
			      unsigned int m,       
			      unsigned int n,       
			      unsigned int j,       
			      unsigned int nt,      
			      unsigned int id,      
			      const float  *v,      
			      float        *twovtQt)
{
  
  unsigned int ept = m/nt; ept += (m%nt) ? 1 : 0; 
  for (int e=0; e<ept; e++) {
    unsigned int i = id + e*nt;
    if (i<n) {
      twovtQt[i] = 0.0;
      for (int ii=j; ii<m; ii++) twovtQt[i] += v[ii]*Qt[rf_indx(ii,i,m,n)];
      twovtQt[i] *= 2.0;
    }
  }  
  __syncthreads();
  return;
}

__device__ void Qt_minus_v_x_wt(unsigned int m,  
			        unsigned int n,  
			        unsigned int j,  
			        unsigned int nt, 
			        unsigned int id, 
			        const float  *v, 
			        const float  *w, 
			        float        *Qt)
{
  unsigned int rpt = (m-j)/nt; rpt += ((m-j)%nt) ? 1 : 0; 
  for (int r=0; r<rpt; r++) {
    unsigned int i = j + id + nt*r;
    if (i<m) {
      for (int jj=0; jj<n; jj++) Qt[rf_indx(i,jj,m,n)] -= v[i]*w[jj];
    }
  }
  __syncthreads();
  return;
}

__device__ bool is_pow_of_two(unsigned int n)
{
  return(!(n & (n-1)));
}
__device__ unsigned int next_pow_of_two(unsigned int n)
{
  n--; n|=n>>1; n|=n>>2; n|=n>>4; n|=n>>8; n|=n>>16; n++;
  return(n);
}


__device__ unsigned int rf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n)
{
  return(i+j*m);    
}


__device__ unsigned int cf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n)
{
  return(i*n+j);
}




__device__ void solve_single(
			     const float *Qt,     
			     const float *R,      
			     const float *y,      
			     unsigned int m,
			     unsigned int n,
			     
			     float       *y_hat)  
{
  
  M_times_v(Qt,y,n,m,y_hat);
  
  back_substitute(R,n,n,y_hat);
}

__device__ void back_substitute(const float  *R,
				unsigned int  m,
				unsigned int  n,
				float        *v)
{
  for (int i=n-1; i>=0; i--) {
    float tmp = v[i];
    for (int j=n-1; j>i; j--) tmp -= R[rf_indx(i,j,m,n)] * v[j];
    v[i] = tmp / R[rf_indx(i,i,m,n)];
  }
  return;
}

__device__ void M_times_v(const float *M,
			  const float *v,
			  unsigned int m,
			  unsigned int n,
			  float       *Mv)
{
  for (int i=0; i<m; i++) {
    Mv[i] = 0.0;
    for (int j=0; j<n; j++) Mv[i] += M[rf_indx(i,j,m,n)] * v[j];
  }
  return;
}



__device__ void cf_KtK_one_mat(const float  *K,   
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, 
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK) 
{
  unsigned int rpt = n/ntr; rpt += (n%ntr) ? 1 : 0; 
  unsigned int cpt = n/ntc; cpt += (n%ntc) ? 1 : 0; 
  for (int i=0; i<rpt; i++) {
    unsigned int r = idr + i*ntr;
    if (r < n) {
      for (int j=0; j<cpt; j++) {
	unsigned int c = idc + j*ntc;
	if (c < n) {
	  float val = 0.0;
	  for (int jj=0; jj<m; jj++) {
	    val += K[cf_indx(jj,r,m,n)]*K[cf_indx(jj,c,m,n)];
	  }
	  if (StS && lambda) KtK[rf_indx(r,c,n,n)] = val + StS[cf_indx(r,c,n,n)];
	  else KtK[rf_indx(r,c,n,n)] = val;
	}
      }
    }
  }
  return;
}

__device__ void rf_KtK_one_mat(const float  *K,   
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, 
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK) 
{
  unsigned int rpt = n/ntr; rpt += (n%ntr) ? 1 : 0; 
  unsigned int cpt = n/ntc; cpt += (n%ntc) ? 1 : 0; 
  for (int i=0; i<rpt; i++) {
    unsigned int r = idr + i*ntr;
    if (r < n) {
      for (int j=0; j<cpt; j++) {
	unsigned int c = idc + j*ntc;
	if (c < n) {
	  float val = 0.0;
	  for (int jj=0; jj<m; jj++) {
	    val += K[rf_indx(jj,r,m,n)]*K[rf_indx(jj,c,m,n)];
	  }
	  if (StS && lambda) KtK[rf_indx(r,c,n,n)] = val + StS[rf_indx(r,c,n,n)];
	  else KtK[rf_indx(r,c,n,n)] = val;
	}
      }
    }
  }
  return;
}

__device__ void Ab_one_mat(
			   const float   *A,  
			   const float   *b,
			   unsigned int  m,
			   unsigned int  n,
			   unsigned int  id,
			   unsigned int  ntr,
			   
			   float         *Ab)
{
  unsigned int ept = m/ntr; ept += (m%ntr) ? 1 : 0; 
  for (int i=0; i<ept; i++) {
    unsigned int r = id + i*ntr;
    if (r < m) {
      float val = 0.0;
      for (int c=0; c<n; c++) {
	val += A[rf_indx(r,c,m,n)]*b[c];            
      }
      Ab[r] = val;
    }
  }
  return;
}

__device__ void Atb_one_mat(
			    const float   *A,  
			    const float   *b,
			    unsigned int  m,
			    unsigned int  n,
			    unsigned int  id,
			    unsigned int  ntr,
			    
			    float         *Atb)
{
  unsigned int ept = n/ntr; ept += (n%ntr) ? 1 : 0; 
  for (int i=0; i<ept; i++) {
    unsigned int c = id + i*ntr;
    if (c < n) {
      float val = 0.0;
      for (int r=0; r<m; r++) {
	val += A[rf_indx(r,c,m,n)]*b[r];            
      }
      Atb[c] = val;
    }
  }
  return;
}

__device__ void Kty_one_mat(
			    const float     *K, 
			    const float     *y,
			    unsigned int    m,
			    unsigned int    n,
			    unsigned int    id,
			    unsigned int    ntr,
			    
			    float           *Kty)
{
  unsigned int ept = n/ntr; ept += (n%ntr) ? 1 : 0; 
  for (int i=0; i<ept; i++) {
    unsigned int r = id + i*ntr;
    if (r < n) {
      float val = 0.0;
      for (int c=0; c<m; c++) {
	val += K[cf_indx(c,r,m,n)]*y[c]; 
      }
      Kty[r] = val;
    }
  }
  return;
}

__device__ void Wirty_one_mat(
			      const float   *y,
			      const float   *Wir,  
			      unsigned int  zstep,
			      unsigned int  mn,
			      unsigned int  id,
			      unsigned int  ntr,
			      
			      float         *Wirty)
{
  unsigned int ept = mn/ntr; ept += (mn%ntr) ? 1 : 0; 
  for (unsigned int i=0; i<ept; i++) {
    unsigned int r = id + i*ntr;
    if (r < mn) {
      Wirty[r] = 0.0;
      for (unsigned int c=0; c<mn; c++) Wirty[r] += Wir[rf_indx(c,r,mn,mn)] * y[c*zstep]; 
    }
  }
  return;
}

__device__ void Wir_one_mat(
			    const float   *zcoord,
			    unsigned int  zstep,
			    unsigned int  mn,
			    unsigned int  id,
			    unsigned int  ntr,
			    
			    float         *Wir)  
{
  unsigned int rpt = mn/ntr; rpt += (mn%ntr) ? 1 : 0;  
  for (unsigned int i=0; i<rpt; i++) {
    unsigned int r = id + i*ntr;
    if (r < mn) {
      for (unsigned int c=0; c<mn; c++) Wir[rf_indx(r,c,mn,mn)] = 0.0;
      float z = zcoord[r*zstep];
      if (z>=0 && z<=(mn-1)) {
        int iz = static_cast<int>(z);
	for (int c=iz-2; c<iz+3; c++) {
	  Wir[rf_indx(r,min(max(0,c),static_cast<int>(mn)-1),mn,mn)] += wgt_at(z-c);
	}
      } 
    }
  }
  return;
}

__device__ float wgt_at(float x)
{
  float wgt = 0;
  x = (x<0.0) ? -x : x;
  if (x < 1) wgt = 2.0/3.0 + 0.5*x*x*(x-2.0);
  else if (x < 2) wgt = (1.0/6.0) * (2.0-x)*(2.0-x)*(2.0-x);

  return(wgt);
}

} 

