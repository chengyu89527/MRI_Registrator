


#ifndef EddyMatrixKernels_h
#define EddyMatrixKernels_h

#include <cuda.h>

namespace EddyMatrixKernels {

__global__ void QR(
		   const float  *K,     
		   unsigned int m,      
		   unsigned int n,      
		   unsigned int nmat,   
		   
		   float        *Qt,    
		   float        *R);    

__global__ void Solve(
		      const float *Qt,     
		      const float *R,      
		      const float *y,      
		      unsigned int m,      
		      unsigned int n,      
		      unsigned int nmat,   
		      
		      float       *y_hat); 

__global__ void KtK(
		    const float  *K,     
		    unsigned int m,      
		    unsigned int n,      
		    unsigned int nmat,   
		    const float  *StS,   
		    float        lambda, 
		    bool         rf,     
		    
		    float        *KtK);  

__global__ void Kty(
                    const float   *K,      
		    const float   *y,      
		    unsigned int  m,       
		    unsigned int  n,       
		    unsigned int  nmat,    
		    
		    float         *Kty);   

__global__ void Wir(
                    const float   *zcoord,  
		    unsigned int  xsz,      
		    unsigned int  ysz,      
		    unsigned int  zsz,      
		    unsigned int  nmat,     
		    unsigned int  xzp,      
		    
		    float         *Wir);    

__global__ void Wirty(
		      const float   *y,       
		      const float   *Wir,     
		      unsigned int  xsz,      
		      unsigned int  ysz,      
		      unsigned int  zsz,      
		      unsigned int  nmat,     
		      unsigned int  xzp,      
		      
		      float         *Wirty);  

__global__ void Atb(
		    const float   *A,      
		    const float   *b,      
		    unsigned int  m,       
		    unsigned int  n,       
		    unsigned int  nmat,    
		    unsigned int  nvec,    
		    
		    float         *Atb);    

__global__ void Ab(
		   const float   *A,      
		   const float   *b,      
		   unsigned int  m,       
		   unsigned int  n,       
		   unsigned int  nmat,    
		   unsigned int  nvec,    
		   
		   float         *Ab);    

__global__ void DiagwA(const float *w,    
		       const float *A,    
		       unsigned int m,    
		       unsigned int n,    
		       unsigned int nvec, 
		       float        *wA); 

} 

#endif 

