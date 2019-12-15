/////////////////////////////////////////////////////////////////////
///
/// \file GpuPredictorChunk.cu
/// \brief Definition of helper class for efficient prediction making on the Gpu.
///
/// \author Jesper Andersson
/// \version 1.0b, March, 2013.
/// \Copyright (C) 2013 University of Oxford 
///


#include <cstdlib>
#include <string>
#include <vector>
#pragma push
#pragma diag_suppress = code_is_unreachable 
#include "newimage/newimage.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "cuda/GpuPredictorChunk.h"

namespace EDDY {

GpuPredictorChunk::GpuPredictorChunk(unsigned int ntot, const NEWIMAGE::volume<float>& ima) EddyTry : _ntot(ntot)
{
  
  int dev;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) throw EddyException("GpuPredictorChunk::GpuPredictorChunk: No device has been allocated");
  struct cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop,dev);
  if (err != cudaSuccess) throw EddyException("GpuPredictorChunk::GpuPredictorChunk: Unable to get device properties");
  
  float *skrutt = NULL;
  size_t memsz;
  for (memsz = 0.5 * prop.totalGlobalMem; memsz > my_sizeof(ima); memsz *= 0.9) {
    
    if (cudaMalloc(&skrutt,memsz) == cudaSuccess) break;
  }
  memsz *= 0.9; 
  if (memsz < my_sizeof(ima)) throw EddyException("GpuPredictorChunk::GpuPredictorChunk: Not enough memory on device");
  cudaFree(skrutt);
  
  _chsz = my_min(_ntot, static_cast<unsigned int>(memsz / my_sizeof(ima)));
  
  
  
  
  
  _ind.resize(_chsz);
  for (unsigned int i=0; i<_chsz; i++) _ind[i] = i;
} EddyCatch

GpuPredictorChunk& GpuPredictorChunk::operator++() EddyTry 
{
  if (_ind.back() == (_ntot-1)) { 
    _ind.resize(1);
    _ind[0] = _ntot;
  } 
  else {
    unsigned int first = _ind.back() + 1;
    unsigned int last = first + _chsz - 1;
    last = (last >= _ntot) ? _ntot-1 : last;
    _ind.resize(last-first+1);
    for (unsigned int i=0; i<_ind.size(); i++) {
      _ind[i] = first + i;
    }
  }
  return(*this);
} EddyCatch

} 

