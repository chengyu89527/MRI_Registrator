/*! \file LSResampler.h
    \brief Contains declaration of class for least-squares resampling of pairs of images

    \author Jesper Andersson
    \version 1.0b, April, 2013.
*/
// 
// LSResampler.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef LSResampler_h
#define LSResampler_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include "newimage/newimageall.h"
#include "ECScanClasses.h"

namespace EDDY {

class LSResamplerImpl;

 
class LSResampler
{
public:
  
  LSResampler(const EDDY::ECScan&                               s1, 
	      const EDDY::ECScan&                               s2,
	      std::shared_ptr<const NEWIMAGE::volume<float> >   hzfield,
	      double                                            lambda=0.01);
  ~LSResampler();
  
  const NEWIMAGE::volume<float>& GetResampledVolume() const;
  
  const NEWIMAGE::volume<float>& GetMask() const;
private:
  LSResamplerImpl* _pimpl;
};

} 
#endif 

