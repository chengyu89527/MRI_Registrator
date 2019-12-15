/*! \file BiasFieldEstimator.h
    \brief Contains declaration of class for estimation of a bias field

    \author Jesper Andersson
    \version 1.0b, April, 2017.
*/
// 
// BiasFieldEstimator.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2017 University of Oxford 
//

#ifndef BiasFieldEstimator_h
#define BiasFieldEstimator_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include "newimage/newimageall.h"
#include "ECScanClasses.h"

namespace EDDY {

class BiasFieldEstimatorImpl;

 
class BiasFieldEstimator
{
public:
  BiasFieldEstimator();
  ~BiasFieldEstimator();
  
  void SetRefScan(const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  
  void AddScan(const NEWIMAGE::volume<float>& predicted, const NEWIMAGE::volume<float>& observed, 
	       const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  
  NEWIMAGE::volume<float> GetField(double ksp, double lambda) const;
  
  MISCMATHS::SpMat<float> GetAtMatrix(const EDDY::ImageCoordinates&  coords, 
				      const NEWIMAGE::volume<float>& predicted, 
				      const NEWIMAGE::volume<float>& mask) const;
  
  MISCMATHS::SpMat<float> GetAtStarMatrix(const EDDY::ImageCoordinates&  coords, const NEWIMAGE::volume<float>& mask) const;
  
  void Write(const std::string& basename) const;
private:
  BiasFieldEstimatorImpl* _pimpl;
};

} 
#endif 

