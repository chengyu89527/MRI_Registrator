/*! \file DWIPredictionMaker.h
    \brief Contains declaration of virtual base class for making predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of virtual base class for
// making predictions about DWI data.
//
// DWIPredictionMaker.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef DWIPredictionMaker_h
#define DWIPredictionMaker_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"

namespace EDDY {


 
class DWIPredictionMaker
{
public:
  DWIPredictionMaker() {}
  virtual ~DWIPredictionMaker() {}
  
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) const = 0;
  
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) = 0;
  
  virtual NEWIMAGE::volume<float> PredictCPU(unsigned int indx, bool exclude=false) = 0;
  
  virtual std::vector<NEWIMAGE::volume<float> > Predict(const std::vector<unsigned int>& indicies, bool exclude=false) = 0;
  
  virtual NEWIMAGE::volume<float> InputData(unsigned int indx) const = 0;
  
  virtual std::vector<NEWIMAGE::volume<float> > InputData(const std::vector<unsigned int>& indicies) const = 0;
  
  virtual double PredictionVariance(unsigned int indx, bool exclude=false) = 0;
  
  virtual double ErrorVariance(unsigned int indx) const = 0;
  
  virtual bool IsPopulated() const = 0;
  
  virtual bool IsValid() const = 0;
  
  virtual void SetNoOfScans(unsigned int n) = 0;
  
  
  virtual void SetScan(const NEWIMAGE::volume<float>& scan,  
                       const DiffPara&                dp,
                       unsigned int                   indx) = 0;
  
  virtual unsigned int NoOfHyperPar() const = 0;
  
  virtual std::vector<double> GetHyperPar() const = 0;
  
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, bool verbose=false) = 0;
  
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, float fwhm, bool verbose=false) = 0;
  
  virtual void WriteImageData(const std::string& fname) const = 0;
  virtual void WriteMetaData(const std::string& fname) const = 0;
  virtual void Write(const std::string& fname) const = 0;
};

} 

#endif 




















