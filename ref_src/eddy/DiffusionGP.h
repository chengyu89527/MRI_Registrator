/*! \file DiffusionGP.h
    \brief Contains declaration of class for making Gaussian process based predictions about DWI data.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of class to make Gaussian-Process
// based predictions about diffusion data.
//
// DiffusionGP.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef DiffusionGP_h
#define DiffusionGP_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "DWIPredictionMaker.h"
#include "KMatrix.h"
#include "HyParEstimator.h"

namespace EDDY {

 
class DiffusionGP : public DWIPredictionMaker
{
public:
  
  DiffusionGP(const std::shared_ptr<const KMatrix>&        Kmat,
	      const std::shared_ptr<const HyParEstimator>& hpe) EddyTry : _Kmat(Kmat->Clone()), _hpe(hpe->Clone()), _pop(true), _mc(false) {} EddyCatch
  
  DiffusionGP(const std::shared_ptr<const KMatrix>&        Kmat,
	      const std::shared_ptr<const HyParEstimator>& hpe,
	      const std::string&                           scans_fname,
              const std::string&                           var_mask_fname,
              const std::vector<DiffPara>&                 dpars,
	      float                                        fwhm=0.0,
	      bool                                         verbose=false);
  
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false) const;
  
  virtual NEWIMAGE::volume<float> Predict(unsigned int indx, bool exclude=false);
  
  virtual NEWIMAGE::volume<float> PredictCPU(unsigned int indx, bool exclude=false);
  
  virtual std::vector<NEWIMAGE::volume<float> > Predict(const std::vector<unsigned int>& indicies, bool exclude=false);
  
  virtual NEWIMAGE::volume<float> InputData(unsigned int indx) const;
  
  virtual std::vector<NEWIMAGE::volume<float> > InputData(const std::vector<unsigned int>& indicies) const;
  
  virtual double PredictionVariance(unsigned int indx, bool exclude=false);
  
  virtual double ErrorVariance(unsigned int indx) const;
  
  virtual bool IsPopulated() const { return(_pop); }
  
  virtual bool IsValid() const EddyTry { return(IsPopulated() && _mc && _Kmat->IsValid()); } EddyCatch
  
  virtual void SetNoOfScans(unsigned int n);
  
  virtual void SetScan(const NEWIMAGE::volume<float>& scan,  
		       const DiffPara&                dp,
		       unsigned int                   indx);
  
  virtual unsigned int NoOfHyperPar() const EddyTry { return(_Kmat->NoOfHyperPar()); } EddyCatch
  
  virtual std::vector<double> GetHyperPar() const EddyTry { return(_Kmat->GetHyperPar()); } EddyCatch
  
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, float fwhm, bool verbose=false);
  
  virtual void EvaluateModel(const NEWIMAGE::volume<float>& mask, bool verbose=false) EddyTry { EvaluateModel(mask,0.0,verbose); } EddyCatch
  
  virtual void WriteImageData(const std::string& fname) const;
  virtual void WriteMetaData(const std::string& fname) const EddyTry { 
    if (!IsPopulated()) throw EddyException("DiffusionGP::WriteMetaData: Not yet fully populated"); _Kmat->Write(fname); 
  } EddyCatch
  virtual void Write(const std::string& fname) const EddyTry { WriteImageData(fname); WriteMetaData(fname); } EddyCatch
private:
  static const unsigned int nvoxhp = 500;

  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > _sptrs;   
  std::shared_ptr<KMatrix>                                _Kmat;    
  std::vector<DiffPara>                                   _dpars;   
  std::shared_ptr<HyParEstimator>                         _hpe;     
  bool                                                    _pop;     
  bool                                                    _mc;      
  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > _mptrs;   

  void mean_correct(const std::vector<std::vector<unsigned int> >& mi);
  bool is_populated() const;
  void predict_image_cpu(unsigned int             indx,
			 bool                     excl,
			 const NEWMAT::RowVector& pv,
			 NEWIMAGE::volume<float>& ima) const;
  void predict_images_cpu(
			  const std::vector<unsigned int>&       indicies,
			  bool                                   exclude,
			  const std::vector<NEWMAT::RowVector>&  pvecs,
			  
			  std::vector<NEWIMAGE::volume<float> >& pi) const;
  #ifdef COMPILE_GPU
  void predict_image_gpu(unsigned int             indx,
			 bool                     excl,
			 const NEWMAT::RowVector& pv,
			 NEWIMAGE::volume<float>& ima) const;
  void predict_images_gpu(
			  const std::vector<unsigned int>&       indicies,
			  bool                                   exclude,
			  const std::vector<NEWMAT::RowVector>&  pvecs,
			  
			  std::vector<NEWIMAGE::volume<float> >& pi) const;
  #endif
  unsigned int which_mean(unsigned int indx) const;

  bool get_y(
	     unsigned int           i,
	     unsigned int           j,
	     unsigned int           k,
	     unsigned int           indx,
	     bool                   exclude,
	     
	     NEWMAT::ColumnVector&  y) const;
};

} 

#endif 

