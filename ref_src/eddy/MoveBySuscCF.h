// Declarations of classes and functions that
// calculates the derivative fields for the
// movement-by-susceptibility modelling.
//
// MoveBySuscCF.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford 
//

#ifndef MoveBySuscCF_h
#define MoveBySuscCF_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"
#include "EddyCommandLineOptions.h"

namespace EDDY {


 
class MoveBySuscCFImpl;
class MoveBySuscCF : public MISCMATHS::NonlinCF
{
public:
  MoveBySuscCF(EDDY::ECScanManager&                 sm,
	       const EDDY::EddyCommandLineOptions&  clo,
	       const std::vector<unsigned int>&     b0s,
	       const std::vector<unsigned int>&     dwis,
	       const std::vector<unsigned int>&     mps,
	       unsigned int                         order,
	       double                               ksp);
  ~MoveBySuscCF();
  double cf(const NEWMAT::ColumnVector& p) const;
  NEWMAT::ReturnMatrix grad(const NEWMAT::ColumnVector& p) const;
  boost::shared_ptr<BFMatrix> hess(const NEWMAT::ColumnVector& p,
				   boost::shared_ptr<BFMatrix> iptr=boost::shared_ptr<BFMatrix>()) const;
  void SetLambda(double lambda);
  NEWMAT::ReturnMatrix Par() const;
  unsigned int NPar() const;
  void WriteFirstOrderFields(const std::string& fname) const;
  void WriteSecondOrderFields(const std::string& fname) const;
  void ResetCache();
private:
  mutable MoveBySuscCFImpl             *_pimpl;
};

} 

#endif 

