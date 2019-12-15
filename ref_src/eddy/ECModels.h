/*! \file ECModels.h
    \brief Contains declaration of classes that implements models for fields from eddy currents.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of classes that implements a hirearchy
// of models for fields from eddy currents induced by
// diffusion gradients.
// 
// ECModels.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef ECModels_h
#define ECModels_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"

namespace EDDY {

 
class ScanMovementModel
{
public:
  ScanMovementModel(unsigned int order) EddyTry : _order(order), _mp(static_cast<int>(6*(order+1))) { _mp=0.0; } EddyCatch
  ScanMovementModel(unsigned int                 order,
		    const NEWMAT::ColumnVector&  mp) EddyTry : _order(order), _mp(mp) {
    if (mp.Nrows() != static_cast<int>(6*(order+1))) throw EddyException("ScanMovementModel::ScanMovementModel: Mismatch between order and mp");
  } EddyCatch
  ~ScanMovementModel() EddyTry {} EddyCatch
  unsigned int Order() const { return(_order); }
  bool IsSliceToVol() const { return(_order!=0); }
  unsigned int NParam() const { return(6*(_order+1)); }
  NEWMAT::ColumnVector GetZeroOrderParams() const EddyTry { return(get_zero_order_mp()); } EddyCatch
  NEWMAT::ColumnVector GetParams() const EddyTry { return(_mp); } EddyCatch
  double GetParam(unsigned int indx) const EddyTry {
    if (int(indx) > _mp.Nrows()) throw EddyException("ScanMovementModel::GetParam: indx out of range");
    return(_mp(indx+1));
  } EddyCatch
  
  NEWMAT::ColumnVector GetGroupWiseParams(unsigned int grp, unsigned int ngrp) const EddyTry { return(get_gmp(grp,ngrp)); } EddyCatch
  
  void SetParams(const NEWMAT::ColumnVector& p) EddyTry {
    if (p.Nrows() == 6) set_zero_order_mp(p);
    else if (p.Nrows() == _mp.Nrows()) _mp = p;
    else throw EddyException("ScanMovementModel::SetParams: mismatched p");
  } EddyCatch
  
  void SetParam(unsigned int indx, double val) EddyTry {
    if (int(indx) > _mp.Nrows()) throw EddyException("ScanMovementModel::SetParam: indx out of range");
    _mp(indx+1) = val;
  } EddyCatch
  
  void SetGroupWiseParameters(const NEWMAT::Matrix& gwmp) EddyTry {
    if (gwmp.Nrows() != 6) throw EddyException("ScanMovementModel::SetGroupWiseParameters: gwmp must have 6 rows");
    NEWMAT::Matrix X = get_design(static_cast<unsigned int>(gwmp.Ncols()));
    NEWMAT::Matrix Hat = (X.t()*X).i()*X.t();
    NEWMAT::ColumnVector dctc;
    for (int i=0; i<6; i++) {
      dctc &= Hat*gwmp.Row(i+1).t();
    }
    _mp = dctc;
  } EddyCatch
  
  void SetOrder(unsigned int order) EddyTry {
    NEWMAT::ColumnVector tmp(6*(order+1)); tmp=0.0;
    unsigned int cpsz = (order < _order) ? order : _order;  
    for (int i=0; i<6; i++) {
      tmp.Rows(i*(order+1)+1,i*(order+1)+cpsz+1) = _mp.Rows(i*(_order+1)+1,i*(_order+1)+cpsz+1);
    }
    _mp=tmp; _order=order;
  } EddyCatch
  
  unsigned int NDerivs() const { return(NParam()); }
  
  double GetDerivScale(unsigned int dindx) const EddyTry {
    if (dindx>=6*(_order+1)) throw EddyException("ScanMovementModel::GetDerivScale: dindx out of range");
    return( (dindx<3*(_order+1)) ? 1e-2 : 1e-5 );
  } EddyCatch
  
  NEWMAT::Matrix GetHessian(unsigned int ngrp) const EddyTry {
    NEWMAT::Matrix hess(NDerivs(),NDerivs()); hess = 0.0;
    if (_order) {
      NEWMAT::DiagonalMatrix D(6); 
      for (int i=0; i<3; i++) D(i+1) = 1.0;
      for (int i=3; i<6; i++) D(i+1) = 100.0;
      hess = NEWMAT::KP(D,get_design_derivative(ngrp,2).t() * get_design_derivative(ngrp,2));
    }
    return(hess);
  } EddyCatch
  
  NEWMAT::Matrix ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan) const;
  
  NEWMAT::Matrix ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan, unsigned int grp, unsigned int ngrp) const;
  
  NEWMAT::Matrix InverseMovementMatrix(const NEWIMAGE::volume<float>& scan) const EddyTry { return(ForwardMovementMatrix(scan).i()); } EddyCatch
  
  NEWMAT::Matrix InverseMovementMatrix(const NEWIMAGE::volume<float>& scan, unsigned int grp, unsigned int ngrp) const EddyTry {
    if (grp>=ngrp) throw EddyException("ScanMovementModel::InverseMovementMatrix: grp has to be smaller than ngrp");
    return(ForwardMovementMatrix(scan,grp,ngrp).i()); 
  } EddyCatch
  
  NEWMAT::Matrix RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan,
						 const std::vector<unsigned int>&     rindx) const;
  
  NEWMAT::Matrix RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan, 
						 unsigned int                         grp, 
						 unsigned int                         ngrp,
						 const std::vector<unsigned int>&     rindx) const;
  
  NEWMAT::Matrix RestrictedInverseMovementMatrix(const NEWIMAGE::volume<float>&       scan,
						 const std::vector<unsigned int>&     rindx) const EddyTry { return(RestrictedForwardMovementMatrix(scan,rindx).i()); } EddyCatch
  
  NEWMAT::Matrix RestrictedInverseMovementMatrix(const NEWIMAGE::volume<float>&       scan, 
						 unsigned int                         grp, 
						 unsigned int                         ngrp,
						 const std::vector<unsigned int>&     rindx) const EddyTry {
    if (grp>=ngrp) throw EddyException("ScanMovementModel::RestrictedInverseMovementMatrix: grp has to be smaller than ngrp");
    return(RestrictedForwardMovementMatrix(scan,grp,ngrp,rindx).i()); 
  } EddyCatch

private:
  unsigned int           _order; 
  
  NEWMAT::ColumnVector   _mp;

  NEWMAT::ColumnVector get_zero_order_mp() const EddyTry {
    NEWMAT::ColumnVector zmp(6); zmp=0.0;
    for (int i=0, j=0; i<6; i++, j+=(int(_order)+1)) zmp(i+1) = _mp(j+1);
    return(zmp);
  } EddyCatch

  void set_zero_order_mp(const NEWMAT::ColumnVector& mp) EddyTry { for (int i=0, j=0; i<6; i++, j+=(int(_order)+1)) _mp(j+1) = mp(i+1); } EddyCatch
  
  NEWMAT::ColumnVector get_gmp(unsigned int grp, unsigned int ngrp) const EddyTry {
    double pi = 3.141592653589793;
    NEWMAT::ColumnVector gmp(6); gmp=0.0;
    for (unsigned int i=0; i<6; i++) {
      for (unsigned int j=0; j<(_order+1); j++) {
	if (j==0) gmp(i+1) += _mp(i*(_order+1)+j+1);
	else gmp(i+1) += _mp(i*(_order+1)+j+1) * cos((pi*double(j)*double(2*grp+1))/double(2*ngrp));
      }
    }
    return(gmp);
  } EddyCatch

  NEWMAT::Matrix get_design(unsigned int ngrp) const EddyTry {
    double pi = 3.141592653589793;
    NEWMAT::Matrix X(ngrp,_order+1);
    for (unsigned int i=0; i<ngrp; i++) {
      for (unsigned int j=0; j<(_order+1); j++) {
	if (j==0) X(i+1,j+1) = 1.0;
	else X(i+1,j+1) = cos((pi*double(j)*double(2*i+1))/double(2*ngrp));
      }
    }
    return(X);
  } EddyCatch

  NEWMAT::Matrix get_design_derivative(unsigned int ngrp, unsigned int dorder) const EddyTry {
    double pi = 3.141592653589793;
    NEWMAT::Matrix dX(ngrp,_order+1);
    for (unsigned int i=0; i<ngrp; i++) {
      for (unsigned int j=0; j<(_order+1); j++) {
	if (j==0) dX(i+1,j+1) = 0.0;
	else {
	  if (dorder==1) dX(i+1,j+1) = - (pi*double(j)/double(ngrp)) * sin((pi*double(j)*double(2*i+1))/double(2*ngrp));
	  else if (dorder==2) dX(i+1,j+1) = - this->sqr((pi*double(j)/double(ngrp))) * cos((pi*double(j)*double(2*i+1))/double(2*ngrp));
	  else throw EddyException("ScanMovementModel::get_design_derivative: Invalid derivative");
	}
      }
    }
    return(dX);
  } EddyCatch

  double sqr(double x) const { return(x*x); }
};

 
class ScanECModel
{
public:
  ScanECModel() EddyTry {} EddyCatch
  ScanECModel(const NEWMAT::ColumnVector& ep) EddyTry : _ep(ep) {} EddyCatch
  virtual ~ScanECModel() EddyTry {} EddyCatch
  
  virtual ECModel WhichModel() const = 0; 
  
  virtual bool HasFieldOffset() const = 0;
  
  virtual double GetFieldOffset() const = 0;
  
  virtual void SetFieldOffset(double ofst) = 0;
  
  unsigned int NParam() const { return(_ep.Nrows()); }
  
  NEWMAT::ColumnVector GetParams() const EddyTry { return(_ep); } EddyCatch
  
  void SetParams(const NEWMAT::ColumnVector& ep) EddyTry {
    if (ep.Nrows() != _ep.Nrows()) throw EddyException("ScanECModel::SetParams: Wrong number of parameters");
    _ep = ep;
  } EddyCatch
  
  virtual unsigned int NDerivs() const =0;
  
  
  
  
  virtual double GetDerivParam(unsigned int dindx) const = 0;
  
  virtual void SetDerivParam(unsigned int dindx, double p) = 0;
  
  virtual std::shared_ptr<ScanECModel> Clone() const = 0; 
  
  virtual double GetDerivScale(unsigned int dindx) const = 0;
  
  virtual NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const = 0;
protected:
  NEWMAT::ColumnVector _ep;
};









class PolynomialScanECModel : public ScanECModel
{
public:
  PolynomialScanECModel(bool field=false) EddyTry : ScanECModel() {
    _nepd = 0;
  } EddyCatch
  PolynomialScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : ScanECModel(ep), _nepd(0) {} EddyCatch
  virtual ~PolynomialScanECModel() EddyTry {} EddyCatch
  virtual ECModel WhichModel() const = 0; 
  virtual bool HasFieldOffset() const = 0;
  virtual double GetFieldOffset() const EddyTry { if (HasFieldOffset()) return(_ep(_nepd)); else return(0.0); } EddyCatch
  virtual void SetFieldOffset(double ofst) EddyTry { 
    if (!HasFieldOffset()) throw EddyException("PolynomialScanECModel::SetFieldOffset: Attempting to set offset for model without offset");
    _ep(_nepd) = ofst; 
  } EddyCatch
  virtual NEWMAT::RowVector GetLinearParameters() const EddyTry { return(_ep.Rows(1,3).t()); } EddyCatch
  virtual unsigned int NDerivs() const { return(_nepd); }
  virtual double GetDerivParam(unsigned int dindx) const EddyTry {
    if (dindx>=NDerivs()) throw EddyException("PolynomialScanECModel::GetDerivParam: dindx out of range");
    return(_ep(dindx+1));
  } EddyCatch
  virtual void SetDerivParam(unsigned int dindx, double p) EddyTry {
    if (dindx>=NDerivs()) throw EddyException("PolynomialScanECModel::SetDerivParam: dindx out of range");
    _ep(dindx+1) = p;
  } EddyCatch
  virtual std::shared_ptr<ScanECModel> Clone() const = 0; 
  virtual double GetDerivScale(unsigned int dindx) const = 0;
  virtual NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const = 0;
protected:
  unsigned int         _nepd;  
};

































class LinearScanECModel : public PolynomialScanECModel
{
public:
  LinearScanECModel(bool field=false) EddyTry : PolynomialScanECModel(field) {
    _ep.ReSize(4); _ep=0.0;
    _nepd = (field) ? 4 : 3;
  } EddyCatch
  LinearScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : PolynomialScanECModel(ep) {
    _nepd = (field) ? 4 : 3;
    if (_ep.Nrows() != int(_nepd)) throw EddyException("LinearScanECModel: Wrong number of elements for ep");
  } EddyCatch
  virtual ~LinearScanECModel() EddyTry {} EddyCatch
  ECModel WhichModel() const { return(Linear); } 
  bool HasFieldOffset() const { return(_nepd==4); }
  virtual std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new LinearScanECModel(*this))); } EddyCatch
  virtual double GetDerivScale(unsigned int dindx) const EddyTry { 
    if (dindx < 3) return(1e-3);
    else if (dindx < _nepd) return(1e-2); 
    else throw EddyException("LinearScanECModel::GetDerivScale: Index out of range");
  } EddyCatch
  virtual NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const;
};




































class QuadraticScanECModel : public PolynomialScanECModel
{
public:
  QuadraticScanECModel(bool field=false) EddyTry : PolynomialScanECModel(field) {
    _ep.ReSize(10); _ep=0.0;
    _nepd = (field) ? 10 : 9;
  } EddyCatch
  QuadraticScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : PolynomialScanECModel(ep,field) {
    _nepd = (field) ? 10 : 9;
    if (_ep.Nrows() != int(_nepd)) throw EddyException("QuadraticScanECModel: Wrong number of elements for ep");
  } EddyCatch
  virtual ~QuadraticScanECModel() EddyTry {} EddyCatch
  ECModel WhichModel() const { return(Quadratic); } 
  bool HasFieldOffset() const { return(_nepd==10); }
  std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new QuadraticScanECModel(*this))); } EddyCatch
  virtual double GetDerivScale(unsigned int dindx) const EddyTry { 
    if (dindx < 3) return(1e-3);
    else if (dindx < 9) return(1e-5);
    else if (dindx < _nepd) return(1e-2); 
    else throw EddyException("QuadraticScanECModel::GetDerivScale: Index out of range");
  } EddyCatch
  NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const;
};






































class CubicScanECModel : public PolynomialScanECModel
{
public:
  CubicScanECModel(bool field=false) EddyTry : PolynomialScanECModel(field) {
    _ep.ReSize(20); _ep=0.0;
    _nepd = (field) ? 20 : 19;
  } EddyCatch
  CubicScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : PolynomialScanECModel(ep) {
    _nepd = (field) ? 20 : 19;
    if (_ep.Nrows() != int(_nepd)) throw EddyException("CubicScanECModel: Wrong number of elements for ep");
  } EddyCatch
  virtual ~CubicScanECModel() EddyTry {} EddyCatch 
  ECModel WhichModel() const EddyTry { return(Cubic); } EddyCatch
  bool HasFieldOffset() const EddyTry { return(_nepd==20); } EddyCatch
  std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new CubicScanECModel(*this))); } EddyCatch
  virtual double GetDerivScale(unsigned int dindx) const EddyTry { 
    if (dindx < 3) return(1e-3);
    else if (dindx < 9) return(1e-5);
    else if (dindx < 19) return(1e-7);
    else if (dindx < _nepd) return(1e-2); 
    else throw EddyException("CubicScanECModel::GetDerivScale: Index out of range");
  } EddyCatch
  NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const;
};












class NoECScanECModel : public ScanECModel
{
public:
  NoECScanECModel() EddyTry : ScanECModel() {} EddyCatch
  NoECScanECModel(const NEWMAT::ColumnVector& ep) EddyTry {
    if (_ep.Nrows()) throw EddyException("NoECScanScanECModel: ep must have 0 elements");
  } EddyCatch
  virtual ~NoECScanECModel() EddyTry {} EddyCatch
  ECModel WhichModel() const { return(NoEC); } 
  bool HasFieldOffset() const { return(false); }
  double GetFieldOffset() const { return(0.0); }
  void SetFieldOffset(double ofst) { }
  unsigned int NDerivs() const { return(0); }
  double GetDerivParam(unsigned int dindx) const { throw EddyException("NoECScanECModel::GetDerivParam: Model has no EC parameters"); }
  void SetDerivParam(unsigned int dindx, double p) { throw EddyException("NoECScanECModel::SetDerivParam: Model has no EC parameters"); }
  double GetDerivScale(unsigned int dindx) const { throw EddyException("NoECScanECModel::GetDerivScale: Model has no EC parameters"); }
  std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new NoECScanECModel(*this))); } EddyCatch
  NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const EddyTry { NEWIMAGE::volume<float> field=scan; field=0.0; return(field); } EddyCatch
};

} 

#endif 

