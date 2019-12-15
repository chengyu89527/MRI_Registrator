/*! \file KMatrix.h
    \brief Contains declaration of virtual base class and a derived class for Covariance matrices for GP

    \author Jesper Andersson
    \version 1.0b, Oct., 2013.
*/
// Declarations of virtual base class for
// Covariance matrices for DWI data.
//
// KMatrix.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford 
//

#ifndef KMatrix_h
#define KMatrix_h

#include <cstdlib>
#include <string>
#include <exception>
#include <vector>
#include <cmath>
#include <memory>
#include "newmat.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"

namespace EDDY {

 
class KMatrix
{
public:
  KMatrix() {}
  virtual ~KMatrix() {}
  
  virtual std::shared_ptr<KMatrix> Clone() const = 0;
  
  virtual NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y) const = 0; 
  
  virtual NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y) = 0;
  
  virtual NEWMAT::RowVector PredVec(unsigned int i, bool excl) const = 0; 
  
  virtual NEWMAT::RowVector PredVec(unsigned int i, bool excl) = 0;
  
  virtual double PredVar(unsigned int i, bool excl) = 0;
  
  virtual double ErrVar(unsigned int i) const = 0;
  
  virtual const NEWMAT::SymmetricMatrix& iK() const = 0;
  
  virtual void Reset() = 0;
  
  virtual void SetDiffusionPar(const std::vector<DiffPara>& dpars) = 0;
  
  virtual const std::vector<std::vector<unsigned int> >& GetMeanIndicies() const = 0;
  
  virtual void SetHyperPar(const std::vector<double>& hpar) = 0;
  
  virtual void MulErrVarBy(double ff) = 0;
  
  virtual void CalculateInvK() = 0;
  
  virtual std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const = 0;
  
  virtual const std::vector<double>& GetHyperPar() const = 0;
  
  virtual unsigned int NoOfScans() const = 0;
  
  virtual unsigned int NoOfHyperPar() const = 0;
  
  virtual bool IsValid() const = 0;
  
  virtual double LogDet() const = 0;
  
  virtual NEWMAT::SymmetricMatrix GetDeriv(unsigned int i) const = 0;
  
  virtual void GetAllDerivs(std::vector<NEWMAT::SymmetricMatrix>& derivs) const = 0;
  
  virtual const NEWMAT::SymmetricMatrix& AsNewmat() const = 0;
  
  virtual void Print() const = 0;
  
  virtual void Write(const std::string& basefname) const = 0;
};

 
class MultiShellKMatrix : public KMatrix
{
public:
  MultiShellKMatrix(bool dcsh) EddyTry : _ngrp(0), _K_ok(false), _iK_ok(false), _dcsh(dcsh) {} EddyCatch
  MultiShellKMatrix(const std::vector<DiffPara>&          dpars,
		    bool                                  dcsh);
  ~MultiShellKMatrix() EddyTry {} EddyCatch
  
  NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y) const; 
  
  NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y);
  
  NEWMAT::RowVector PredVec(unsigned int i, bool excl) const; 
  
  NEWMAT::RowVector PredVec(unsigned int i, bool excl);
  
  double PredVar(unsigned int i, bool excl);
  
  double ErrVar(unsigned int i) const EddyTry { return(err_var(i,_grpi,_ngrp,_thpar)); } EddyCatch
  
  const NEWMAT::SymmetricMatrix& iK() const;
  
  const NEWMAT::SymmetricMatrix& iK();
  
  void Reset();
  
  void SetDiffusionPar(const std::vector<DiffPara>& dpars);
  
  const std::vector<std::vector<unsigned int> >& GetMeanIndicies() const EddyTry { return(_grps); } EddyCatch
  
  virtual void SetHyperPar(const std::vector<double>& hpar);
  
  virtual void MulErrVarBy(double ff);
  
  void CalculateInvK();
  
  virtual std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const = 0;
  
  const std::vector<double>& GetHyperPar() const EddyTry { return(_hpar); } EddyCatch
  
  bool IsValid() const { return(_K_ok); }
  
  unsigned int NoOfScans() const { return(_dpars.size()); }
  
  const unsigned int& NoOfGroups() const { return(_ngrp); }
  
  unsigned int NoOfHyperPar() const { return(_hpar.size()); }
  
  double LogDet() const;
  
  NEWMAT::SymmetricMatrix GetDeriv(unsigned int di) const;
  
  void GetAllDerivs(std::vector<NEWMAT::SymmetricMatrix>& derivs) const;
  
  const NEWMAT::SymmetricMatrix& AsNewmat() const EddyTry { return(_K); } EddyCatch
  
  void Print() const;
  
  void Write(const std::string& basefname) const;
protected:
  std::pair<unsigned int, unsigned int> parameter_index_to_ij(unsigned int pi,
							      unsigned int ngrp) const;
  
  double variance(const NEWMAT::ColumnVector&      data, 
		  const std::vector<unsigned int>& indx) const;
  
  void set_hpar(const std::vector<double>& hpar) EddyTry { _hpar = hpar; } EddyCatch
  
  void set_thpar(const std::vector<double>& thpar) EddyTry { _thpar = thpar; } EddyCatch
  
  
  
  NEWMAT::SymmetricMatrix& give_me_a_K() EddyTry { if (_K_ok) return(_K); else throw EddyException("MultiShellKMatrix::give_me_a_K: invalid K"); } EddyCatch
  
  void validate_K_matrix();
  
  void set_K_matrix_invalid() EddyTry { _K_ok = false; _iK_ok = false; _pv_ok.assign(_dpars.size(),false); } EddyCatch
  
  bool valid_hpars(const std::vector<double>& hpar) const;
  
  const std::vector<std::vector<unsigned int> >& grps() const EddyTry { return(_grps); } EddyCatch
  const std::vector<unsigned int>& grpi() const EddyTry { return(_grpi); } EddyCatch
  const std::vector<double>& grpb() const EddyTry { return(_grpb); } EddyCatch
  const std::vector<double>& thpar() const EddyTry { return(_thpar); } EddyCatch
  const NEWMAT::SymmetricMatrix& angle_mat() const EddyTry { return(_angle_mat); } EddyCatch
private:
  std::vector<DiffPara>                     _dpars;      
  std::vector<unsigned int>                 _grpi;       
  std::vector<std::vector<unsigned int> >   _grps;       
  std::vector<double>                       _grpb;       
  unsigned int                              _ngrp;       
  std::vector<double>                       _hpar;       
  std::vector<double>                       _thpar;      
  NEWMAT::SymmetricMatrix                   _angle_mat;  
  bool                                      _K_ok;       
  NEWMAT::SymmetricMatrix                   _K;          
  NEWMAT::LowerTriangularMatrix             _cK;         
  bool                                      _iK_ok;      
  NEWMAT::SymmetricMatrix                   _iK;         
  std::vector<NEWMAT::RowVector>            _pv;         
  std::vector<bool>                         _pv_ok;      
  bool                                      _dcsh;       

  virtual unsigned int ij_to_parameter_index(unsigned int i,
					     unsigned int j, 
					     unsigned int n) const = 0;
  virtual unsigned int n_par(unsigned int ngrp) const = 0;
  virtual void calculate_K_matrix(const std::vector<unsigned int>& grpi,
				  unsigned int                     ngrp,
				  const std::vector<double>&       thpar,
				  const NEWMAT::SymmetricMatrix&   angle_mat,
				  NEWMAT::SymmetricMatrix&         K) const = 0;
  virtual void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
				   unsigned int                     ngrp,
				   const std::vector<double>&       thpar,
				   const NEWMAT::SymmetricMatrix&   angle_mat,
				   unsigned int                     i,
				   unsigned int                     j,
				   unsigned int                     off,
				   NEWMAT::SymmetricMatrix&         dK) const = 0;
  virtual NEWMAT::RowVector k_row(unsigned int                     indx,
				  bool                             excl,
				  const std::vector<unsigned int>& grpi,
				  unsigned int                     ngrp,
				  const std::vector<double>&       thpar,
				  const NEWMAT::SymmetricMatrix&   angle_mat) const = 0;
  virtual std::vector<double> exp_hpar(unsigned int               ngrp,
				       const std::vector<double>& hpar) const = 0;
  virtual std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const = 0;
  virtual double sig_var(unsigned int                     i,
			 const std::vector<unsigned int>& grpi,
			 unsigned int                     ngrp,
			 const std::vector<double>&       thpar) const = 0;
  virtual double err_var(unsigned int                     i,
			 const std::vector<unsigned int>& grpi,
			 unsigned int                     ngrp,
			 const std::vector<double>&       thpar) const = 0;
  void calculate_iK();
  NEWMAT::SymmetricMatrix calculate_iK_index(unsigned int i) const;
  NEWMAT::Matrix make_pred_vec_matrix(bool excl=false) const;
  void make_angle_mat();
  double mean(const NEWMAT::ColumnVector&      data, 
	      const std::vector<unsigned int>& indx) const;
  double sqr(double a) const { return(a*a); }
};

 
class NewSphericalKMatrix : public MultiShellKMatrix
{
public:
  NewSphericalKMatrix(bool dcsh=false) EddyTry : MultiShellKMatrix(dcsh) {} EddyCatch
  NewSphericalKMatrix(const std::vector<DiffPara>&          dpars,
		      bool                                  dcsh=false);
  std::shared_ptr<KMatrix> Clone() const EddyTry { return(std::shared_ptr<NewSphericalKMatrix>(new NewSphericalKMatrix(*this))); } EddyCatch
  std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const;
  void SetHyperPar(const std::vector<double>& hpar);
  void MulErrVarBy(double ff);
private:
  unsigned int n_par(unsigned int ngrp) const { return(ngrp==1 ? 3 : 3 + ngrp); }
  unsigned int ij_to_parameter_index(unsigned int i,
				     unsigned int j, 
				     unsigned int n) const EddyTry { throw EddyException("NewSphericalKMatrix::ij_to_parameter_index: Invalid call"); } EddyCatch
  void calculate_K_matrix(const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat,
			  NEWMAT::SymmetricMatrix&         K) const;
  void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
			   unsigned int                     ngrp,
			   const std::vector<double>&       thpar,
			   const NEWMAT::SymmetricMatrix&   angle_mat,
			   unsigned int                     gi,
			   unsigned int                     gj,
			   unsigned int                     off,
			   NEWMAT::SymmetricMatrix&         dK) const;
  NEWMAT::RowVector k_row(unsigned int                     indx,
			  bool                             excl,
			  const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat) const;
  std::vector<double> exp_hpar(unsigned int               ngrp,
			       const std::vector<double>& hpar) const EddyTry
  { std::vector<double> epar = hpar; for (unsigned int i=0; i<hpar.size(); i++) epar[i] = std::exp(hpar[i]); return(epar); } EddyCatch
  std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const;
  double sig_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[0]); } EddyCatch
  double err_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { const double *ev = (ngrp > 1) ? &(thpar[3]) : &(thpar[2]); return(ev[grpi[i]]); } EddyCatch
};

 
class SphericalKMatrix : public MultiShellKMatrix
{
public:
  SphericalKMatrix(bool dcsh=false) EddyTry : MultiShellKMatrix(dcsh) {} EddyCatch
  SphericalKMatrix(const std::vector<DiffPara>&          dpars,
		   bool                                  dcsh=false);
  std::shared_ptr<KMatrix> Clone() const EddyTry { return(std::shared_ptr<SphericalKMatrix>(new SphericalKMatrix(*this))); } EddyCatch
  std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const;
private:
  unsigned int n_par(unsigned int ngrp) const { return(3*ngrp + (ngrp*ngrp - ngrp)); }
  unsigned int ij_to_parameter_index(unsigned int i,
					     unsigned int j, 
					     unsigned int n) const;
  void calculate_K_matrix(const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat,
			  NEWMAT::SymmetricMatrix&         K) const;
  void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
			   unsigned int                     ngrp,
			   const std::vector<double>&       thpar,
			   const NEWMAT::SymmetricMatrix&   angle_mat,
			   unsigned int                     gi,
			   unsigned int                     gj,
			   unsigned int                     off,
			   NEWMAT::SymmetricMatrix&         dK) const;
  NEWMAT::RowVector k_row(unsigned int                     indx,
			  bool                             excl,
			  const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat) const;
  std::vector<double> exp_hpar(unsigned int               ngrp,
			       const std::vector<double>& hpar) const;
  std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const;
  double sig_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)]); } EddyCatch
  double err_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)+2]); } EddyCatch
};

 
class ExponentialKMatrix : public MultiShellKMatrix
{
public:
  ExponentialKMatrix(bool dcsh=false) EddyTry : MultiShellKMatrix(dcsh) {} EddyCatch
  ExponentialKMatrix(const std::vector<DiffPara>&          dpars,
		     bool                                  dcsh=false);
  std::shared_ptr<KMatrix> Clone() const EddyTry { return(std::shared_ptr<ExponentialKMatrix>(new ExponentialKMatrix(*this))); } EddyCatch
  std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const;
private:
  unsigned int n_par(unsigned int ngrp) const { return(3*ngrp + (ngrp*ngrp - ngrp)); }
  unsigned int ij_to_parameter_index(unsigned int i,
					     unsigned int j, 
					     unsigned int n) const;
  void calculate_K_matrix(const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat,
			  NEWMAT::SymmetricMatrix&         K) const;
  void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
			   unsigned int                     ngrp,
			   const std::vector<double>&       thpar,
			   const NEWMAT::SymmetricMatrix&   angle_mat,
			   unsigned int                     gi,
			   unsigned int                     gj,
			   unsigned int                     off,
			   NEWMAT::SymmetricMatrix&         dK) const;
  NEWMAT::RowVector k_row(unsigned int                     indx,
			  bool                             excl,
			  const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat) const;
  std::vector<double> exp_hpar(unsigned int               ngrp,
			       const std::vector<double>& hpar) const;
  std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const;
  double sig_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)]); } EddyCatch
  double err_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)+2]); } EddyCatch
};

} 

#endif 

