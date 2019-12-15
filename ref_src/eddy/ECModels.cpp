// Declarations of classes that implements a hirearchy
// of models for fields from eddy currents induced by
// diffusion gradients.
// 
// ECModels.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
#include "newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "topup/topup_file_io.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"

using namespace EDDY;

NEWMAT::Matrix ScanMovementModel::ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan) const EddyTry
{ 
  if (_order) return(TOPUP::MovePar2Matrix(get_zero_order_mp(),scan)); 
  else return(TOPUP::MovePar2Matrix(_mp,scan)); 
} EddyCatch

NEWMAT::Matrix ScanMovementModel::ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan, 
							unsigned int                   grp, 
							unsigned int                   ngrp) const EddyTry
{
  if (grp>=ngrp) throw EddyException("ScanMovementModel::ForwardMovementMatrix: grp has to be smaller than ngrp");
  NEWMAT::ColumnVector gmp = get_gmp(grp,ngrp);
  return(TOPUP::MovePar2Matrix(gmp,scan)); 
} EddyCatch

NEWMAT::Matrix ScanMovementModel::RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan,
								  const std::vector<unsigned int>&     rindx) const EddyTry
{ 
  NEWMAT::ColumnVector rmp;
  if (_order) rmp = get_zero_order_mp(); 
  else rmp = _mp; 
  for (unsigned int i=0; i<rindx.size(); i++) {
    if (rindx[i] > 5) throw EddyException("ScanMovementModel::RestrictedForwardMovementMatrix: rindx has to be less than 6");
    else rmp(rindx[i]+1) = 0.0;
  }
  return(TOPUP::MovePar2Matrix(rmp,scan)); 
} EddyCatch

NEWMAT::Matrix ScanMovementModel::RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan, 
								  unsigned int                         grp, 
								  unsigned int                         ngrp,
								  const std::vector<unsigned int>&     rindx) const EddyTry
{
  if (grp>=ngrp) throw EddyException("ScanMovementModel::RestrictedForwardMovementMatrix: grp has to be smaller than ngrp");
  NEWMAT::ColumnVector rgmp = get_gmp(grp,ngrp);
  for (unsigned int i=0; i<rindx.size(); i++) {
    if (rindx[i] > 5) throw EddyException("ScanMovementModel::RestrictedForwardMovementMatrix: rindx has to be less than 6");
    else rgmp(rindx[i]+1) = 0.0;
  }
  return(TOPUP::MovePar2Matrix(rgmp,scan)); 
} EddyCatch

NEWIMAGE::volume<float> LinearScanECModel::ECField(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  NEWIMAGE::volume<float> field = scan;
  field = _ep(4);
  for (int k=0; k<field.zsize(); k++) {
    float zcomp = _ep(3)*field.zdim()*(k-(field.zsize()-1)/2);
    for (int j=0; j<field.ysize(); j++) {
      float ycomp = _ep(2)*field.ydim()*(j-(field.ysize()-1)/2);
      for (int i=0; i<field.xsize(); i++) {
        field(i,j,k) += _ep(1)*field.xdim()*(i-(field.xsize()-1)/2) + ycomp + zcomp;
      }
    }
  }
  return(field);
} EddyCatch

NEWIMAGE::volume<float> QuadraticScanECModel::ECField(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  NEWIMAGE::volume<float> field = scan;
  field = _ep(10); 
  for (int k=0; k<field.zsize(); k++) {
    double z = field.zdim()*(k-(field.zsize()-1)/2);
    double zcomp = _ep(3)*z;
    double z2comp = _ep(6)*z*z;
    for (int j=0; j<field.ysize(); j++) {
      double y = field.ydim()*(j-(field.ysize()-1)/2);
      double ycomp = _ep(2)*y;
      double y2comp = _ep(5)*y*y;
      double yzcomp = _ep(9)*y*z;
      for (int i=0; i<field.xsize(); i++) {
        double x = field.xdim()*(i-(field.xsize()-1)/2);
        double xcomp = _ep(1)*x;
        double x2comp = _ep(4)*x*x;
	double xycomp = _ep(7)*x*y;
	double xzcomp = _ep(8)*x*z;
        field(i,j,k) += xcomp + ycomp + zcomp + x2comp + y2comp + z2comp + xycomp + xzcomp + yzcomp;
      }
    }
  }
  return(field);
} EddyCatch

NEWIMAGE::volume<float> CubicScanECModel::ECField(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  NEWIMAGE::volume<float> field = scan;
  field = _ep(20); 
  for (int k=0; k<field.zsize(); k++) {
    double z = field.zdim()*(k-(field.zsize()-1)/2);
    double z2 = z*z;
    double zcomp = _ep(3)*z;
    double z2comp = _ep(6)*z2;
    double z3comp = _ep(12)*z*z2;
    for (int j=0; j<field.ysize(); j++) {
      double y = field.ydim()*(j-(field.ysize()-1)/2);
      double y2 = y*y;
      double ycomp = _ep(2)*y;
      double y2comp = _ep(5)*y2;
      double y3comp = _ep(11)*y*y2;
      double yzcomp = _ep(9)*y*z;
      double y2zcomp = _ep(17)*y2*z;
      double yz2comp = _ep(19)*y*z2;
      for (int i=0; i<field.xsize(); i++) {
        double x = field.xdim()*(i-(field.xsize()-1)/2);
	double x2 = x*x;
        double xcomp = _ep(1)*x;
        double x2comp = _ep(4)*x2;
        double x3comp = _ep(10)*x*x2;
	double xycomp = _ep(7)*x*y;
	double xzcomp = _ep(8)*x*z;
	double x2ycomp = _ep(13)*x2*y;
	double x2zcomp = _ep(14)*x2*z;
	double xyzcomp = _ep(15)*x*y*z;
	double xy2comp = _ep(16)*x*y2;
	double xz2comp = _ep(18)*x*z2;
        field(i,j,k) += xcomp + ycomp + zcomp + x2comp + y2comp + z2comp + xycomp + xzcomp + yzcomp;
        field(i,j,k) += x3comp + y3comp + z3comp + x2ycomp + x2zcomp + xyzcomp + xy2comp + y2zcomp + xz2comp + yz2comp;
      }
    }
  }
  return(field);
} EddyCatch


