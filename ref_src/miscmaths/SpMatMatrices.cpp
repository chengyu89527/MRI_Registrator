//
//  Declarations for functions generating specialised 
//  sparse matrices of type SpMat
//
//  SpMatMatrices.h
//
//  Declares global functions for generating specialised
//  sparse matrices of type SpMat
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2019 University of Oxford
//
/*  Part of FSL - FMRIB's Software Library
    http://www.fmrib.ox.ac.uk/fsl
    fsl@fmrib.ox.ac.uk

    Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
    Imaging of the Brain), Department of Clinical Neurology, Oxford
    University, Oxford, UK


    LICENCE

    FMRIB Software Library, Release 6.0 (c) 2018, The University of
    Oxford (the "Software")

    The Software remains the property of the Oxford University Innovation
    ("the University").

    The Software is distributed "AS IS" under this Licence solely for
    non-commercial use in the hope that it will be useful, but in order
    that the University as a charitable foundation protects its assets for
    the benefit of its educational and research purposes, the University
    makes clear that no condition is made or to be implied, nor is any
    warranty given or to be implied, as to the accuracy of the Software,
    or that it will be suitable for any particular purpose or for use
    under any specific conditions. Furthermore, the University disclaims
    all responsibility for the use which is made of the Software. It
    further disclaims any liability for the outcomes arising from using
    the Software.

    The Licensee agrees to indemnify the University and hold the
    University harmless from and against any and all claims, damages and
    liabilities asserted by third parties (including claims for
    negligence) which arise directly or indirectly from the use of the
    Software or the sale of any products based on the Software.

    No part of the Software may be reproduced, modified, transmitted or
    transferred in any form or by any means, electronic or mechanical,
    without the express permission of the University. The permission of
    the University is not required if the said reproduction, modification,
    transmission or transference is done without financial return, the
    conditions of this Licence are imposed upon the receiver of the
    product, and all original and amended source code is included in any
    transmitted product. You may be held legally responsible for any
    copyright infringement that is caused or encouraged by your failure to
    abide by these terms and conditions.

    You are not permitted under this Licence to use this Software
    commercially. Use for which any financial return is received shall be
    defined as commercial use, and includes (1) integration of all or part
    of the source code or the Software into a product for sale or license
    by or on behalf of Licensee to third parties or (2) use of the
    Software or any derivative of it for research with the final aim of
    developing software products for sale or license to a third party or
    (3) use of the Software or any derivative of it for research with the
    final aim of developing non-software products for sale or license to a
    third party, or (4) use of the Software to provide any service to an
    external organisation for which payment is received. If you are
    interested in using the Software commercially, please contact Oxford
    University Innovation ("OUI"), the technology transfer company of the
    University, to negotiate a licence. Contact details are:
    fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.*/

#include <vector>
#include "newmat.h"
#include "SpMat.h"
#include "SpMatMatrices.h"

namespace MISCMATHS {

/*!
 * Global function that creates and returns a symmetric
 * Toeplitz matrix with dimensions col.Nrows() x col.Nrows() and where the 
 * first column is given by col and all subsequent columns are translated
 * and shifted versions of that column.
 * \return A sparse symmetric Toeplitz matrix
 * \param[in] col First column of matrix
 */
MISCMATHS::SpMat<float> SparseSymmetricToeplitz(const NEWMAT::ColumnVector& col)
{
  unsigned int mn = static_cast<unsigned int>(col.Nrows());
  unsigned int nnz = 0; // No of non-zeros per column
  for (unsigned int i=0; i<mn; i++) nnz += (col(i+1) == 0) ? 0 : 1;
  std::vector<unsigned int> indx(nnz);
  std::vector<float> val(nnz);
  {
    unsigned int i = 0; unsigned int j = 0;
    for (i=0, j=0; i<mn; i++) if (col(i+1) != 0) { indx[j] = i; val[j++] = static_cast<float>(col(i+1)); }
  }
  unsigned int *irp = new unsigned int[nnz*mn];
  unsigned int *jcp = new unsigned int[mn+1];
  double *sp = new double[nnz*mn];
  unsigned int irp_cntr = 0;
  for (unsigned int col=0; col<mn; col++) {
    jcp[col] = irp_cntr;
    for (unsigned int r=0; r<nnz; r++) {
      irp[irp_cntr] = indx[r];
      sp[irp_cntr++] = val[r];
      indx[r] = (indx[r] == mn-1) ? 0 : indx[r]+1;      
    }
  }
  jcp[mn] = irp_cntr;
  MISCMATHS::SpMat<float> tpmat(mn,mn,irp,jcp,sp);
  delete [] irp; delete [] jcp; delete [] sp;
  return(tpmat);
}

/*!
 * Global function that creates and returns a symmetric matrix with dimensions 
 * prod(isz) x prod(isz) and which represent an approximate Hessian for
 * Bending energy. It is approximate because it only considers the straight
 * second derivatives.
 * \return A sparse symmetric Hessian of Bending Energy
 * \param[in] isz 3 element vector specifying matrix size of image
 * \param[in] vxs 3 element vector with voxel size in mm
 * \param[in] bc Boundary condition (PERIODIC or MIRROR)
 */
MISCMATHS::SpMat<float> Sparse3DBendingEnergyHessian(const std::vector<unsigned int>& isz, 
						     const std::vector<double>&       vxs,
						     MISCMATHS::BoundaryCondition     bc) 
{
  unsigned int mn = isz[0]*isz[1]*isz[2];
  unsigned int *irp = new unsigned int[3*mn]; // Worst case, might be slightly smaller
  unsigned int *jcp = new unsigned int[mn+1];
  double *sp = new double[3*mn];
  // x-direction
  unsigned int irp_cntr = 0;
  double sf = 1 / vxs[0]; // Let us scale all directions to mm^{-1}
  for (unsigned int k=0; k<isz[2]; k++) {
    for (unsigned int j=0; j<isz[1]; j++) {
      for (unsigned int i=0; i<isz[0]; i++) {
	jcp[k*isz[0]*isz[1] + j*isz[0] + i] = irp_cntr;
	if (bc == MISCMATHS::PERIODIC) {
	  if (i==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0];
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + isz[0]-1;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	  else if (i==isz[0]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0];
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i + 1;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
	else if (bc == MISCMATHS::MIRROR) {
	  if (i==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0];
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + 1;
	    sp[irp_cntr++] = -2.0 * sf;
	  }
	  else if (i==isz[0]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i + 1;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
      }
    }
  }
  jcp[mn] = irp_cntr;
  MISCMATHS::SpMat<float> At(mn,mn,irp,jcp,sp);
  MISCMATHS::SpMat<float> AtA = At * At.t();

  // y-direction
  irp_cntr = 0;
  sf = 1 / vxs[1]; // Let us scale all directions to mm^{-1}
  for (unsigned int k=0; k<isz[2]; k++) {
    for (unsigned int j=0; j<isz[1]; j++) {
      for (unsigned int i=0; i<isz[0]; i++) {
	jcp[k*isz[0]*isz[1] + j*isz[0] + i] = irp_cntr;
	if (bc == MISCMATHS::PERIODIC) {
	  if (j==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + isz[0] + i;
	    sp[irp_cntr++] = -1.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + (isz[1]-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0;
	  }
	  else if (j==isz[1]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + i;
	    sp[irp_cntr++] = -1.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0;	    
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j+1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
	else if (bc == MISCMATHS::MIRROR) {
	  if (j==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + isz[0] + i;
	    sp[irp_cntr++] = -2.0;
	  }
	  else if (j==isz[1]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -2.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0;	    
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j+1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
      }
    }
  }
  jcp[mn] = irp_cntr;
  At = MISCMATHS::SpMat<float>(mn,mn,irp,jcp,sp);
  AtA += At * At.t();

  // z-direction
  irp_cntr = 0;
  sf = 1 / vxs[2]; // Let us scale all directions to mm^{-1}
  for (unsigned int k=0; k<isz[2]; k++) {
    for (unsigned int j=0; j<isz[1]; j++) {
      for (unsigned int i=0; i<isz[0]; i++) {
	jcp[k*isz[0]*isz[1] + j*isz[0] + i] = irp_cntr;
	if (bc == MISCMATHS::PERIODIC) {
	  if (k==0) {
	    irp[irp_cntr] = j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = (isz[2]-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	  else if (k==isz[2]-1) {
	    irp[irp_cntr] = j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	  }
	  else {
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	    irp[irp_cntr] = (k+1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
	else if (bc == MISCMATHS::MIRROR) {
	  if (k==0) {
	    irp[irp_cntr] = j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -2.0 * sf;
	  }
	  else if (k==isz[2]-1) {
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	  }
	  else {
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;	    
	    irp[irp_cntr] = (k+1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
      }
    }
  }
  jcp[mn] = irp_cntr;
  At = MISCMATHS::SpMat<float>(mn,mn,irp,jcp,sp);
  AtA += At * At.t();
  delete [] irp; delete [] jcp; delete [] sp;
  return(AtA);
}

} // End namespace MISCMATHS
