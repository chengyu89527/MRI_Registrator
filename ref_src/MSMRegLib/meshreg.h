/*  meshreg.h

    Emma Robinson, FMRIB Image Analysis Group

    Copyright (C) 2012 University of Oxford  */

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
/* Class for overseeing the running of each resolution level of the registration */
#ifndef meshreg_h
#define meshreg_h


#include "meshmodify.h"
#include "miscmaths/SpMat.h"


#define CORLIM  1E-10  // when calculating corrzero consider any correlations below this value to be ZERO
#define D_DIST 0.25   // sampling distance along tangent plane for computing derivatives - 0.25 is what Bruce uses for FreeSurfer

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

namespace MESHREG {

  struct spcoord
  {
    double theta;
    double phi;
    
    int ind;
  };
  
  struct SortPhi
  {
    bool operator() (const spcoord& m1, const spcoord& m2)
    { return m1.phi < m2.phi; }
  };
  
  struct SortAscTheta
  {
    bool operator() (const spcoord& m1, const spcoord& m2)
    { return m1.theta < m2.theta; }
  };
  
  struct SortDecTheta
  {
    bool operator() (const spcoord& m1, const spcoord& m2)
    {return m1.theta > m2.theta; }
  };
  
  
  class MeshReg: public MeshModify {
    
  private:
 
    newmesh icoref;
    newmesh ANAT_RES;
    
    bool _isaffine;

    boost::shared_ptr<affineMeshCF> affinecf;

  protected:
    int _level;
    boost::shared_ptr<SRegDiscreteModel> MODEL;
  public:
    
    // Constructors
    inline MeshReg(){_isaffine=false; _level=0;};
    
    //MeshReg(int);
    
    // Destructor
    inline ~MeshReg(){};
    
    //////////////// INITIALIZATION ////////////////////
    //// loops over steps/resolution levels
    void run_multiresolutions(const int &, const double &, const string &);
    ///// initialises featurespace and cost function model for each step (either a single resolution level of the discrete opt or an Affine step)
    void Initialize_level(int);
    /// downsample anatomy
    newmesh  resample_anatomy(newmesh, vector<map<int,double> > &, vector<vector<int > > &,int );
    ///// additionally resamples the costfunction weighting to match the downsampled data grids for that resolution leve;
    Matrix downsample_cfweighting(const double &, NEWMESH::newmesh, boost::shared_ptr<NEWMESH::newmesh>, boost::shared_ptr<NEWMESH::newmesh>);
    
    //////////////// RUN //////////////////////
    void Evaluate();
    void Transform(const string &);
    void saveTransformedData(const double &,const string &);

    ///// projects warp from a lower resolution level to the next one
    NEWMESH::newmesh project_CPgrid(NEWMESH::newmesh,NEWMESH::newmesh, int num=0);
    /////// runs all iterations of the discrete optimisation for each resolution level
    void run_discrete_opt(NEWMESH::newmesh &);
   
   
  };

  
  
  
}

  

#endif

