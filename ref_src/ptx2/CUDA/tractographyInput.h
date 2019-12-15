/*  tractographyInput.h

    Moises Hernandez-Fernandez  - FMRIB Image Analysis Group

    Copyright (C) 2015 University of Oxford  */

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

#include <CUDA/tractographyData.h>
#include <CUDA/options/options.h>
#include <csv.h>

#define ASCII 5
#define VTK   6
#define GIFTI 7

/** \brief This class contain the methods to read the input files passsed to GPU tractography.*/
class tractographyInput{

 public:

  tractographyInput();

  /// Method load all the necessary data from the input files to perform GPU Tractography
  void load_tractographyData( tractographyData&	tData,	
				volume<float>*&		m_prob,
				volume<float>*&		m_prob2,
				float**&		ConNet,
				float**&		ConNetb,
				int&			nRowsNet,
				int&			nColsNet,
				float**&		ConMat1,
				float**&		ConMat1b,
				int&			nRowsMat1,
				int&			nColsMat1,
				float**&		ConMat3,
				float**&		ConMat3b,
				int&			nRowsMat3,
				int&			nColsMat3,
				float*&			m_s2targets,
				float*&			m_s2targetsb,
				volume4D<float>*&	m_localdir);

  /// General Method to read a Surface file in ASCII, VTK or GIFTI format
  void load_mesh(	string&		filename,
			vector<float>&	vertices,	// all the vertices, same order than file
			vector<int>&	faces,		// all the faces, same order than file
			vector<int>& 	locs,		// used to store the id of a vertex in the Matrix. If -1, then vertex is non-activated	
			int&		nlocs,		// number of ids(vertices) in the Matrix
			bool		wcoords,	// save coordinates of the vertices in a file ?
			int		nroi,		// number of ROI to identify coordinates
			vector<float>&	coords);	// coordinates xyz of the vertices

  /// Method to read a surface file in ASCII format
  void load_mesh_ascii(	string&		filename,
			vector<float>&	vertices,
			vector<int>&	faces,					
			vector<int>&	locs,		
			int&		nlocs,	
			bool		wcoords,	
			int		nroi,	
			vector<float>&	coords);

  /// Method to read a surface file in VTK format
  void load_mesh_vtk(	string&		filename,
			vector<float>&	vertices,
			vector<int>&	faces,
			vector<int>&	locs,
			int&		nlocs,
			bool		wcoords,
			int		nroi,
			vector<float>&	coords);

  /// Method to read a surface file in GIFTI format
  void load_mesh_gifti(	string&		filename,
			vector<float>&	vertices,
			vector<int>&	faces,
			vector<int>&	locs,
			int&		nlocs,
			bool		wcoords,
			int		nroi,
			vector<float>&	coords);

  /// Method to read a Volume
  void load_volume(	string&		filename,
			int*		Ssizes,
			float*		Vout,
			int&		nlocs,
			bool		reset,
			bool		wcoords,
			int		nroi,
			vector<float>&	coords);
	
  /// Method to initialise the realtionship between voxels and triangles for a Surface
  void init_surfvol(	int*		Ssizes,
			Matrix&		mm2vox,	
			vector<float>&	vertices,
			int*		faces,
			int		sizefaces,	// number of faces this time (maybe there are several ROIs for the same mask)
			int		initfaces,	// number of faces in previos times
			vector<int>&	voxFaces,	// list of faces of all the voxels
			int*		voxFacesIndex,	// starting point of each voxel in the list
			vector<int>&	locsV);

  /// Method to find out what voxels are crossed by a triangle
  void csv_tri_crossed_voxels(float			tri[3][3],
				vector<ColumnVector>&	crossed);

  /// Method to read all the ROIs of a mask in the same structure: for stop and avoid masks
  void load_rois_mixed(string		filename,
			Matrix		mm2vox,
			float*		Sdims,
			int*		Ssizes,
			// Output
			MaskData&	matData);

  /// Method to read the ROIs of a mask in concatenated structures: for wtstop and waypoints masks
  void load_rois(// Input
		 string		filename,
		 Matrix		mm2vox,
		 float*		Sdims,
		 int*		Ssizes,
		 int		wcoords,
		 volume<float>& refVol, 
		 // Output
		 MaskData&	matData,
		 Matrix&	coords);

  /// Same than load_rois but it includes the initialisation of the rows (including triangles) of Matrix1
  void  load_rois_matrix1(	tractographyData&	tData,
				// Input
				string			filename,
				Matrix			mm2vox,
				float*			Sdims,
				int*			Ssizes,
				bool			wcoords,
				volume<float>& 		refVol, 
				// Output
				MaskData&		data,
				Matrix&			coords);

  /// Method to load the seeds. Can be defined by volumes and/or by surfaces
  int load_seeds_rois(tractographyData&	tData,
			string			seeds_filename,
			string			ref_filename,
			float*			Sdims,
			int*			Ssizes,
			int			convention,
			float*&			seeds,
			int*&			seeds_ROI,
			Matrix&			mm2vox,
			float*			vox2mm,
			volume<float>*&		m_prob,
			bool 			initialize_m_prob,
			volume<float>*&		m_prob2,
			bool			initialize_m_prob2,
			volume4D<float>*&	m_localdir,
			volume<float>&          refVol);

  /// Method to set the transformation: voxel to milimeters
  void set_vox2mm(int		convention,
		  float*        Sdims,
		  int*		Ssizes,
		  volume<float>	vol,
		  Matrix&       mm2vox,   // 4x4
		  float*        vox2mm);  // 4x4

};

