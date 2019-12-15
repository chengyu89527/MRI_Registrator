/*  swefns.h
    Bryan Guillaume & Tom Nichols
    Copyright (C) 2019 University of Oxford  */

#ifndef ____swefns__
#define ____swefns__

#include "sweopts.h"
#include "newimage/newimageall.h"
#include "libprob.h"

using namespace NEWIMAGE;
using namespace NEWMAT;
using namespace arma;

namespace SWE {
	
	////////////////////////////////////////
	// Functions related to matrix algebra
	////////////////////////////////////////
	
	// compute the duplication matrix converting vech(A) into vec(A)
	mat duplicationMatrix(uword n);
	
	// compute the elimination indices which transform vec(A) into vech(A)
	uvec eliminationIndices(uword n);
	
	////////////////////////////////////////
	// check from vech(A) if A is positive semidefinite (note that the function can cbind several vech(A) into a matrix) using a row reduction approach and, if not, make vechA positive semidefinite by zeroing the negative eigenvalues
	////////////////////////////////////////
	
	void checkAndMakePSD(mat& vechA);
	
	////////////////////////////////////////
	// Temporary function allowing to combine arma::mat objects with NEWMAT::Matrix objects
	////////////////////////////////////////
	
	// convert a NEWMAT::Matrix into an arma::mat
	mat matrix2Mat(Matrix& newmatMatrix);
	
	// convert an arma::mat into a NEWMAT::Matrix
	Matrix mat2Matrix(mat& armaMat);

//	// convert a NEWMAT::Matrix into an arma::vec
//	vec matrix2vec(NEWMAT::Matrix& newmatMatrix);
//	
//	// convert a NEWMAT::Matrix into an arma::uvec
//	uvec matrix2uvec(NEWMAT::Matrix& newmatMatrix);
	
	vec columnVector2vec(NEWMAT::ColumnVector& newmatColumnVector);

	uvec columnVector2uvec(NEWMAT::ColumnVector& newmatColumnVector);

	
	////////////////////////////////////////
	// Template for the intersection of 2 arma vectors
	////////////////////////////////////////
	template< typename T, template <typename> class Type >
	Type<T> vintersection(Type<T> a, Type<T> b)
	{
		std::vector<T> o;
		std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(o)) ;
		std::reverse(o.begin(), o.end()) ;
		
		Type<T> result = conv_to< Type<T> >::from(o);
		return result;
	}
	
	////////////////////////////////////////
	// Function to read vest format and convert the info into a mat object
	////////////////////////////////////////
	
	// inspired from MISCMATHS::read_vest (needed to be changed to allow armadillo matric class)
	mat read_vest_swe(string p_fname);
	
	void T2ZP(mat& t, mat& z, mat& pValues, const mat& dof, const bool logP);
	
	void F2XP(mat& f, mat& x, mat& pValues, double dof1, const mat& dof2, const bool logP);
	void F2XZP(mat& f, mat& x, mat& z, mat& pValues, double dof1, const mat& dof2, const bool logP);
	
	////////////////////////////////////////
	// Class related voxelwise designs
	////////////////////////////////////////
	
	// VoxelwiseDesign is a class similar to RANDOMISE::VoxelwiseDesign, but using armadillo instead of newmat
	class VoxelwiseDesign
	{
	public:
		bool isSet;
		field<mat> EV;
		uvec  location;
		void setup(const vector<int>& voxelwise_ev_numbers, const vector<string>& voxelwise_ev_filenames, const volume<float>& mask, const int maximumLocation, const bool isVerbose);
		VoxelwiseDesign() { isSet=false; }
		mat adjustDesign(const mat& originalDesign, const int voxelNo);
	private:
	};

	////////////////////////////////////////
	// Class containing info about the SwE Model
	////////////////////////////////////////
	
	class SweModel
	{
	public:
		bool modified; // modifed SwE or classic SwE
		bool wb; // non parametric Wild Bootstrap or parametric inference
		bool voxelWiseDesign; // indicate if there are voxelwise covariates or not
		bool voxelwiseOutput; // output voxelwise corrected p-values
		bool doFOnly; // indicate if only F-contrasts are considered
		bool doTOnly; // indicate if only F-contrasts are considered
		bool verbose;
		bool logP; // if false, the p-value images will be 1-p; if true, the p-value images will be -log10(p)
		bool outputRaw; // output t & f statistic images
		bool outputEquivalent; // output z & x statitic images
		bool outputDof; // output effective number of degrees of freedom images
		bool outputUncorr; // output voxelwise uncorrected p-values
		bool outputGlm; // output pe, cope & varcope images
		bool outputTextNull; // output a text file with max. stats
		bool doTfce; // If true, will run a TFCE inference
		
		string out_fileroot;
		uword nVox; // total number of in-mask voxels
		uword nData; // total number of input images
		mat designMatrix;
		VoxelwiseDesign voxelWiseEV;
		mat pinvDesignMatrix;
		volume<float> mask;
		mat tc;
		mat fc;
		uword nContrasts;
		uword startFC;
		field<mat> fullFContrast;
		uvec sizeFcontrast;
		uvec nCovCFSweCF;
		field<rowvec> weightT; // link the contrasted SwE to the subject (or group) covariance matrices for t-contrasts
		field<mat> weightF; // link the contrasted SwE to the subject (or group) covariance matrices for F-contrasts
		uvec subject;
		uvec uSubject;
		uvec visit;
		uvec group;
		uvec uGroup;
		field<uvec> uVisit; // contains the unique visit category labels for each group (for modified SwE)
		uvec n_i; // number of visit per subject
		uvec nCovVis_i; // number of unique variances/covariances per subject
		uvec n_g; // number of visit per groups
		uvec nCovVis_g; // number of unique variances/covariances per group (for modified SwE)
		field<uvec> subjectIndex;
		field<umat> subjectCovIndex;
		vec subjectDof;
		uvec iGrDof; // indicates which separate design matrix each row (or observation) belongs to
		uvec indDiag;
		uvec indOffDiag;
		umat indCorrDiag;
		field<uvec> indDiagRes;
		field<umat> indOffDiagRes;
		field<mat> dofMat;
		uword nBootstrap;
		float clusterThresholdT;
		float clusterThresholdF;
		float clusterMassThresholdT;
		float clusterMassThresholdF;
		mat resamplingMult;
		float tfceHeight;
		float tfceExtent;
		bool tfceDeltaOverride;
		vec tfceDelta;
		uword tfceConnectivity;

		volume<float> nonConstantMask(volume4D<float>& data, const bool allOnes);
		void computeSubjectDof();
		void computeWeightClassic();
		void computeWeightModified();
		void setup(const sweopts& opts);
		void checkModel(const uword numberOfDataPoints); // check if the model looks alright
		void loadData(mat& data, const sweopts& opts);
		void adjustDofMat();
		void adjustDesign(const uword iVox); // adjust the design for the voxel iVox (only useful for voxelwise designs)
		mat computePe(const mat& data); // compute the parameter estimates
		mat computeAdjustedResiduals(const mat& data, const mat& beta);
		mat computeAdjustedRestrictedResiduals (const mat& data, const mat& beta, const mat& contrast);
		mat computeGroupCovarianceMatrices();
		void saveImage(mat& dataToSave, string filename); // cannot compile with const argument...To check later
		void saveImage(subview_row<double>& dataToSave, string filename);
		void computeVechCovVechV(mat& VechCovVechV, const mat& covVis_g, const uword g);
		void computeClusterStatsT(mat& score, uvec& clusterLabels, uvec& clusterExtentSizes);
		void computeClusterStatsT(mat& score, uvec& clusterLabels, vec& clusterMassSizes);
		void computeClusterStatsT(mat& pValue, mat& score, uvec& clusterLabels, vec& clusterMassSizes);
		void computeClusterStatsF(mat& score, uvec& clusterLabels, uvec& clusterExtentSizes);
		void computeClusterStatsF(mat& score, uvec& clusterLabels, vec& clusterMassSizes);
		void computeClusterStatsF(mat& pValue, mat& score, uvec& clusterLabels, vec& clusterMassSizes);
		void tfce(mat& zScore, mat& tfceScore, const uword indexContrast);
		void generateResamplingMatrix(); // generate resampling matrix for the WB
		void printMaxStats(const vec& maxStat, const string typeLabel, const string statLabel); // print maximum statistics
		void saveClusterWisePValues(const mat& oneMinusFwePClusterMass, const uvec& clusterMassLabels, const string typeLabel, const string statLabel);
		void computeSweT(mat& swe, const mat& residuals, const uword iContrast); 	// compute the modified SwE for t-contrasts
		void computeSweT(mat& swe, mat& dof, const mat& residuals, const uword iContrast); 	// compute the modified SwE for t-contrasts
		void computeSweF(mat& swe, const mat& residuals, const uword iContrast); 	// compute the modified SwE for f-contrasts
		void computeSweF(mat& swe, mat& dof, const mat& residuals, const uword iContrast); 	// compute the modified SwE for f-contrasts
		void computeT(const mat& data, mat& tstat, mat& cope, mat& varcope, const uword iContrast); 	// compute t-contrasts
		void computeT(const mat& data, mat& tstat, mat& cope, mat& varcope, mat& dof, const uword iContrast); // compute t-contrasts
		void computeF(const mat& data, mat& fstat, mat& cope, mat& varcope, const uword iContrast); // compute f-contrasts (no effective degrees of freedom adjustments)
		void computeF(const mat& data, mat& fstat, mat& cope, mat& varcope, mat& dof, const uword iContrast); // compute f-contrasts
		void computeSweAll(field<mat>& swe, const mat& residuals); // compute the SwE for all contrasts
		void computeSweAll(field<mat>& swe, mat& dof, const mat& residuals); // compute the SwE and dof for all contrasts
		void computeSweAll(field<mat>& swe, const mat& residuals, const uword iVox); // compute the SwE for all contrasts, but only voxel iVox
		void computeSweAll(field<mat>& swe, mat& dof, const mat& residuals, const uword iVox); // compute the SwE and dof for all contrasts, but only voxel iVox
		void computeStatsClassicSwe(const mat& data); // estimate the model & stats for the classic SwE (parametric inferences)
		void computeStatsModifiedSwe(const mat& data); // estimate the model & stats for the modified SwE (parametric inferences)
		void computeStatsWb(const mat& data); // estimate the model & statsf using the WB procedure

	private:
	};

} // end of namespace SWE

#endif /* defined(____swedesign__) */

