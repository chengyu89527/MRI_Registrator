/*  swefns.cc
    Bryan Guillaume & Tom Nichols
    Copyright (C) 2019 University of Oxford  */

// define the type of cluster connectivity (26 like randomise)
#define CLUST_CON 26

#include "swefns.h"

// declare labels for output filenames
const string LABEL_PE = "glm_pe";
const string LABEL_COPE = "glm_cope";
const string LABEL_VARCOPE = "glm_varcope";
const string LABEL_LOG = "l";
const string LABEL_P = "p";
const string LABEL_FWE_CORRECTED = "corr";
const string LABEL_DOF = "dof";
const string LABEL_TSTAT = "tstat";
const string LABEL_FSTAT = "fstat";
const string LABEL_EQUIVALENT_Z = "z";
const string LABEL_EQUIVALENT_CHISQUARE = "x";
const string LABEL_VOX = "vox";
const string LABEL_CLUSTERE = "clustere";
const string LABEL_CLUSTERM = "clusterm";
const string LABEL_TFCE = "tfce";
const double TOLERANCE = 0.0001L;

namespace SWE {

	////////////////////////////////////////
	// Functions related to matrix algebra
	////////////////////////////////////////
	
	// compute the duplication matrix converting vech(A) into vec(A)
	mat duplicationMatrix(uword n)
	{
		mat dupl = zeros<mat>(n * n, n * (n +1) / 2);
		uword it = 0;
		for (uword j = 0; j < n; j++) {
			dupl(j * n + j, it + j) = 1;
	  for (uword i = j + 1; i < n; i++) {
			dupl(j * n + i, it + i) = 1;
			dupl(i * n + j, it + i) = 1;
		}
	  it += n - j - 1;
		}
		return dupl;
	}
	
	// compute the elimination indices which transform vec(A) into vech(A)
	uvec eliminationIndices(uword n)
	{
		uvec elimIndices(n * (n +1) / 2);
		uword it = 0;
		for (uword i = 0; i < n; i++){
			elimIndices(span(it, it + n - 1 - i)) = linspace<uvec>((n+1) * i, (i+1) * n - 1, n - i);
			it += n - i;
		}
		return elimIndices;
	}
	
	////////////////////////////////////////
	// Function checking from vech(A) if A is positive semidefinite (note that the function can cbind several vech(A) into a matrix) using a row reduction approach and, if not, make vechA positive semidefinite by zeroing the negative eigenvalues
	////////////////////////////////////////

	void checkAndMakePSD(mat& vechA) // note that I could return notPSD to know where non-postive semidefinitiveness occured
	{
		// first detect which A is not positive semidefinite
		urowvec notPSD = vechA.row(0) < 0; // indicates which A is not positive semidefinite
		mat tmp = vechA.cols(find(notPSD == 0));
		mat tmp2;
		uword nVis = (-1 + sqrt(1  + 8 * vechA.n_rows)) / 2;
		uword it;
		urowvec index;
		
		for (uword n = nVis; n > 1; n--){
			it = n;
			for (uword i = 1; i < n; i++) {
				tmp2 = tmp.row(i) / tmp.row(0);
				for (uword j = 0; j < n - i; j++, it++) {
					tmp.row(it) = tmp.row(it) - tmp2 % tmp.row(j + i);
				}
			}
			tmp = tmp.tail_rows(tmp.n_rows - n);
			urowvec tmp3 = all(tmp.head_rows(n - 1));
			index = tmp.row(0) < 0 || (tmp.row(0) == 0 && (1 - all(tmp.head_rows(n - 1))));
			notPSD(find(notPSD == 0)) = index;
			tmp = tmp.cols(find(index == 0));
		}
		notPSD(find(notPSD == 0)) = tmp.row(0) == 0;
		
		// convert into indices
		uvec indicesNotPSD = find(notPSD == 1);
		
		// make the non-positive semidefinite A positive semidefinite
		tmp.zeros(nVis, nVis);
		vec eigval(nVis);
		mat eigvec(nVis, nVis);
		uvec ind;
		uvec ind2 = eliminationIndices(nVis);
		for (uword i = 0; i < indicesNotPSD.n_elem; i++) {
			tmp(ind2) = vechA.col(indicesNotPSD(i));
			eig_sym(eigval, eigvec, symmatl(tmp));
			ind = find(eigval < 0);
			eigval(ind) = zeros<vec>(ind.n_elem);
			tmp = eigvec * diagmat(eigval) * eigvec.t();
			vechA.col(indicesNotPSD(i)) = tmp(ind2);
		}
	}
	
	////////////////////////////////////////
	// Temporary functions allowing to combine arma::mat objects with NEWMAT::Matrix objects (might be changed later)
	////////////////////////////////////////
	
	// convert a NEWMAT::Matrix into an arma::mat
	mat matrix2Mat(NEWMAT::Matrix& newmatMatrix)
	{
//		mat armaMat(newmatMatrix.Ncols(), newmatMatrix.Nrows());
//		mat::iterator a = armaMat.begin();
//		mat::iterator b = armaMat.end();
//		Real* p = newmatMatrix.Store(); // fetch the pointer to the array of elements
//		for(mat::iterator i = a; i != b; ++i, ++p){
//			*i = *p;
//		}
//		return armaMat.t();
		return newmatMatrix.get_at_ref();
	}
	
	// convert an arma::mat into a NEWMAT::Matrix
	Matrix mat2Matrix(mat& armaMat)
	{
//		Matrix newmatMatrix(armaMat.n_cols, armaMat.n_rows);
//		mat::iterator a = armaMat.begin();
//		mat::iterator b = armaMat.end();
//		Real* p = newmatMatrix.Store(); // fetch the pointer to the array of elements
//		for(mat::iterator i = a; i != b; ++i, ++p){
//			*p = *i;
//		}
//		return newmatMatrix.t();
		return Matrix(armaMat);
	}

//	// convert a NEWMAT::Matrix into an arma::vec
//	vec matrix2vec(NEWMAT::Matrix& newmatMatrix)
//	{
//		vec armaVec(newmatMatrix.Ncols() * newmatMatrix.Nrows());
//		vec::iterator a = armaVec.begin();
//		vec::iterator b = armaVec.end();
//		Real* p = newmatMatrix.Store(); // fetch the pointer to the array of elements
//		for(vec::iterator i = a; i != b; ++i, ++p){
//			*i = *p;
//		}
//		return armaVec;
//	}
//	
//	// convert a NEWMAT::Matrix into an arma::uvec
//	uvec matrix2uvec(NEWMAT::Matrix& newmatMatrix)
//	{
//		uvec armaUvec(newmatMatrix.Ncols() * newmatMatrix.Nrows());
//		uvec::iterator a = armaUvec.begin();
//		uvec::iterator b = armaUvec.end();
//		Real* p = newmatMatrix.Store(); // fetch the pointer to the array of elements
//		for(uvec::iterator i = a; i != b; ++i, ++p){
//			*i = *p;
//		}
//		return armaUvec;
//	}
	
	// convert a NEWMAT::columnVector into an arma::vec
	vec columnVector2vec(NEWMAT::ColumnVector& newmatColumnVector)
	{
//		vec armaVec(newmatColumnVector.Nrows());
//		vec::iterator a = armaVec.begin();
//		vec::iterator b = armaVec.end();
//		Real* p = newmatColumnVector.Store(); // fetch the pointer to the array of elements
//		for(vec::iterator i = a; i != b; ++i, ++p){
//			*i = *p;
//		}
//		return armaVec;
		return newmatColumnVector.get_at_ref();
	}
	
	// convert a NEWMAT::columnVector into an arma::uvec
	uvec columnVector2uvec(NEWMAT::ColumnVector& newmatColumnVector)
	{
//		uvec armaUvec(newmatColumnVector.Nrows());
//		uvec::iterator a = armaUvec.begin();
//		uvec::iterator b = armaUvec.end();
//		Real* p = newmatColumnVector.Store(); // fetch the pointer to the array of elements
//		for(uvec::iterator i = a; i != b; ++i, ++p){
//			*i = *p;
//		}
//		return armaUvec;
		return conv_to<uvec>::from(newmatColumnVector.get_at_ref().col(0));
	}
	
	////////////////////////////////////////
	// Function to read vest format and convert the info into a mat object
	////////////////////////////////////////
	
	// inspired from MISCMATHS::read_vest (needed to be changed to allow armadillo matric class)
	mat read_vest_swe(string p_fname)
	{
		ifstream in;
		in.open(p_fname.c_str(), ios::in);
		
		if(!in) throw Exception(string("Unable to open " + p_fname).c_str());
		
		uword numWaves = 0;
		uword numPoints = 0;
		
		string str;
		
		while(true)
		{
			if(!in.good()) throw Exception(string(p_fname+" is not a valid vest file").c_str());
			in >> str;
			if(str == "/Matrix")
				break;
			else if(str == "/NumWaves")
			{
				in >> numWaves;
			}
			else if(str == "/NumPoints" || str == "/NumContrasts")
			{
				in >> numPoints;
			}
		}
		
		mat p_mat(numPoints, numWaves);
		
		for(uword i = 0; i < numPoints; i++)
		{
			for(uword j = 0; j < numWaves; j++)
			{
				if (!in.eof()) in >> ws >> p_mat(i,j) >> ws;
				else throw Exception(string(p_fname+" has insufficient data points").c_str());
			}
		}
		
		in.close();
		
		return p_mat;
	}

	////////////////////////////////////////
	// Detect non constant voxels if required (inspired from randomise)
	////////////////////////////////////////
	
	volume<float> SweModel::nonConstantMask(volume4D<float>& data, const bool allOnes)
	{
		volume<float> nonConstantMask(data.xsize(), data.ysize(), data.zsize());
		nonConstantMask.copyproperties(data[0]);
		if (allOnes) {
			nonConstantMask = 1;
			return nonConstantMask;
		}

		// setup some useful variables
		uword nSeparableX = iGrDof.max() + 1;
		field<uvec> indSubjInSeparableX(nSeparableX);
		uvec sizeSeparableX(nSeparableX);
		for(int g=0; g<nSeparableX; g++) {
			indSubjInSeparableX(g) = find(iGrDof==g);
			sizeSeparableX(g) = indSubjInSeparableX(g).n_elem;
		}

		nonConstantMask = 1;
		bool ok;
		for(int z=0; z<data.zsize(); z++) {
			for(int y=0; y<data.ysize(); y++) {
				for(int x=0; x<data.xsize(); x++) {
					// if there is a nan, remove the voxel (might need to be adapted if there is a voxelwise design specified)
					ok = true;
					for(int t=0; t < data.tsize(); t++) {
						if (isnan(data(x,y,z,t))) { 
							ok = false;
							break;
						}
					}
					// look for constant voxels into each separable design matrix
					if (ok) {
						for(uword g = 0; g < nSeparableX; g++) {
							ok = true;
							// skip separable design matrices composed of a single row
							if ( sizeSeparableX(g) > 1 ) {
								// this voxel is not ok if there is no difference in this separable design matrix
								ok = false;
								for(int t=1; t < sizeSeparableX(g); t++) {
									if (data(x,y,z, indSubjInSeparableX(g)(t)) != data(x,y,z, indSubjInSeparableX(g)(0))) {
										ok = true;
										break;
									}
								}
								// if this is not ok for this separable design matrix, no need to continue
								if (!ok) break;
							}
						}
					}
					
					// For the modified SwE, for each "homogeneous" group, this voxel is not ok if the data is contant over subject for each visit category
					if (ok & modified) {
						// check each visit of each group
						for (uword it = 0; it < indDiagRes.n_elem; it++)
						{
							ok = true;
							if (indDiagRes(it).n_elem > 1) {
								ok = false;
								for(int t=1; t < indDiagRes(it).n_elem; t++) {
									if (data(x,y,z, indDiagRes(it)(t)) != data(x,y,z, indDiagRes(it)(0))) {
										ok = true;
										break;
									}
								}
								// if not ok for one visit, no need to continue
								if (!ok) break;
							}
						}
					}

					// For the modified SwE, for each "homogeneous" group, this voxel is not ok if the data is contant between 2 visits for all subjects  
					if (ok & modified) {					
						for (uword it = 0; it < indOffDiagRes.n_elem; it++)
						{
							// check each pair of visit
							ok = true;
							if (indOffDiagRes(it).n_rows > 0) {
								ok = false;
								for(int t=0; t < indOffDiagRes(it).n_rows; t++) {
									if (data(x,y,z, indOffDiagRes(it)(t,1)) != data(x,y,z, indOffDiagRes(it)(t,0))) {
										ok = true;
										break;
									}
								}
								// if not ok for one pair of visit, no need to continue
								if (!ok) break;
							}
						}
					}

					if (!ok)
						nonConstantMask(x,y,z) = 0;
				}
			}
		}
		return nonConstantMask;
	}
	
	////////////////////////////////////////
	// Transform t-scores into equivalent z-scores and fill up a matrix of 1 - p-values or -log10(p-values)
	////////////////////////////////////////

	void T2ZP(mat& t, mat& z, mat& pValues, const mat& dof, const bool logP)
	{
		mat::iterator a = t.begin();
		mat::iterator b = t.end();
		mat::iterator c = pValues.begin();
		mat::const_iterator d = dof.begin();
		mat::iterator e = z.begin();
		
		for(mat::iterator i = a; i != b; i++, c++, d++, e++) {
			if (*i < 0) { // negative t-score
				*c = 0.5 * MISCMATHS::incbet(0.5 * *d, 0.5, *d / (*d + *i * *i)); // 1-p < 0.5
				*e = MISCMATHS::ndtri(*c); // eqZ
				if (logP) { // -log10(p)
					*c = -log10(1 - *c);
				}
			}
			else if (*i > 0) { // positive t-score
				*c = 0.5 * MISCMATHS::incbet(0.5 * *d, 0.5, *d / (*d + *i * *i)); // p < 0.5
				*e = - MISCMATHS::ndtri(*c); // eqZ
				if (logP) { // -log10(p)
					*c = -log10(*c);
				}
				else { // 1-p
					*c = 1 - *c; // 1-p > 0.5
				}
					
			}
			else { // t-score = 0
				if (logP) { // -log10(p)
					*c = -log10(0.5);
				}
				else { // 1-p
					*c = 0.5;
				}
				*e = 0;
			}
		}
	}
	
	////////////////////////////////////////
	// transform F-scores into equivalent chi-scores and fill up a matrix of 1 - p-values or -log10(p-values)
	////////////////////////////////////////
	
	void F2XP(mat& f, mat& x, mat& pValues, double dof1, const mat& dof2, const bool logP)
	{
		mat::iterator a = f.begin();
		mat::iterator b = f.end();
		mat::iterator c = pValues.begin();
		mat::const_iterator d = dof2.begin();
		mat::iterator e = x.begin();
		
		double tmp;
		
		for(mat::iterator i = a; i != b; i++, c++, d++, e++) {
			
			if(dof1 == 1) {
				*c = MISCMATHS::incbet(0.5 * *d, 0.5, *d / (*d + *i)); // p (may not be precise for large p, but this is not important)
				*e = MISCMATHS::ndtri(0.5 * *c); // -eqZ
				*e = *e * *e; // eqX
				if (logP) { // -log10(p)
					*c = -log10(*c);
				}
				else {
					*c = 1 - *c; // 1-p
				}
			}
			else {
				if (*i > 0) {
					tmp = *d - dof1 + 1;
					*c = MISCMATHS::incbet(0.5 * tmp, 0.5 * dof1, tmp / (tmp + dof1 * *i)); // p(may not be precise for large p, but this is not important)
					*e = MISCMATHS::ndtri(0.5 * *c); // -eqZ
					*e = *e * *e; // eqX
					if (logP) { // -log10(p)
						*c = -log10(*c);
					}
					else {
						*c = 1 - *c; // 1-p
					}
				}
				else {
					*c = 0;
					*e = 0;
				}
			}
		}
	}
	
  ///////////////////////////////////////
	// transform F-scores into equivalent chi-scores, equivalent Z-scores and fill up a matrix of 1 - p-values or -log10(p-values)
	////////////////////////////////////////
	
	void F2XZP(mat& f, mat& x, mat& z, mat& pValues, double dof1, const mat& dof2, const bool logP)
	{
		mat::iterator a = f.begin();
		mat::iterator b = f.end();
		mat::iterator c = pValues.begin();
		mat::const_iterator d = dof2.begin();
		mat::iterator e = x.begin();
		mat::iterator g = z.begin();
		double tmp;
		for(mat::iterator i = a; i != b; i++, c++, d++, e++, g++) {
			
			if(dof1 == 1) {
				*c = MISCMATHS::incbet(0.5 * *d, 0.5, *d / (*d + *i)); // p (may not be precise for large p, but this is not important)
				*e = MISCMATHS::ndtri(0.5 * *c); // -eqZ
				*e = *e * *e; // eqX
        *g = -MISCMATHS::ndtri(*c);
				if (logP) { // -log10(p)
					*c = -log10(*c);
				}
				else {
					*c = 1 - *c; // 1-p
				}
			}
			else {
				if (*i > 0) {
					tmp = *d - dof1 + 1;
					*c = MISCMATHS::incbet(0.5 * tmp, 0.5 * dof1, tmp / (tmp + dof1 * *i)); // p(may not be precise for large p, but this is not important)
					*e = MISCMATHS::ndtri(0.5 * *c); // -eqZ
					*e = *e * *e; // eqX
					*g = -MISCMATHS::ndtri(*c);
					if (logP) { // -log10(p)
						*c = -log10(*c);
					}
					else {
						*c = 1 - *c; // 1-p
					}
				}
				else {
					*c = 0;
					*e = 0;
					*g = -datum::inf;
				}
			}
		}
	}

	////////////////////////////////////////
	// Methods for the VoxelwiseDesign Class (related to voxelwise designs and inspired from randomise)
	////////////////////////////////////////
	
	// adjust the design matrix for the specified voxel
	mat VoxelwiseDesign::adjustDesign(const mat& originalDesign, const int voxelNo)
	{
		mat newDesign(originalDesign);
		for (uword currentEV=0; currentEV < location.n_elem; currentEV++) {
			newDesign.col(location(currentEV)) = EV(currentEV).col(voxelNo);
		}
		return newDesign;
	}
	
	// setup a VoxelwiseDesign object
	void VoxelwiseDesign::setup(const vector<int>& EVnumbers, const vector<string>& EVfilenames, const volume<float>& mask, const int maximumLocation, const bool isVerbose)
	{
		isSet=false;
		if(EVnumbers.size() != EVfilenames.size())
			throw Exception("Number of input voxelwise_ev_filenames must match number of voxelwise_ev_numbers");
		location.set_size(EVnumbers.size());
		for(uword i = 0; i < EVnumbers.size(); i++) location(i) = EVnumbers.at(i) - 1; // -1 due to the fact that armadillo start counting at 0
		EV.set_size(EVfilenames.size());
		volume4D<float> input;
		Matrix tmp;
		if (isVerbose) cout << "\r";
		for(uword i = 0; i < EV.size(); i++) {
			if(EVnumbers.at(i) > maximumLocation)
				throw Exception("voxelwise_ev_numbers option specifies a number greater than number of design EVs)");
			if (isVerbose) cout << "Loading voxelwise ev " << EVfilenames.at(i) << " for EV " << EVnumbers.at(i) << ": " << flush;
			read_volume4D(input, EVfilenames.at(i));
			tmp = input.matrix(mask);
			EV(i) = matrix2Mat(tmp);
			if (isVerbose) cout << "done" << endl;
		}
		if (isVerbose) cout << "Setting up the model: " << flush;
		isSet=true;
	}

	////////////////////////////////////////
	// Methods for the SwEModel class
	////////////////////////////////////////
	
	// compute the subject effective number of degrees-of-fredom
	void SweModel::computeSubjectDof()
	{
		// first detect if the design matrix is separable
		iGrDof.set_size(designMatrix.n_rows); // indicate which separate design matrix each row (or observation) belongs to
		uvec iBetaDof = ones<uvec>(designMatrix.n_cols) * designMatrix.n_cols; // indicate which separate design matrix each column (or beta) belongs to (set at designMatrix.n_rows to indicate which one have not been assigned yet)
		umat tmp = eye<umat>(designMatrix.n_cols, designMatrix.n_cols); // indicates if the covariates share at least a non-zero value
		for (uword i = 0; i < designMatrix.n_cols; i++) {
			for (uword j = i + 1; j < designMatrix.n_cols; j++) {
				if (any(designMatrix.col(i) != 0 && designMatrix.col(j) != 0)) {
					tmp(i,j) = 1;
					tmp(j,i) = 1;
				}
			}
		}
		uword label = 0; // numeric label for the separate design matrix (uvec for compatibilty with uvec vectors)
		while (any(iBetaDof == designMatrix.n_cols)) { // if condition is false == every covariates has been labelled (a label can be max. designMatrix.n_rows - 1)
			uvec ind = tmp.col(as_scalar(find(iBetaDof == designMatrix.n_cols, 1))); // fetch the elements of tmp corresponding the first not-yet classified covariate
			while (1) {
				uvec ind2 = any(tmp.cols(find(ind == 1)), 1);
				if (all(ind == ind2)) {
					break; // no change == the current separate matrix has been completely found
				} else {
					ind = ind2; // changes == the current separate matrix may be larger (so continue)
				}
			}
			ind = find(ind == 1); // convert into indices
			iBetaDof(ind) = ones<uvec>(ind.n_elem) * label;
			ind = find(any(designMatrix.cols(ind) != 0, 1));
			iGrDof(ind) = ones<uvec>(ind.n_elem) * label;
			label++; // change the numeric labelling
		}
		
		uvec nSubject(label);
		vec pB = zeros<vec>(label);
		uvec tmp3(1); // used to convert a uword into a uvec
		vec tmp4; // used to save a temporary variable
		for (uword i = 0; i < label; i++) {
			uvec tmp = unique(subject(find(iGrDof == i)));
			nSubject(i) = tmp.n_elem;
			uvec tmp2 = find(iBetaDof == i);
			for (uword j =0; j < tmp2.n_elem; j++) {
				tmp3(0) = tmp2(j); // convert into a uvec object (if not submatrix views does not work)
				pB(i) += 1;
				for (uword k =0; k < tmp.n_elem; k++) {
					tmp4 = unique(designMatrix(find(subject==tmp(k) && iGrDof == i), tmp3));
					if(tmp4.n_elem > 1) {
						pB(i) -= 1;
						break;
					}
				}
			}
		}
		
		// compute subject effective dof
		pB = pB / nSubject; // convert pB into a correctif factor
		subjectDof.set_size(subjectIndex.n_elem);
		for (uword i = 0; i < subjectIndex.n_elem; i++) {
			subjectDof(i) = 1 - min(pB(iGrDof(subjectIndex(i))));
		}
	}
	
	// compute the weights matrix linking the contrasted SwEs to the subject covariance matrices (for the classic SwE only)
	void SweModel::computeWeightClassic()
	{
		mat tmp;
		uword it = 0;
		for (uword i = 0; i < n_i.n_elem; it += nCovVis_i(i), i++) {
			// for t-contrasts
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					tmp = tc.row(j) * pinvDesignMatrix.cols(subjectIndex(i));
					weightT(j)(span(it, it + nCovVis_i(i) - 1)) = kron(tmp, tmp) * duplicationMatrix(n_i(i));
				}
			}
			// for F-contrasts
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++) {
					tmp = fullFContrast(j) * pinvDesignMatrix.cols(subjectIndex(i));
					tmp = kron(tmp, tmp) * duplicationMatrix(n_i(i));
					weightF(j).cols(span(it, it + nCovVis_i(i) - 1)) = tmp.rows(eliminationIndices(sizeFcontrast(j)));
				}
			}
		}
	}

	// compute the weights matrix linking the contrasted SwEs to the group covariance matrices (for the modified SwE only)
	void SweModel::computeWeightModified()
	{
		mat tmp, tmp2;
		uword it = 0;
		uword it2 = 0;
		uword it3;
		for (uword g = 0; g < uGroup.n_elem; g++) {
			for (uword i = 0; i < n_g(g); i++, it++) {
				// for t-contrasts
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						weightT(j)(it + it2) = as_scalar(arma::sum(square(tc.row(j) * pinvDesignMatrix.cols(indDiagRes(it))), 1));
					}
				}
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++) {
						tmp = fullFContrast(j) * pinvDesignMatrix.cols(indDiagRes(it));
						it3 = 0;
						for (uword l = 0; l < tmp.n_rows; l++) {
							for (uword m = l; m < tmp.n_rows; m++, it3++) {
								weightF(j)(it3, it + it2) = as_scalar(accu(tmp.row(l) % tmp.row(m)));
							}
						}
					}
				}
				for (uword k = i + 1; k < n_g(g); k++, it2++) {
					// for t-contrasts
					if (!doFOnly) {
						for (uword j = 0; j < tc.n_rows; j++) {
							weightT(j)(it + it2 + 1) = 2 * as_scalar(arma::sum((tc.row(j) * pinvDesignMatrix.cols(indOffDiagRes(it2).col(0))) % (tc.row(j) * pinvDesignMatrix.cols(indOffDiagRes(it2).col(1))), 1));
						}
					}
					// for F-contrasts
					if (!doTOnly) {
						for (uword j = 0; j < fc.n_rows; j++) {
							tmp  = fullFContrast(j) * pinvDesignMatrix.cols(indOffDiagRes(it2).col(0));
							tmp2 = fullFContrast(j) * pinvDesignMatrix.cols(indOffDiagRes(it2).col(1));
							it3 = 0;
							for (uword l = 0; l < tmp.n_rows; l++) {
								for (uword m = l; m < tmp.n_rows; m++, it3++) {
									weightF(j)(it3, it + it2 + 1) = as_scalar(accu(tmp.row(l) % tmp2.row(m) + tmp.row(m) % tmp2.row(l)));
								}
							}
						}
					}
				}
			}
		}
	}
	
	// set up a SweModel object
	void SweModel::setup(const sweopts& opts)
	{
		// set variables indicating what to output
		voxelwiseOutput = opts.voxelwiseOutput.value();
		outputRaw = opts.outputRaw.value();
		outputEquivalent = opts.outputEquivalent.value();
		outputDof = opts.outputDof.value();
		outputUncorr = opts.outputUncorr.value();
		outputGlm = opts.outputGlm.value();
		outputTextNull = opts.outputTextNull.value();
		
		modified = opts.modifiedSwE.value();
		verbose = opts.verbose.value();
		wb = opts.wb.value();
		if (wb) {
			if (opts.n_boot.set()){
				nBootstrap = opts.n_boot.value();
			}
			else {
				nBootstrap = 999;
			}
			clusterThresholdT = opts.cluster_thresh.value();
			if (clusterThresholdT > 0 && clusterThresholdT < 1) clusterThresholdT = 1 - clusterThresholdT; // convert into 1-pValue threshold if needed
			clusterThresholdF = opts.f_thresh.value();
			if (clusterThresholdF > 0 && clusterThresholdF < 1) clusterThresholdF = 1 - clusterThresholdF; // convert into 1-pValue threshold if needed
			clusterMassThresholdT = opts.clustermass_thresh.value();
			if (clusterMassThresholdT > 0 && clusterMassThresholdT < 1) clusterMassThresholdT = 1 - clusterMassThresholdT; // convert into 1-pValue threshold if needed
			clusterMassThresholdF = opts.fmass_thresh.value();
			if (clusterMassThresholdF > 0 && clusterMassThresholdF < 1) clusterMassThresholdF = 1 - clusterMassThresholdF; // convert into 1-pValue threshold if needed
		}
		logP = opts.logP.value();
		out_fileroot = opts.out_fileroot.value();
		if (opts.voxelwise_ev_numbers.set() && opts.voxelwise_ev_filenames.set()) {
			// voxelWiseDesign = true;
			voxelWiseDesign = false;
			throw Exception("The option of using a voxelwise design using --vxl and --vxf is currently disabled as it has not been fully tested yet. Please run again without these options to avoid this error. Voxelwise design will be avaible soon after a thorough testing!");
		}
		else {
			if (opts.voxelwise_ev_numbers.set() || opts.voxelwise_ev_filenames.set())
				throw Exception("either the voxelwise covariate images or their locations is not specified");
			voxelWiseDesign = false;
		}
		doFOnly = opts.doFOnly.value();
		
		if (opts.dm_file.value()=="" || opts.tc_file.value()=="" || opts.subj_file.value()=="" ) throw Exception("swe requires a design matrix,  contrast(s) and subject specification as input");
		if (opts.dm_file.value()!="") designMatrix = read_vest_swe(opts.dm_file.value());
		if (opts.tc_file.value()!="") tc = read_vest_swe(opts.tc_file.value());
		if (opts.fc_file.value()!="") {
			fc = read_vest_swe(opts.fc_file.value());
			doTOnly = false;
		}
		else {
			doTOnly = true;
		}
		if (opts.subj_file.value()!=""){
			umat tmp = conv_to<umat>::from(read_vest_swe(opts.subj_file.value()));
			subject = tmp.col(0);
			if (modified){
				visit = tmp.col(1);
				group = tmp.col(2);
			}
		}
		
		// load the mask if one is specified by the user
		if (opts.maskname.value()!="") {
			read_volume(mask, opts.maskname.value());
			if (mask.sum() < 1) throw Exception("Data mask is blank.");
		}
		
		// setup the voxelWiseDesign member if needed
		if (voxelWiseDesign) {
			voxelWiseEV.setup(opts.voxelwise_ev_numbers.value(), opts.voxelwise_ev_filenames.value(), mask, designMatrix.n_rows, verbose);
		}
		
		// pseudoinverse of the design matrix
		pinvDesignMatrix = pinv(designMatrix);
		
		// set up variables related to the subjects
		uSubject = unique(subject);
		subjectIndex.set_size(uSubject.n_elem);
		subjectCovIndex.set_size(uSubject.n_elem);
		nCovVis_i.set_size(uSubject.n_elem);
		n_i.set_size(uSubject.n_elem);
		
		for (uword i = 0; i < uSubject.n_elem; i++) {
			subjectIndex(i) = find(subject == uSubject(i));
			n_i(i) = subjectIndex(i).n_elem;
			nCovVis_i(i) = n_i(i) * (n_i(i) + 1) / 2;
			subjectCovIndex(i) = zeros<umat>(nCovVis_i(i), 2);
			uword it = 0;
			for (uword k = 0; k < n_i(i); k++){
				for (uword kk = k; kk < n_i(i); kk++) {
					subjectCovIndex(i)(it, 0) = subjectIndex(i)(k);
					subjectCovIndex(i)(it, 1) = subjectIndex(i)(kk);
					it++;
				}
			}
		}

		// full F-contrasts
		if (!doTOnly){
			fullFContrast.set_size(fc.n_rows);
			sizeFcontrast.set_size(fc.n_rows);
			for (uword j = 0; j < fc.n_rows; j++){
				fullFContrast(j) = tc.rows(find(fc.row(j) == 1));
				sizeFcontrast(j) = fullFContrast(j).n_rows;
			}
			nCovCFSweCF = sizeFcontrast % (sizeFcontrast + 1) / 2;
		}
		// compute the subject-specific effective dof
		computeSubjectDof();

		// compute "modified"-specific variables
		if (modified){

			uGroup = unique(group);
			uVisit.set_size(uGroup.n_elem);
			n_g.set_size(uGroup.n_elem);
			for (uword g = 0; g < uGroup.n_elem; g++) {
				uVisit(g) = unique(visit(find(group == uGroup(g))));
				n_g(g) = uVisit(g).n_elem;

			}
			nCovVis_g = n_g % (n_g + 1) / 2;
	
			indDiag.set_size(arma::sum(n_g));
			indOffDiag.set_size(arma::sum(n_g % (n_g - 1)) / 2);
			indCorrDiag.set_size(indOffDiag.n_elem, 2);
			indDiagRes.set_size(arma::sum(n_g));
			indOffDiagRes.set_size(indOffDiag.n_elem);
		
			uword it = 0;
			uword it2 = 0;
			uword it3, it4;
			uvec subjI, subjJ, subjIntersection, uSubjGroup;
			uvec flag(subject.n_elem);
			mat tmp, tmp2;
			dofMat.set_size(n_g.n_elem); // field containing information about the dof of each group (note that it accounts for the missing data)
			uword mij, mab, tmpMij, tmpMab;
			umat ind;
			double tmp3;

			for (uword g = 0; g < uGroup.n_elem; g++) {
				it3 = 0;
				for (uword i = 0; i < n_g(g); i++) {
					indDiag(it2) = it3;
					indDiagRes(it2) = find(visit == uVisit(g)(i) && group == uGroup(g));
					it3++;
					for (uword j = i + 1; j < n_g(g); j++) {
						indOffDiag(it) = it3;
						indCorrDiag(it, 0) = it2;
						indCorrDiag(it, 1) = it2 + j - i;
						// need to find the subjects in group g with data at both visits i and j
						subjI = sort(subject(find(visit == uVisit(g)(i) && group == uGroup(g)))); // subjects in group g with data at visit i
						subjJ = sort(subject(find(visit == uVisit(g)(j) && group == uGroup(g)))); // subjects in group g with data at visit j
						subjIntersection = vintersection(subjI, subjJ);
						flag.zeros(); // reset the flag to zero
						// build up the flag selecting the indices of the subjects in group g with data at both visits i and j
						for (uword k = 0; k < subjIntersection.n_elem; k++) {
							flag = flag || (subject == subjIntersection(k));
						}
						indOffDiagRes(it) = umat(subjIntersection.n_elem, 2);
						indOffDiagRes(it).col(0) = find(visit == uVisit(g)(i) && flag);
						indOffDiagRes(it).col(1) = find(visit == uVisit(g)(j) && flag);

						it3++;
						it++;
					}
					it2++;
				}

				// dof information into dofMat
				dofMat(g).zeros(nCovVis_g(g), nCovVis_g(g));
				uSubjGroup = unique(subject(find(group == uGroup(g))));
				it3 = 0;
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i; j < n_g(g); j++) {
						it4 = 0;
						for (uword a = 0; a < n_g(g); a++) {
							for (uword b = a; b < n_g(g); b++) {
								mab  = 0;
								mij = 0;
								tmp3 = 0;
								for (uword iSubj = 0; iSubj < uSubjGroup.n_elem; iSubj++) {
									tmpMij = any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(i)) && any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(j));
									mij += tmpMij;
									tmpMab = 1 * any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(a)) && any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(b));
									mab += tmpMab;
									if (tmpMij && tmpMab) tmp3 += 1 / as_scalar(subjectDof(find(uSubject == uSubjGroup(iSubj))));
								}
								if (mij * mab != 0) dofMat(g)(it3,it4) = tmp3 / (mij * mab);
								it4++;
							}
						}
						it3++;
					}
				}
			}
		}

		// set the sizes of the weights and compute them
		nContrasts = 0;
		if (!doFOnly) {
			weightT.set_size(tc.n_rows);
			for (uword j = 0; j < tc.n_rows; j++) {
				weightT(j).zeros(arma::sum(nCovVis_i));
			}
			nContrasts += tc.n_rows;
		}

		if (!doTOnly){
			weightF.set_size(fc.n_rows);
			for (uword j = 0; j < fc.n_rows; j++) {
				weightF(j).zeros(nCovCFSweCF(j), arma::sum(nCovVis_i));
			}
			nContrasts += fc.n_rows;
		}

		if (modified) {
			computeWeightModified();
		}
		else {
			computeWeightClassic();
		}

		// if WB, check if a resampling matrix was supplied or generate it now
		// assign the number of scans to nData now as generateResamplingMatrix needs its value (will be overwritten latter when loading the data)
		nData = designMatrix.n_rows;
		if (wb) {
			if (opts.resamplingMatrix_file.set()) {
				resamplingMult = read_vest_swe(opts.resamplingMatrix_file.value());
			}
			else {
				generateResamplingMatrix();
			}
		}

		// TFCE variables
		if (wb) {
			doTfce = false;
			if (opts.tfce.value()) {
				doTfce = true;
				tfceHeight = opts.tfce_height.value();
				tfceExtent = opts.tfce_size.value();
				tfceDeltaOverride = opts.tfce_delta.set();
				tfceDelta.set_size(nContrasts);
				tfceDelta.fill(opts.tfce_delta.value());
				tfceConnectivity = opts.tfce_connectivity.value();       
			}
			// if tfce2D, override some values like in randomise 
			if (opts.tfce2D.value()) {
				doTfce = true;
				tfceHeight = 2;
				tfceExtent = 1;
				tfceDeltaOverride = opts.tfce_delta.set();
				tfceDelta.set_size(nContrasts);
				tfceDelta.fill(opts.tfce_delta.value());
				tfceConnectivity = 26;
			}
		}
	}
	
	// check if the model looks alright (additional checking steps could be added)
	void SweModel::checkModel(const uword numberOfDataPoints)
	{
		if (designMatrix.n_rows != numberOfDataPoints) throw Exception("number of rows in design matrix doesn't match number of \"data points\" in input data!");
		if (tc.n_cols != designMatrix.n_cols) throw Exception("number of columns in t-contrast matrix doesn't match number of columns in design matrix!");
		if (fc.n_cols !=0 && fc.n_cols!=tc.n_rows) throw Exception("number of columns in f-contrast matrix doesn't match number of rows in t-contrast matrix!");
		if (wb) {
			if (resamplingMult.n_rows != numberOfDataPoints || resamplingMult.n_cols != nBootstrap)
			{
				std::stringstream stringError;
				stringError << " the size of the resampling matrix (" << resamplingMult.n_rows << " x " << resamplingMult.n_cols << ") is not the number of data points (" << numberOfDataPoints << ") x the number of bootstraps (" << nBootstrap << ")!";
				throw Exception(stringError.str());
			} 
		}
	}
	
	// load in-mask data and produce a mask if needed
	void SweModel::loadData(mat& data, const sweopts& opts)
	{
		volume4D<float> tmpData;
		read_volume4D(tmpData, opts.in_fileroot.value());
		// load explicit mask if specified
		if (!(opts.maskname.value()!="")) {
			mask = nonConstantMask(tmpData, opts.disableNonConstantMask.value());
			if (mask.sum() < 1) throw Exception("Data mask is blank.");
		}
		Matrix tmp = tmpData.matrix(mask); // data.matrix() returns a ReturnMatrix format
		
		data = matrix2Mat(tmp); // convert into mat format
		nVox = data.n_cols; // assign the total number of in-mask voxels
		nData = data.n_rows; // assign the total number of input images
	}
	
	// adjust the member dofMat (matrix containing information about the dof for the modified SwE only) for a specific voxel
	void SweModel::adjustDofMat()
	{
		uvec uSubjGroup;
		uword mij, mab, tmpMij, tmpMab, it3, it4;
		double tmp3;
		for (uword g = 0; g < uGroup.n_elem; g++) {
			it3 = 0;
			dofMat(g).zeros(nCovVis_g(g), nCovVis_g(g));
			uSubjGroup = unique(subject(find(group == uGroup(g))));
			for (uword i = 0; i < n_g(g); i++) {
				for (uword j = i; j < n_g(g); j++) {
					it4 = 0;
					for (uword a = 0; a < n_g(g); a++) {
						for (uword b = a; b < n_g(g); b++) {
							mab  = 0;
							mij = 0;
							tmp3 = 0;
							for (uword iSubj = 0; iSubj < uSubjGroup.n_elem; iSubj++) {
								tmpMij = any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(i)) && any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(j));
								mij += tmpMij;
								tmpMab = 1 * any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(a)) && any(subject == uSubjGroup(iSubj) && visit == uVisit(g)(b));
								mab += tmpMab;
								if (tmpMij && tmpMab) tmp3 += 1 / as_scalar(subjectDof(find(uSubject == uSubjGroup(iSubj))));
							}
							if (mij * mab != 0) dofMat(g)(it3,it4) = tmp3 / (mij * mab);
							it4++;
						}
					}
					it3++;
				}
			}
		}
	}
	
	// adjust the design for a specific voxel
	void SweModel::adjustDesign(uword iVox)
	{
		for (uword currentEV=0; currentEV < voxelWiseEV.location.n_elem; currentEV++) {
			designMatrix.col(voxelWiseEV.location(currentEV)) = voxelWiseEV.EV(currentEV).col(iVox);
		}
		
		pinvDesignMatrix = pinv(designMatrix);
		
		if (modified){
			computeWeightModified();
			vec prevSubjectDof = subjectDof;
			computeSubjectDof();
			if (!all(prevSubjectDof == subjectDof)) adjustDofMat();
		}
		else {
			computeWeightClassic();
			computeSubjectDof();
		}
	}
	
	// compute the parameter estimates
	mat SweModel::computePe(const mat& data)
	{
		return pinvDesignMatrix * data;
	}
	
	// compute the adjusterd residuals (adjustment S_{C2})
	mat SweModel::computeAdjustedResiduals(const mat& data, const mat& pe)
	{
		mat residuals = data - designMatrix * pe;
		
		// compute Identity - Hat matrix
		mat I_H = eye<mat>(designMatrix.n_rows, designMatrix.n_rows) - designMatrix * pinvDesignMatrix;
		
		// compute the adjusted residuals subject per subject
		mat tmp;
		vec eigval;
		mat eigvec;
		for (uword i = 0; i < subjectIndex.n_elem; i++){
			tmp = I_H(subjectIndex(i), subjectIndex(i));
			eig_sym(eigval, eigvec, tmp);
			tmp = eigvec * diagmat(1 / sqrt(eigval)) * eigvec.t();
			residuals.rows(subjectIndex(i)) = tmp * residuals.rows(subjectIndex(i));
		}
		return residuals;
	}

	// compute the adjusterd restricted residuals (adjustment S_{RC2})
	mat SweModel::computeAdjustedRestrictedResiduals(const mat& data, const mat& restrictedFittedData, const mat& restrictedMatrix)
	{
		mat restrictedResiduals = data - restrictedFittedData;
		
		// compute Identity - restrited Hat matrix
		mat I_H = eye<mat>(designMatrix.n_rows, designMatrix.n_rows) - restrictedMatrix * pinvDesignMatrix;
		
		// compute the adjusted residuals subject per subject
		mat tmp;
		vec eigval;
		mat eigvec;
		for (uword i = 0; i < subjectIndex.n_elem; i++){
			tmp = I_H(subjectIndex(i), subjectIndex(i));
			eig_sym(eigval, eigvec, tmp);
			tmp = eigvec * diagmat(1 / sqrt(eigval)) * eigvec.t();
			restrictedResiduals.rows(subjectIndex(i)) = tmp * restrictedResiduals.rows(subjectIndex(i));
		}
		return restrictedResiduals;
	}
	
	// save image
	void SweModel::saveImage(mat& dataToSave, string filename) // cannot compile with const argument...
	{
		volume4D<float> tmpVol(mask.xsize(), mask.ysize(), mask.zsize(), dataToSave.n_rows);
		Matrix tmp = mat2Matrix(dataToSave);
		tmpVol.setmatrix(tmp, mask);
		tmpVol.copyproperties(mask);
		save_volume4D(tmpVol, filename);
	}

	// compute vechCovVechV for the "Homogeneous" group g (could be optimised for voxelwise design)
	void SweModel::computeVechCovVechV(mat& vechCovVechV, const mat& covVis_g, const uword g)
 {
		uword it = 0;
		umat ind(nCovVis_g(g),2);
		uword indkl, indkkll, indkk, indkkkk, indkkl, indkll, indll, indllll;
		for (uword i = 0; i < n_g(g); i++) {
			for (uword j = i; j < n_g(g); j++, it++) {
				ind(it, 0) = i;
				ind(it, 1) = j;
			}
		}
		it = 0;
		for (uword i = 0; i < nCovVis_g(g); i++) {
			for (uword j = i; j < nCovVis_g(g); j++, it++) {
				indkl   = as_scalar(find((ind.col(0) == ind(i, 0) && ind.col(1) == ind(j, 0)) || (ind.col(1) == ind(i, 0) && ind.col(0) == ind(j, 0))));
				indkkll = as_scalar(find((ind.col(0) == ind(i, 1) && ind.col(1) == ind(j, 1)) || (ind.col(1) == ind(i, 1) && ind.col(0) == ind(j, 1))));
				indkll  = as_scalar(find((ind.col(0) == ind(i, 0) && ind.col(1) == ind(j, 1)) || (ind.col(1) == ind(i, 0) && ind.col(0) == ind(j, 1))));
				indkkl  = as_scalar(find((ind.col(0) == ind(i, 1) && ind.col(1) == ind(j, 0)) || (ind.col(1) == ind(i, 1) && ind.col(0) == ind(j, 0))));
				indkk   = as_scalar(find((ind.col(0) == ind(i, 0) && ind.col(1) == ind(i, 0))));
				indkkkk = as_scalar(find((ind.col(0) == ind(i, 1) && ind.col(1) == ind(i, 1))));
				indll   = as_scalar(find((ind.col(0) == ind(j, 0) && ind.col(1) == ind(j, 0))));
				indllll = as_scalar(find((ind.col(0) == ind(j, 1) && ind.col(1) == ind(j, 1))));
				
				vechCovVechV.row(it) = dofMat(g)(i, j) * (covVis_g.row(indkl) % covVis_g.row(indkkll) + covVis_g.row(indkll) % covVis_g.row(indkkl))
				+ covVis_g.row(i) % ((covVis_g.row(indkl) % covVis_g.row(indkll) / covVis_g.row(indkk)) * (dofMat(g)(indkk, j) - dofMat(g)(i, j)) + (covVis_g.row(indkkl) % covVis_g.row(indkkll) / covVis_g.row(indkkkk)) * (dofMat(g)(indkkkk, j) - dofMat(g)(i, j)))
				+ covVis_g.row(j) % ((covVis_g.row(indkl) % covVis_g.row(indkkl) / covVis_g.row(indll)) * (dofMat(g)(i, indll) - dofMat(g)(i, j)) + (covVis_g.row(indkll) % covVis_g.row(indkkll) / covVis_g.row(indllll)) * (dofMat(g)(i, indllll) - dofMat(g)(i, j)))
				+ covVis_g.row(i) % covVis_g.row(j) % (
																							 square(covVis_g.row(indkl)) / (covVis_g.row(indkk) % covVis_g.row(indll))
																							 * (dofMat(g)(indkk, indll) + dofMat(g)(i, j) - dofMat(g)(indkk, j) - dofMat(g)(i, indll))
																							 + square(covVis_g.row(indkll)) / (covVis_g.row(indkk) % covVis_g.row(indllll))
																							 * (dofMat(g)(indkk, indllll) + dofMat(g)(i, j) - dofMat(g)(indkk, j) - dofMat(g)(i, indllll))
																							 + square(covVis_g.row(indkkl)) / (covVis_g.row(indkkkk) % covVis_g.row(indll))
																							 * (dofMat(g)(indkkkk, indll) + dofMat(g)(i, j) - dofMat(g)(indkkkk, j) - dofMat(g)(i, indll))
																							 + square(covVis_g.row(indkkll)) / (covVis_g.row(indkkkk) % covVis_g.row(indllll))
																							 * (dofMat(g)(indkkkk, indllll) + dofMat(g)(i, j) - dofMat(g)(indkkkk, j) - dofMat(g)(i, indllll))
																							 ) / 2;
			}
		}
	}
	
	// compute cluster extent stats for t-contrasts
	void SweModel::computeClusterStatsT(mat& score, uvec& clusterLabels, uvec& clusterExtentSizes)
	{
		ColumnVector tmpClusterExtentSizes;
		volume4D<float> spatialStatistic;
		Matrix tmp = mat2Matrix(score);
		spatialStatistic.setmatrix(tmp, mask);
		spatialStatistic.binarise(clusterThresholdT);
		volume<int> tmpClusterLabels = connected_components(spatialStatistic[0], tmpClusterExtentSizes, CLUST_CON);
		// convert into arma format
		// convert labels into arma format
		uword it = 0;
		clusterLabels.set_size(score.n_cols);
		for(int z=0; z<mask.zsize(); z++) {
			for(int y=0; y<mask.ysize(); y++) {
				for(int x=0; x<mask.xsize(); x++) {
					if(mask(x,y,z)>0) {
						clusterLabels(it) = tmpClusterLabels(x, y, z);
						it++;
					}
				}
			}
		}
		clusterExtentSizes = columnVector2uvec(tmpClusterExtentSizes);
	}

	// compute Cluster mass stats for t-contrasts
	void SweModel::computeClusterStatsT(mat& score, uvec& clusterLabels, vec& clusterMassSizes)
	{
		ColumnVector tmpClusterMassSizes;
		volume4D<float> spatialStatistic;
		Matrix tmp = mat2Matrix(score);
		spatialStatistic.setmatrix(tmp, mask);
		spatialStatistic.binarise(clusterMassThresholdT);
		volume<int> tmpClusterLabels = connected_components(spatialStatistic[0], tmpClusterMassSizes, CLUST_CON);
		// convert labels into arma format
		uword it = 0;
		clusterLabels.set_size(score.n_cols);
		for(int z=0; z<mask.zsize(); z++) {
			for(int y=0; y<mask.ysize(); y++) {
				for(int x=0; x<mask.xsize(); x++) {
					if(mask(x,y,z)>0) {
						clusterLabels(it) = tmpClusterLabels(x, y, z);
						it++;
					}
				}
			}
		}
		// compute mass statistics
		clusterMassSizes.set_size(tmpClusterMassSizes.Nrows());
		for (uword i = 0; i < tmpClusterMassSizes.Nrows(); i++)
			clusterMassSizes(i) = arma::sum(score(find(clusterLabels == (i + 1))));
	}
	
	void SweModel::computeClusterStatsT(mat& pValue, mat& score, uvec& clusterLabels, vec& clusterMassSizes)
	{
		ColumnVector tmpClusterMassSizes;
		volume4D<float> spatialStatistic;
		Matrix tmp = mat2Matrix(pValue);
		spatialStatistic.setmatrix(tmp, mask);
		spatialStatistic.binarise(clusterMassThresholdT);
		volume<int> tmpClusterLabels = connected_components(spatialStatistic[0], tmpClusterMassSizes, CLUST_CON);
		// convert labels into arma format
		uword it = 0;
		clusterLabels.set_size(score.n_cols);
		for(int z=0; z<mask.zsize(); z++) {
			for(int y=0; y<mask.ysize(); y++) {
				for(int x=0; x<mask.xsize(); x++) {
					if(mask(x,y,z)>0) {
						clusterLabels(it) = tmpClusterLabels(x, y, z);
						it++;
					}
				}
			}
		}
		// compute mass statistics
		clusterMassSizes.set_size(tmpClusterMassSizes.Nrows());
		for (uword i = 0; i < tmpClusterMassSizes.Nrows(); i++)
			clusterMassSizes(i) = arma::sum(score(find(clusterLabels == (i + 1))));
	}

	// compute Cluster extent stats for f-contrasts
	void SweModel::computeClusterStatsF(mat& score, uvec& clusterLabels, uvec& clusterExtentSizes)
	{
		ColumnVector tmpClusterExtentSizes;
		volume4D<float> spatialStatistic;
		Matrix tmp = mat2Matrix(score);
		spatialStatistic.setmatrix(tmp, mask);
		spatialStatistic.binarise(clusterThresholdF);
		volume<int> tmpClusterLabels = connected_components(spatialStatistic[0], tmpClusterExtentSizes, CLUST_CON);
		// convert labels into arma format
		uword it = 0;
		clusterLabels.set_size(score.n_cols);
		for(int z=0; z<mask.zsize(); z++) {
			for(int y=0; y<mask.ysize(); y++) {
				for(int x=0; x<mask.xsize(); x++) {
					if(mask(x,y,z)>0) {
						clusterLabels(it) = tmpClusterLabels(x, y, z);
						it++;
					}
				}
			}
		}
		clusterExtentSizes = columnVector2uvec(tmpClusterExtentSizes);
	}
	
	// compute Cluster mass stats for f-contrasts
	void SweModel::computeClusterStatsF(mat& score, uvec& clusterLabels, vec& clusterMassSizes)
	{
		ColumnVector tmpClusterMassSizes;
		volume4D<float> spatialStatistic;
		Matrix tmp = mat2Matrix(score);
		spatialStatistic.setmatrix(tmp, mask);
		spatialStatistic.binarise(clusterMassThresholdF);
		volume<int> tmpClusterLabels = connected_components(spatialStatistic[0], tmpClusterMassSizes, CLUST_CON);
		// convert labels into arma format
		uword it = 0;
		clusterLabels.set_size(score.n_cols);
		for(int z=0; z<mask.zsize(); z++) {
			for(int y=0; y<mask.ysize(); y++) {
				for(int x=0; x<mask.xsize(); x++) {
					if(mask(x,y,z)>0) {
						clusterLabels(it) = tmpClusterLabels(x, y, z);
						it++;
					}
				}
			}
		}
		// compute mass statistics
		clusterMassSizes.set_size(tmpClusterMassSizes.Nrows());
		for (uword i = 0; i < tmpClusterMassSizes.Nrows(); i++)
			clusterMassSizes(i) = arma::sum(score(find(clusterLabels == (i + 1))));
	}

	void SweModel::computeClusterStatsF(mat& pValue, mat& score, uvec& clusterLabels, vec& clusterMassSizes)
	{
		ColumnVector tmpClusterMassSizes;
		volume4D<float> spatialStatistic;
		Matrix tmp = mat2Matrix(pValue);
		spatialStatistic.setmatrix(tmp, mask);
		spatialStatistic.binarise(clusterMassThresholdF);
		volume<int> tmpClusterLabels = connected_components(spatialStatistic[0], tmpClusterMassSizes, CLUST_CON);
		// convert labels into arma format
		uword it = 0;
		clusterLabels.set_size(score.n_cols);
		for(int z=0; z<mask.zsize(); z++) {
			for(int y=0; y<mask.ysize(); y++) {
				for(int x=0; x<mask.xsize(); x++) {
					if(mask(x,y,z)>0) {
						clusterLabels(it) = tmpClusterLabels(x, y, z);
						it++;
					}
				}
			}
		}
		// compute mass statistics
		clusterMassSizes.set_size(tmpClusterMassSizes.Nrows());
		for (uword i = 0; i < tmpClusterMassSizes.Nrows(); i++)
			clusterMassSizes(i) = arma::sum(score(find(clusterLabels == (i + 1))));
	}
	
	// compute TFCE scores from z-scores
	void SweModel::tfce(mat& zScore, mat& tfceScore, const uword indexContrast){
		volume<float> spatialStatistic;
		spatialStatistic.setmatrix(zScore, mask);
		NEWIMAGE::tfce(spatialStatistic, tfceHeight, tfceExtent, tfceConnectivity, 0, tfceDelta(indexContrast));
		tfceScore = spatialStatistic.matrix(mask);
	}

	// generate resampling matrix for the WB
	void SweModel::generateResamplingMatrix()
	{
		resamplingMult.ones(nData, nBootstrap);
		uvec ind;
		for (uword iB = 0; iB < nBootstrap; iB++) {
			for (uword i = 0; i < uSubject.n_elem; i++) {
				if ((float)rand()/RAND_MAX > 0.5) {
					ind = find(subject == uSubject(i));
					for (uword j = 0; j < ind.n_elem; j++) resamplingMult(ind(j),iB) = -1;
				}
			}
		}
	}
	
	// print maximum statistics
	void SweModel::printMaxStats(const vec& maxStat, const string typeLabel, const string statLabel)
	{
		ofstream output_file((out_fileroot + typeLabel + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel + ".txt").c_str());
		output_file << maxStat;
		output_file.close();
	}

	// save cluster-wise p-values images
	void SweModel::saveClusterWisePValues(const mat& oneMinusFwePCluster, const uvec& clusterLabels, const string typeLabel, const string statLabel)
	{
		mat tmp(1, nVox, fill::zeros);
		if (oneMinusFwePCluster.n_elem > 0) {
			if(logP){
				for (uword i = 0; i < nVox; i++) {
					if (clusterLabels(i) > 0) tmp(i) = -log10(1-oneMinusFwePCluster(clusterLabels(i) - 1));
				}
				saveImage(tmp, out_fileroot + typeLabel + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
			}
			else{
				for (uword i = 0; i < nVox; i++) {
					if (clusterLabels(i) > 0) tmp(i) = oneMinusFwePCluster(clusterLabels(i) - 1);
				}
				saveImage(tmp, out_fileroot + typeLabel + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
			}
		}
		else {
			if(logP){
				saveImage(tmp, out_fileroot + typeLabel + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);       
			}
			else {
				saveImage(tmp, out_fileroot + typeLabel + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);       
			}
		}
	}
	
	// compute the SwE for t-contrasts
	void SweModel::computeSweT(mat& swe, const mat& residuals, const uword iContrast)
	{
		uword it = 0;
		swe.zeros(1, residuals.n_cols);
		if (modified){
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g;
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				swe += weightT(iContrast).cols(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
			}
		}
		else { // classic SwE
			mat covVis_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				swe += weightT(iContrast)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
			}
		}
	}
	// compute the SwE for t-contrasts + dof
	void SweModel::computeSweT(mat& swe, mat& dof, const mat& residuals, const uword iContrast)
	{
		uword it = 0;
		swe.zeros(1, residuals.n_cols);
		dof.zeros(1, residuals.n_cols);
		if (modified){
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g, vechCovVechV;
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				swe += weightT(iContrast).cols(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
				
				// compute vechCovVechV for group g and update dof
				vechCovVechV.zeros(nCovVis_g(g) * (nCovVis_g(g) + 1) / 2, covVis_g.n_cols);
				computeVechCovVechV(vechCovVechV, covVis_g, g);
				dof += (kron(weightT(iContrast)(span(it3, it3 + nCovVis_g(g) - 1)), weightT(iContrast)(span(it3, it3 + nCovVis_g(g) - 1))) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV;
			} // end loop g
			dof = 2 * square(swe) / dof;
		}
		else { // classic SwE
			mat covVis_i, swe_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				swe_i = weightT(iContrast)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
				swe += swe_i;
				dof += square(swe_i) / subjectDof(i);
			}
			dof = square(swe) / dof;
		}
	}
	
	// compute the SwE for f-contrasts
	void SweModel::computeSweF(mat& swe, const mat& residuals, const uword iContrast)
	{
		uword it = 0;
		swe.zeros(nCovCFSweCF(iContrast), residuals.n_cols);
		if (modified){
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g;
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				swe += weightF(iContrast).cols(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
			}
		}
		else { // classic SwE
			mat covVis_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				swe += weightF(iContrast).cols(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
			}
		}
	}

	// compute the SwE for f-contrasts + dof
	void SweModel::computeSweF(mat& swe, mat& dof, const mat& residuals, const uword iContrast)
	{
		uword it = 0;
		swe.zeros(nCovCFSweCF(iContrast), residuals.n_cols);
		dof.zeros(1, residuals.n_cols);
		if (modified){
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g, vechCovVechV, tmp;
			if (sizeFcontrast(iContrast) > 1) {
				tmp.eye(sizeFcontrast(iContrast) * sizeFcontrast(iContrast), sizeFcontrast(iContrast) * sizeFcontrast(iContrast)); // note that tr(A) = vec(I).t() * vec(A)
				tmp = vectorise(tmp).t() * kron(duplicationMatrix(sizeFcontrast(iContrast)), duplicationMatrix(sizeFcontrast(iContrast)));
			}
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				swe += weightF(iContrast).cols(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
				
				// compute vechCovVechV for group g and update dof
				vechCovVechV.zeros(nCovVis_g(g) * (nCovVis_g(g) + 1) / 2, covVis_g.n_cols);
				computeVechCovVechV(vechCovVechV, covVis_g, g);
				if (sizeFcontrast(iContrast) == 1) {
					dof += (kron(weightF(iContrast).cols(it3, it3 + nCovVis_g(g) - 1), weightF(iContrast).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV;
				}
				else {
					dof += (tmp * (kron(weightF(iContrast).cols(it3, it3 + nCovVis_g(g) - 1), weightF(iContrast).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))))	* vechCovVechV;;
				}
			} // end loop g
			if (sizeFcontrast(iContrast) == 1) {
				dof = 2 * square(swe) / dof;
			}
			else {
				uvec indDiagF(sizeFcontrast(iContrast));
				uvec indOffDiagF(nCovCFSweCF(iContrast) - sizeFcontrast(iContrast));
				uword it3 = 0;
				for (uword k = 0; k < sizeFcontrast(iContrast); k++){
					indDiagF(k) = it3;
					it3++;
					for (uword kk = k + 1; kk < sizeFcontrast(iContrast); kk++, it3++){
						indOffDiagF(it3 - k - 1) = it3;
					}
				}
				dof = (square(arma::sum(swe.rows(indDiagF))) + arma::sum(square(swe.rows(indDiagF))) + 2 * arma::sum(square(swe.rows(indOffDiagF)))) / dof;
			}
		}
		else { // classic SwE
			mat covVis_i, swe_i;
			uvec indDiagF;
			uvec indOffDiagF;
			if (sizeFcontrast(iContrast) > 1) {
				indDiagF.set_size(sizeFcontrast(iContrast));
				indOffDiagF.set_size(nCovCFSweCF(iContrast) - sizeFcontrast(iContrast));
				uword it3 = 0;
				for (uword k = 0; k < sizeFcontrast(iContrast); k++){
					indDiagF(k) = it3;
					it3++;
					for (uword kk = k + 1; kk < sizeFcontrast(iContrast); kk++, it3++){
						indOffDiagF(it3 - k - 1) = it3;
					}
				}
			}
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				swe_i = weightF(iContrast).cols(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
				swe += swe_i;
				if (sizeFcontrast(iContrast) == 1) {
					dof += square(swe_i) / subjectDof(i);
				}
				else {
					dof += (square(arma::sum(swe_i.rows(indDiagF))) + arma::sum(square(swe_i.rows(indDiagF))) + 2 * arma::sum(square(swe_i.rows(indOffDiagF)))) / subjectDof(i);
				}
			}
			if (sizeFcontrast(iContrast) == 1) {
				dof = square(swe) / dof;
			}
			else {
				dof = (square(arma::sum(swe.rows(indDiagF))) + arma::sum(square(swe.rows(indDiagF))) + 2 * arma::sum(square(swe.rows(indOffDiagF)))) / dof;
			}
		}
	}
	
	// compute t-contrasts (no effective number of degrees of freedom)
	void SweModel::computeT(const mat& data, mat& tstat, mat& cope, mat& varcope, const uword iContrast)
	{
		cope = computePe(data); // cope is pe
		computeSweT(varcope, computeAdjustedResiduals(data, cope), iContrast);
		cope = tc.row(iContrast) * cope; // cope is cope now
		tstat = cope / sqrt(varcope);
	}
	
	// compute t-contrasts (with effective number of degrees of freedom)
	void SweModel::computeT(const mat& data, mat& tstat, mat& cope, mat& varcope, mat& dof, const uword iContrast)
	{
		cope = computePe(data); // cope is pe
		computeSweT(varcope, dof, computeAdjustedResiduals(data, cope), iContrast);
		cope = tc.row(iContrast) * cope; // cope is cope now
		tstat = cope / sqrt(varcope);
	}
	
	// compute f-contrasts (no effective number of degrees of freedom)
	void SweModel::computeF(const mat& data, mat& fstat, mat& cope, mat& varcope, const uword iContrast)
	{
		cope = computePe(data); // cope is pe
		computeSweF(varcope, computeAdjustedResiduals(data, cope), iContrast);
		cope = fullFContrast(iContrast) * cope; // cope is cope now
		if (sizeFcontrast(iContrast) == 1) {
			fstat = square(cope) / varcope; // cope is replaced by the F-scores
		}
		else { // need to do it voxel by voxel
			mat tmp(sizeFcontrast(iContrast), sizeFcontrast(iContrast));
			uvec ind = find(trimatl(ones<umat>(sizeFcontrast(iContrast), sizeFcontrast(iContrast)))== 1); // indices for the lower triangular of tmp
			for (uword iVox = 0; iVox < data.n_cols; iVox++) {
				tmp(ind) = varcope.col(iVox);
				fstat(iVox) = as_scalar(cope.col(iVox).t() * solve(symmatl(tmp), cope.col(iVox))) / sizeFcontrast(iContrast);
			}
		}
	}
	
	// compute f-contrasts (with effective number of degrees of freedom into account)
	void SweModel::computeF(const mat& data, mat& fstat, mat& cope, mat& varcope, mat& dof, const uword iContrast)
	{
		cope = computePe(data); // cope is pe
		computeSweF(varcope, dof, computeAdjustedResiduals(data, cope), iContrast);
		cope = fullFContrast(iContrast) * cope; // cope is cope now
		if (sizeFcontrast(iContrast) == 1) {
			fstat = square(cope) / varcope; // cope is replaced by the F-scores
		}
		else { // need to do it voxel by voxel
			mat tmp(sizeFcontrast(iContrast), sizeFcontrast(iContrast));
			uvec ind = find(trimatl(ones<umat>(sizeFcontrast(iContrast), sizeFcontrast(iContrast)))== 1); // indices for the lower triangular of tmp
			for (uword iVox = 0; iVox < data.n_cols; iVox++) {
				tmp(ind) = varcope.col(iVox);
				if (dof(iVox) - sizeFcontrast(iContrast) > -1) { // condition to avoid negative f-constrast
					fstat(iVox) = (dof(iVox) - sizeFcontrast(iContrast) + 1) / (dof(iVox) * sizeFcontrast(iContrast)) * as_scalar(cope.col(iVox).t() * solve(symmatl(tmp), cope.col(iVox)));
				}
				else {
					fstat(iVox) = 0;
				}
			}
		}
	}
	
	// compute the SwE for all contrasts
	void SweModel::computeSweAll(field<mat>& swe, const mat& residuals)
	{
		uword it = 0;
		
		if (modified) {
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g;
			
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j) += weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
					}
				}
				
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC) += weightF(j).cols(it3, it3 + nCovVis_g(g) - 1) * covVis_g;
					}
				}
			}
		}
		else { // classic SwE
			mat covVis_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j) += weightT(j)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
					}
				}
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC) += weightF(j).cols(it, it + nCovVis_i(i) - 1) * covVis_i;
					}
				}
			}
		}
	}

	// compute the SwE and dof for all contrasts
	void SweModel::computeSweAll(field<mat>& swe, mat& dof, const mat& residuals)
	{
		uword it = 0;
		
		if (modified) {
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g, vechCovVechV, tmp;
			
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				// compute vechCovVechV if needed
				vechCovVechV.zeros(nCovVis_g(g) * (nCovVis_g(g) + 1) / 2, covVis_g.n_cols);
				computeVechCovVechV(vechCovVechV, covVis_g, g);
				
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j) += weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
						dof.row(j) += (kron(weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)), weightT(j)(span(it3, it3 + nCovVis_g(g) - 1))) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV;
					}
				}
				
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC) += weightF(j).cols(it3, it3 + nCovVis_g(g) - 1) * covVis_g;
						if (sizeFcontrast(j) == 1) {
							dof.row(j + startFC) += (kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV;
						}
						else {
							tmp.eye(sizeFcontrast(j) * sizeFcontrast(j), sizeFcontrast(j) * sizeFcontrast(j)); // note that tr(A) = vec(I).t() * vec(A)
							dof.row(j + startFC) += ((vectorise(tmp).t() * kron(duplicationMatrix(sizeFcontrast(j)), duplicationMatrix(sizeFcontrast(j))))
																			 * (kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))))
							* vechCovVechV;;
					
						}
					}
				}
			} // end loop over g
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					dof.row(j) = 2 * square(swe(j)) / dof.row(j);
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					if (sizeFcontrast(j) == 1) {
						dof.row(j + startFC) = 2 * square(swe(j + startFC)) / dof.row(j + startFC);
					}
					else {
						uvec indDiagF(sizeFcontrast(j));
						uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
						uword it3 = 0;
						for (uword k = 0; k < sizeFcontrast(j); k++){
							indDiagF(k) = it3;
							it3++;
							for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
								indOffDiagF(it3 - k - 1) = it3;
							}
						}
						dof.row(j + startFC) = (square(arma::sum(swe(j + startFC).rows(indDiagF))) + arma::sum(square(swe(j + startFC).rows(indDiagF))) + 2 * arma::sum(square(swe(j + startFC).rows(indOffDiagF)))) / dof.row(j + startFC);
					}
				}
			}
		}
		else { // classic SwE
			mat covVis_i, swe_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe_i = weightT(j)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
						swe(j) += swe_i;
						dof.row(j) += square(swe_i) / subjectDof(i);
					}
				}
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe_i = weightF(j).cols(it, it + nCovVis_i(i) - 1) * covVis_i;
						swe(j + startFC) += swe_i;
						if (sizeFcontrast(j) == 1) {
							dof.row(j + startFC) += square(swe_i) / subjectDof(i);
						}
						else {
							uvec indDiagF(sizeFcontrast(j));
							uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
							uword it3 = 0;
							for (uword k = 0; k < sizeFcontrast(j); k++){
								indDiagF(k) = it3;
								it3++;
								for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
									indOffDiagF(it3 - k - 1) = it3;
								}
							}
							dof.row(j + startFC) += (square(arma::sum(swe_i.rows(indDiagF))) + arma::sum(square(swe_i.rows(indDiagF))) + 2 * arma::sum(square(swe_i.rows(indOffDiagF)))) / subjectDof(i);
						}
					}
				}
			} // loop over i
			// compute the dof
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					dof.row(j) = square(swe(j)) / dof.row(j);
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					if (sizeFcontrast(j) == 1) {
						dof.row(j + startFC) = square(swe(j + startFC)) / dof.row(j + startFC);
					}
					else {
						uvec indDiagF(sizeFcontrast(j));
						uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
						uword it3 = 0;
						for (uword k = 0; k < sizeFcontrast(j); k++){
							indDiagF(k) = it3;
							it3++;
							for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
								indOffDiagF(it3 - k - 1) = it3;
							}
						}
						dof.row(j + startFC) = (square(arma::sum(swe(j + startFC).rows(indDiagF))) + arma::sum(square(swe(j + startFC).rows(indDiagF))) + 2 * arma::sum(square(swe(j + startFC).rows(indOffDiagF)))) / dof.row(j + startFC);
					}
				}
			}
		}
	}
	
	// compute the SwE for all contrasts, but only for voxel iVox
	void SweModel::computeSweAll(field<mat>& swe, const mat& residuals, const uword iVox)
	{
		uword it = 0;
		
		if(modified) {
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g;
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), 1);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
				
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j).col(iVox) += weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
					}
				}
				
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC).col(iVox) += weightF(j).cols(it3, it3 + nCovVis_g(g) - 1) * covVis_g;
					}
				}
			}
		}
		else { // classic SwE
			mat covVis_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j).col(iVox) += weightT(j)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
					}
				}
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC).col(iVox) += weightF(j).cols(it, it + nCovVis_i(i) - 1) * covVis_i;
					}
				}
			}
		}
	}
	
	// compute the SwE and dof for all contrasts, but only for voxel iVox
	void SweModel::computeSweAll(field<mat>& swe, mat& dof, const mat& residuals, const uword iVox)
	{
		uword it = 0;
		uvec iVox2(1);
		iVox2(0) = iVox; // conversion into a uvec object
		if(modified) {
			uword it2 = 0;
			uword it3 = 0;
			mat covVis_g, vechCovVechV, tmp;
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), 1);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}
				
				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
	
				// compute vechCovVechV for group g and update dof
				vechCovVechV.zeros(nCovVis_g(g) * (nCovVis_g(g) + 1) / 2, covVis_g.n_cols);
				computeVechCovVechV(vechCovVechV, covVis_g, g);
				
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j).col(iVox) += weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
						dof(j, iVox) += as_scalar((kron(weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)), weightT(j)(span(it3, it3 + nCovVis_g(g) - 1))) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV); // could be optimised for voxelwise design
					}
				}
				
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC).col(iVox) += weightF(j).cols(it3, it3 + nCovVis_g(g) - 1) * covVis_g;
						if (sizeFcontrast(j) == 1) {
							dof(j + startFC, iVox) += as_scalar((kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV);
						}
						else { // note: the code below could be done once for the whole brain (to be changed later)
							tmp.eye(sizeFcontrast(j) * sizeFcontrast(j), sizeFcontrast(j) * sizeFcontrast(j)); // note that tr(A) = vec(I).t() * vec(A)
							dof(j + startFC, iVox) += as_scalar(((vectorise(tmp).t() * kron(duplicationMatrix(sizeFcontrast(j)), duplicationMatrix(sizeFcontrast(j))))
																									 * (kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))))
																									* vechCovVechV);
						}
					}
				}
			} // end loop g
			
			// compute the dof
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					dof(j, iVox) = as_scalar(square(swe(j).col(iVox)) / dof(j, iVox));
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					if (sizeFcontrast(j) == 1) {
						dof(j + startFC, iVox) = as_scalar(square(swe(j + startFC).col(iVox)) / dof(j + startFC, iVox));
					}
					else {
						uvec indDiagF(sizeFcontrast(j));
						uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
						uword it3 = 0;
						for (uword k = 0; k < sizeFcontrast(j); k++){
							indDiagF(k) = it3;
							it3++;
							for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
								indOffDiagF(it3 - k - 1) = it3;
							}
						}
						dof(j + startFC, iVox) = as_scalar((square(arma::sum(swe(j + startFC)(indDiagF, iVox2))) + arma::sum(square(swe(j + startFC)(indDiagF, iVox2))) + 2 * arma::sum(square(swe(j + startFC)(indOffDiagF, iVox2)))) / dof(j + startFC, iVox));
					}
				}
			}
		}
		else { // classic SwE
			mat covVis_i, swe_i;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe_i = weightT(j)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
						swe(j).col(iVox) += swe_i;
						dof(j, iVox) +=  as_scalar(square(swe_i) / subjectDof(i));
					}
				}
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe_i = weightF(j).cols(it, it + nCovVis_i(i) - 1) * covVis_i;
						swe(j + startFC).col(iVox) += swe_i;
						if (sizeFcontrast(j) == 1) {
							dof(j + startFC, iVox) += as_scalar(square(swe_i) / subjectDof(i));
						}
						else { // note: the code below should be done once for the whole brain (to be changed later)
							uvec indDiagF(sizeFcontrast(j));
							uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
							uword it3 = 0;
							for (uword k = 0; k < sizeFcontrast(j); k++){
								indDiagF(k) = it3;
								it3++;
								for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
									indOffDiagF(it3 - k - 1) = it3;
								}
							}
							dof(j + startFC, iVox) += as_scalar((square(arma::sum(swe_i.rows(indDiagF))) + arma::sum(square(swe_i.rows(indDiagF))) + 2 * arma::sum(square(swe_i.rows(indOffDiagF)))) / subjectDof(i));
						}
					}
				}
			} // end loop g
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					dof(j, iVox) = as_scalar(square(swe(j).col(iVox)) / dof(j, iVox));
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					if (sizeFcontrast(j) == 1) {
						dof(j + startFC, iVox) = as_scalar(square(swe(j + startFC).col(iVox)) / dof(j + startFC, iVox));
					}
					else {
						uvec indDiagF(sizeFcontrast(j));
						uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
						uword it3 = 0;
						for (uword k = 0; k < sizeFcontrast(j); k++){
							indDiagF(k) = it3;
							it3++;
							for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
								indOffDiagF(it3 - k - 1) = it3;
							}
						}
						dof(j + startFC, iVox) = as_scalar((square(arma::sum(swe(j + startFC)(indDiagF, iVox2))) + arma::sum(square(swe(j + startFC)(indDiagF, iVox2))) + 2 * arma::sum(square(swe(j + startFC)(indOffDiagF, iVox2)))) / dof(j + startFC, iVox));
					}
				}
			}
		}
	}

	// estimate the model & stats for the classic SwE (parametric inferences)
	void SweModel::computeStatsClassicSwe(const mat& data)
	{
		// setup variables
		mat pe(designMatrix.n_cols, nVox); // parameter estimates
		
		field<mat> swe(nContrasts); // contrasted sandwich estimators
		startFC = 0; // 0 if only F-contrasts
		if (!doFOnly) {
			for (uword j = 0; j < tc.n_rows; j++) swe(j).zeros(1, nVox);
			startFC = tc.n_rows; // tc.n_rows if also t-contrasts
		}
		if (!doTOnly) {
			for (uword j = 0; j < fc.n_rows; j++) swe(j + startFC).zeros(nCovCFSweCF(j), nVox);
		}
		mat residuals;
		mat swe_i; // subject contribution to swe
		mat covVis_i; // subject covariance estimate
		mat dof; // number of degrees of freedom
		if(!wb) dof.zeros(nContrasts, nVox); // number of degrees of freedom (only for parametric tests)
		uword it, it4; // counter for loops
		
		// estimate the model
		if (verbose) cout << "Computing PEs, VARCOPEs & effective numbers of degrees of freedom: " << flush;
		if (voxelWiseDesign){ // need to do the job voxel by voxel
			uvec v = linspace<uvec>(0, nVox - 1, 101);
			it4 = 0;
			for(uword iVox = 0; iVox < nVox; iVox++) {
				adjustDesign(iVox); // change the design matrix and its pseudoinverse
				pe.col(iVox) = computePe(data.col(iVox));
				residuals = computeAdjustedResiduals(data.col(iVox), pe.col(iVox));
				it = 0;
				for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
					covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
					// for t-contrasts if required
					if (!doFOnly) {
						for (uword j = 0; j < tc.n_rows; j++) {
							swe_i = weightT(j)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
							swe(j).col(iVox) += swe_i;
							if(!wb) dof(j, iVox) +=  as_scalar(square(swe_i) / subjectDof(i));
						}
					}
					// for F-contrasts
					if (!doTOnly) {
						for (uword j = 0; j < fc.n_rows; j++){
							swe_i = weightF(j).cols(it, it + nCovVis_i(i) - 1) * covVis_i;
							swe(j + startFC).col(iVox) += swe_i;
							if(!wb) {
								if (sizeFcontrast(j) == 1) {
									dof(j + startFC, iVox) += as_scalar(square(swe_i) / subjectDof(i));
								}
								else { // note: the code below should be done once for the whole brain (to be changed later)
									uvec indDiagF(sizeFcontrast(j));
									uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
									//				indDiagF = linspace(0, sizeFcontrast(j) - 1, sizeFcontrast(j));
									//				indDiagF = ((2 * sizeFcontrast(j) + 1) * indDiagF - square(indDiagF)) / 2 + it2;
									uword it3 = 0;
									for (uword k = 0; k < sizeFcontrast(j); k++){
										indDiagF(k) = it3;
										it3++;
										for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
											indOffDiagF(it3 - k - 1) = it3;
										}
									}
									dof(j + startFC, iVox) += as_scalar((square(arma::sum(swe_i.rows(indDiagF))) + arma::sum(square(swe_i.rows(indDiagF))) + 2 * arma::sum(square(swe_i.rows(indOffDiagF)))) / subjectDof(i));
								}
							}
						}
					}
				}
				if(any(v==iVox) && verbose){
					if (it4 == 0) cout << "0%" << flush;
					if (it4 > 0 && it4 <= 10) cout << "\b\b" << it4 << "%" << flush;
					if (it4 > 10) cout << "\b\b\b" << it4 << "%" << flush;
					it4++;
				}
			}
			if(verbose) cout << "\b\b\b\b" << flush;
		}
		else { // same design at every voxel -> do everything in one go
			pe = computePe(data);
			residuals = computeAdjustedResiduals(data, pe);
			it = 0;
			for (uword i = 0; i < subjectCovIndex.n_elem; it += nCovVis_i(i), i++) {
				covVis_i = residuals.rows(subjectCovIndex(i).col(0)) % residuals.rows(subjectCovIndex(i).col(1));
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe_i = weightT(j)(span(it, it + nCovVis_i(i) - 1)) * covVis_i;
						swe(j) += swe_i;
						if(!wb) dof.row(j) += square(swe_i) / subjectDof(i);
					}
				}
				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe_i = weightF(j).cols(it, it + nCovVis_i(i) - 1) * covVis_i;
						swe(j + startFC) += swe_i;
						if(!wb) {
							if (sizeFcontrast(j) == 1) {
								dof.row(j + startFC) += square(swe_i) / subjectDof(i);
							}
							else {
								uvec indDiagF(sizeFcontrast(j));
								uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
								//				indDiagF = linspace(0, sizeFcontrast(j) - 1, sizeFcontrast(j));
								//				indDiagF = ((2 * sizeFcontrast(j) + 1) * indDiagF - square(indDiagF)) / 2 + it2;
								uword it3 = 0;
								for (uword k = 0; k < sizeFcontrast(j); k++){
									indDiagF(k) = it3;
									it3++;
									for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
										indOffDiagF(it3 - k - 1) = it3;
									}
								}
								dof.row(j + startFC) += (square(arma::sum(swe_i.rows(indDiagF))) + arma::sum(square(swe_i.rows(indDiagF))) + 2 * arma::sum(square(swe_i.rows(indOffDiagF)))) / subjectDof(i);
							}
						}
					}
				}
			}
		}
		// compute the dof
		if (!doFOnly) {
			for (uword j = 0; j < tc.n_rows; j++) {
				dof.row(j) = square(swe(j)) / dof.row(j);
			}
		}
		if (!doTOnly) {
			for (uword j = 0; j < fc.n_rows; j++){
				if (sizeFcontrast(j) == 1) {
					dof.row(j + startFC) = square(swe(j + startFC)) / dof.row(j + startFC);
				}
				else {
					uvec indDiagF(sizeFcontrast(j));
					uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
					//				indDiagF = linspace(0, sizeFcontrast(j) - 1, sizeFcontrast(j));
					//				indDiagF = ((2 * sizeFcontrast(j) + 1) * indDiagF - square(indDiagF)) / 2 + it2;
					uword it3 = 0;
					for (uword k = 0; k < sizeFcontrast(j); k++){
						indDiagF(k) = it3;
						it3++;
						for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
							indOffDiagF(it3 - k - 1) = it3;
						}
					}
					dof.row(j + startFC) = (square(arma::sum(swe(j + startFC).rows(indDiagF))) + arma::sum(square(swe(j + startFC).rows(indDiagF))) + 2 * arma::sum(square(swe(j + startFC).rows(indOffDiagF)))) / dof.row(j + startFC);
				}
			}
		}

		if (outputGlm) saveImage(pe, out_fileroot + "_" + LABEL_PE);
		if (verbose) cout << "done" << endl;
		residuals.reset(); // to release memory (TBC)
		
		// compute cope, scores, equivalentScores, p-values and save results
		mat cope, tmp;
		mat equivalentScores(1, nVox);
		mat pValues(1, nVox);
		
		// for t-contrasts if required
		//
		if (!doFOnly) {
			if (verbose) cout << "Computing COPEs & scores for t-contrasts and saving the corresponding results: " << flush;
			for (uword j = 0; j < tc.n_rows; j++) {
				cope = tc.row(j) * pe;
				if (outputGlm) saveImage(cope, out_fileroot + "_" + LABEL_COPE + "_" + LABEL_TSTAT + num2str(j + 1));
				cope = cope / sqrt(swe(j)); // cope is replaced by the t-scores
				T2ZP(cope, equivalentScores, pValues, dof.row(j), logP);
				if (outputGlm) saveImage(swe(j), out_fileroot + "_" + LABEL_VARCOPE + "_" + LABEL_TSTAT + num2str(j + 1));
				if (outputDof){
					tmp = dof.row(j); // does not compile when fed directly to the method saveImage...
					saveImage(tmp, out_fileroot + "_" + LABEL_DOF + "_" + LABEL_TSTAT + num2str(j + 1));
				}
				if (outputRaw) saveImage(cope, out_fileroot + "_" + LABEL_TSTAT + num2str(j + 1));
				if (outputEquivalent) saveImage(equivalentScores, out_fileroot + "_" + LABEL_EQUIVALENT_Z + "_" + LABEL_TSTAT + num2str(j + 1));
				if (logP){
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
				}
				else {
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" +LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
				}
			}
			if (verbose) cout << " done" << endl;
		}
		// for F-contrasts
		if (!doTOnly) {
			if (verbose) cout << "Computing COPEs & scores for f-contrasts and saving the corresponding results: " << flush;
			for (uword j = 0; j < fc.n_rows; j++){
				cope = fullFContrast(j) * pe;
				if (outputGlm) saveImage(cope, out_fileroot + "_" + LABEL_COPE + "_" + LABEL_FSTAT + num2str(j + 1));
				if (sizeFcontrast(j) == 1) {
					cope = square(cope) / swe(j + startFC); // cope is replaced by the F-scores
					F2XP(cope, equivalentScores, pValues, sizeFcontrast(j), dof.row(j + startFC), logP);
					if (outputRaw) saveImage(cope, out_fileroot + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				else { // need to do it voxel by voxel
					tmp.zeros(sizeFcontrast(j), sizeFcontrast(j));
					uvec ind = find(trimatl(ones<umat>(sizeFcontrast(j), sizeFcontrast(j)))== 1); // indices for the lower triangular of tmp2
					mat tmp2(1, nVox);
					for (uword iVox = 0; iVox < nVox; iVox++) {
						tmp(ind) = swe(j + startFC).col(iVox);
						if (dof(j + startFC, iVox) - sizeFcontrast(j) > -1) { // condition to avoid negative f-constrast
						tmp2(iVox) = (dof(j + startFC, iVox) - sizeFcontrast(j) + 1)/ (dof(j + startFC, iVox) * sizeFcontrast(j)) * as_scalar(cope.col(iVox).t() * solve(symmatl(tmp), cope.col(iVox)));
						}
						else {
							tmp2(iVox) = 0;
						}
					}
					F2XP(tmp2, equivalentScores, pValues, sizeFcontrast(j), dof.row(j + startFC), logP);
					if (outputRaw) saveImage(tmp2, out_fileroot + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				if (outputGlm) saveImage(swe(j + startFC), out_fileroot + "_" + LABEL_VARCOPE + "_" + LABEL_FSTAT + num2str(j + 1));
				if (outputDof) {
					tmp = dof.row(j + startFC); // does not compile when fed directly to the method saveImage...
					saveImage(tmp, out_fileroot + "_" + LABEL_DOF + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				if (outputEquivalent) saveImage(equivalentScores, out_fileroot + "_" + LABEL_EQUIVALENT_CHISQUARE + "_" + LABEL_FSTAT + num2str(j + 1));
				if (logP){
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				else {
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
				}
			}
			if (verbose) cout << " done" << endl;
		}
	}
	
	// estimate the model & stats for the modified SwE (parametric inferences)
	void SweModel::computeStatsModifiedSwe(const mat& data)
	{
		// setup variables
		mat pe(designMatrix.n_cols, nVox); // parameter estimates
		
		field<mat> swe(nContrasts); // contrasted sandwich estimators
		startFC = 0; // 0 if only F-contrasts
		if (!doFOnly) {
			for (uword j = 0; j < tc.n_rows; j++) swe(j).zeros(1, nVox);
			startFC = tc.n_rows; // tc.n_rows if also t-contrasts
		}
		if (!doTOnly) {
			for (uword j = 0; j < fc.n_rows; j++) swe(j + startFC).zeros(nCovCFSweCF(j), nVox);
		}
		mat residuals;
		mat covVis_g; // group covariance estimate
		mat vechCovVechV;
		mat dof;
		mat tmp;
		if(!wb) {
			dof.zeros(nContrasts, nVox); // number of degrees of freedom (only for parametric tests)
		}
		uword it, it2, it3, it4; // counters for loops
		
		// estimate the model
		if (verbose) cout << "Computing PEs, VARCOPEs & effective numbers of degrees of freedom: " << flush;
		if (voxelWiseDesign){ // need to do the job voxel by voxel
			uvec v = linspace<uvec>(0, nVox - 1, 101);
			it4 = 0;
			for(uword iVox = 0; iVox < nVox; iVox++) {
				adjustDesign(iVox); // change the design matrix and its pseudoinverse
				pe.col(iVox) = computePe(data.col(iVox));
				residuals = computeAdjustedResiduals(data.col(iVox), pe.col(iVox));
				it = 0;
				it2 = 0;
				it3 = 0;
				for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
					covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
					
					// first, diagonal elements
					for (uword i = 0; i < n_g(g); i++, it++) {
						covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
					}
					
					// second, off-diag. elements
					for (uword i = 0; i < n_g(g); i++) {
						for (uword j = i + 1; j < n_g(g); j++, it2++) {
							if (indOffDiagRes(it2).n_rows > 0)
								covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
						}
					}
					
					// ensure positive-definitiveness
					checkAndMakePSD(covVis_g);
					
					// compute vechCovVechV if needed
					if(!wb) {
						vechCovVechV.zeros(nCovVis_g(g) * (nCovVis_g(g) + 1) / 2, 1);
						computeVechCovVechV(vechCovVechV, covVis_g, g);
					}
					
					// for t-contrasts if required
					if (!doFOnly) {
						for (uword j = 0; j < tc.n_rows; j++) {
							swe(j).col(iVox) += weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
							if(!wb) {
								dof(j, iVox) += as_scalar((kron(weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)), weightT(j)(span(it3, it3 + nCovVis_g(g) - 1))) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV); // could be optimised for voxelwise design
							}
						}
					}
					// for F-contrasts
					if (!doTOnly) {
						for (uword j = 0; j < fc.n_rows; j++){
							swe(j + startFC).col(iVox) += weightF(j).cols(it3, it3 + nCovVis_g(g) - 1) * covVis_g;
							if(!wb) {
								if (sizeFcontrast(j) == 1) {
									dof(j + startFC, iVox) += as_scalar((kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV);
								}
								else { // note: the code below could be done once for the whole brain (to be changed later)
									tmp.eye(sizeFcontrast(j) * sizeFcontrast(j), sizeFcontrast(j) * sizeFcontrast(j)); // note that tr(A) = vec(I).t() * vec(A)
									dof(j + startFC, iVox) += as_scalar(((vectorise(tmp).t() * kron(duplicationMatrix(sizeFcontrast(j)), duplicationMatrix(sizeFcontrast(j))))
																						* (kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))))
																						* vechCovVechV);
								}
							}
						}
					}
				}
				if(any(v==iVox) && verbose){
					if (it4 == 0) cout << "0%" << flush;
					if (it4 > 0 && it4 <= 10) cout << "\b\b" << it4 << "%" << flush;
					if (it4 > 10) cout << "\b\b\b" << it4 << "%" << flush;
					it4++;
				}
			}
			if(verbose) cout << "\b\b\b\b" << flush;
		}
		else { // same design at every voxel -> do everything in one go
			pe = computePe(data);
			residuals = computeAdjustedResiduals(data, pe);
			it = 0;
			it2 = 0;
			it3 = 0;
			for (uword g = 0; g < uGroup.n_elem; it3 += nCovVis_g(g), g++) {
				covVis_g.zeros(nCovVis_g(g), residuals.n_cols);
				
				// first, diagonal elements
				for (uword i = 0; i < n_g(g); i++, it++) {
					covVis_g.row(indDiag(it)) = mean(square(residuals.rows(indDiagRes(it))));
				}
				
				// second, off-diag. elements
				for (uword i = 0; i < n_g(g); i++) {
					for (uword j = i + 1; j < n_g(g); j++, it2++) {
						if (indOffDiagRes(it2).n_rows > 0)
							covVis_g.row(indOffDiag(it2)) = mean(residuals.rows(indOffDiagRes(it2).col(0)) % residuals.rows(indOffDiagRes(it2).col(1))) % sqrt((covVis_g.row(indDiag(indCorrDiag(it2, 0))) % covVis_g.row(indDiag(indCorrDiag(it2, 1)))) / (mean(square(residuals.rows(indOffDiagRes(it2).col(0)))) % mean(square(residuals.rows(indOffDiagRes(it2).col(1))))));
					}
				}

				// ensure positive-definitiveness
				checkAndMakePSD(covVis_g);
	
				// compute vechCovVechV if needed
				if(!wb) {
					vechCovVechV.zeros(nCovVis_g(g) * (nCovVis_g(g) + 1) / 2, covVis_g.n_cols);
					computeVechCovVechV(vechCovVechV, covVis_g, g);
				}
			
				// for t-contrasts if required
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						swe(j) += weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)) * covVis_g;
						if(!wb) {
							dof.row(j) += (kron(weightT(j)(span(it3, it3 + nCovVis_g(g) - 1)), weightT(j)(span(it3, it3 + nCovVis_g(g) - 1))) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV;
						}
					}
				}

				// for F-contrasts
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						swe(j + startFC) += weightF(j).cols(it3, it3 + nCovVis_g(g) - 1) * covVis_g;
						if(!wb) {
							if (sizeFcontrast(j) == 1) {
								dof.row(j + startFC) += (kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))) * vechCovVechV;
							}
							else {
								tmp.eye(sizeFcontrast(j) * sizeFcontrast(j), sizeFcontrast(j) * sizeFcontrast(j)); // note that tr(A) = vec(I).t() * vec(A)
								dof.row(j + startFC) += ((vectorise(tmp).t() * kron(duplicationMatrix(sizeFcontrast(j)), duplicationMatrix(sizeFcontrast(j))))
															* (kron(weightF(j).cols(it3, it3 + nCovVis_g(g) - 1), weightF(j).cols(it3, it3 + nCovVis_g(g) - 1)) * duplicationMatrix(nCovVis_g(g))))
															* vechCovVechV;;
							}
						}
					}
				}
			}
		}

		// compute the dof
		if (!wb) {
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					dof.row(j) = 2 * square(swe(j)) / dof.row(j);
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					if (sizeFcontrast(j) == 1) {
						dof.row(j + startFC) = 2 * square(swe(j + startFC)) / dof.row(j + startFC);
					}
					else {
						uvec indDiagF(sizeFcontrast(j));
						uvec indOffDiagF(nCovCFSweCF(j) - sizeFcontrast(j));
						//				indDiagF = linspace(0, sizeFcontrast(j) - 1, sizeFcontrast(j));
						//				indDiagF = ((2 * sizeFcontrast(j) + 1) * indDiagF - square(indDiagF)) / 2 + it2;
						uword it3 = 0;
						for (uword k = 0; k < sizeFcontrast(j); k++){
							indDiagF(k) = it3;
							it3++;
							for (uword kk = k + 1; kk < sizeFcontrast(j); kk++, it3++){
								indOffDiagF(it3 - k - 1) = it3;
							}
						}
						dof.row(j + startFC) = (square(arma::sum(swe(j + startFC).rows(indDiagF))) + arma::sum(square(swe(j + startFC).rows(indDiagF))) + 2 * arma::sum(square(swe(j + startFC).rows(indOffDiagF)))) / dof.row(j + startFC);
					}
				}
			}
		}
		
		if (outputGlm) saveImage(pe, out_fileroot  + "_" + LABEL_PE);
		if (verbose) cout << "done" << endl;
		residuals.reset(); // to release memory (TBC)
		
		// compute cope, scores, equivalentScores, p-values and save results
		mat cope;
		mat equivalentScores(1, nVox);
		mat pValues(1, nVox);
		
		// for t-contrasts if required
		if (!doFOnly) {
			if (verbose) cout << "Computing COPEs & scores for t-contrasts and saving the corresponding results: " << flush;
			for (uword j = 0; j < tc.n_rows; j++) {
				cope = tc.row(j) * pe;
				if (outputGlm) saveImage(cope, out_fileroot + "_" + LABEL_COPE + "_" + LABEL_TSTAT + num2str(j + 1));
				cope = cope / sqrt(swe(j)); // cope is replaced by the t-scores
				T2ZP(cope, equivalentScores, pValues, dof.row(j), logP);
				if (outputGlm) saveImage(swe(j), out_fileroot + "_" + LABEL_VARCOPE + "_" + LABEL_TSTAT + num2str(j + 1));
				if (outputDof){
					tmp = dof.row(j); // does not compile when fed directly to the method saveImage...
					saveImage(tmp, out_fileroot + "_" + LABEL_DOF + "_" + LABEL_TSTAT + num2str(j + 1));
				}
				if (outputRaw) saveImage(cope, out_fileroot + "_" + LABEL_TSTAT + num2str(j + 1));
				if (outputEquivalent) saveImage(equivalentScores, out_fileroot + "_" + LABEL_EQUIVALENT_Z + "_" + LABEL_TSTAT + num2str(j + 1));
				if (logP){
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
				}
				else {
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" +LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
				}
			}
			if (verbose) cout << "done" << endl;
		}
		// for F-contrasts
		if (!doTOnly) {
			if (verbose) cout << "Computing COPEs & scores for f-contrasts and saving the corresponding results: " << flush;
			for (uword j = 0; j < fc.n_rows; j++){
				cope = fullFContrast(j) * pe;
				if (outputGlm) saveImage(cope, out_fileroot + "_" + LABEL_COPE + "_" + LABEL_FSTAT + num2str(j + 1));
				if (sizeFcontrast(j) == 1) {
					cope = square(cope) / swe(j + startFC); // cope is replaced by the F-scores
					F2XP(cope, equivalentScores, pValues, sizeFcontrast(j), dof.row(j + startFC), logP);
					if (outputRaw) saveImage(cope, out_fileroot + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				else { // need to do it voxel by voxel
					tmp.zeros(sizeFcontrast(j), sizeFcontrast(j));
					uvec ind = find(trimatl(ones<umat>(sizeFcontrast(j), sizeFcontrast(j)))== 1); // indices for the lower triangular of tmp2
					mat tmp2(1, nVox);
					for (uword iVox = 0; iVox < nVox; iVox++) {
						tmp(ind) = swe(j + startFC).col(iVox);
						if (dof(j + startFC, iVox) - sizeFcontrast(j) > -1) { // condition to avoid negative f-constrast
							tmp2(iVox) = (dof(j + startFC, iVox) - sizeFcontrast(j) + 1)/ (dof(j + startFC, iVox) * sizeFcontrast(j)) * as_scalar(cope.col(iVox).t() * solve(symmatl(tmp), cope.col(iVox)));
						}
						else {
							tmp2(iVox) = 0;
						}
					}
					F2XP(tmp2, equivalentScores, pValues, sizeFcontrast(j), dof.row(j + startFC), logP);
					if (outputRaw) saveImage(tmp2, out_fileroot + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				if (outputGlm) saveImage(swe(j + startFC), out_fileroot + "_" + LABEL_VARCOPE + "_" + LABEL_FSTAT + num2str(j + 1));
				if (outputDof){
					tmp = dof.row(j + startFC); // does not compile when fed directly to the method saveImage...
					saveImage(tmp, out_fileroot + "_" + LABEL_DOF + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				if (outputEquivalent) saveImage(equivalentScores, out_fileroot + "_" + LABEL_EQUIVALENT_CHISQUARE + "_" + LABEL_FSTAT + num2str(j + 1));
				if (logP){
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
				}
				else {
					if (outputUncorr) saveImage(pValues, out_fileroot + "_" + LABEL_VOX + "_" +LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
				}
			}
			if (verbose) cout << "done" << endl;
		}
	}
	
	// estimate the model & stats for the modified SwE using the WB procedure
	void SweModel::computeStatsWb(const mat& data)
	{
		// setup variables
		mat pe(designMatrix.n_cols, nVox); // parameter estimates
		
		field<mat> swe(nContrasts); // contrasted sandwich estimators
		mat	sweB;
		startFC = 0; // 0 if only F-contrasts
		if (!doFOnly) {
			for (uword j = 0; j < tc.n_rows; j++) swe(j).zeros(1, nVox);
			startFC = tc.n_rows; // tc.n_rows if also t-contrasts
		}
		if (!doTOnly) {
			for (uword j = 0; j < fc.n_rows; j++) swe(j + startFC).zeros(nCovCFSweCF(j), nVox);
		}
		mat residuals, dof;
		mat tmp, tmp2; // temporary variable containing various temporary variable
		vec tmpVec;
		mat copeB, varcopeB;
		mat peB;
		mat residualsB;
		mat restrictedMatrix;
		mat restrictedFittedData;
		mat restrictedResiduals;
		mat resampledData;
		uword it4; // counters for loops
		mat equivalentScore(nContrasts, nVox); // will contain the original equivalent score
		mat scoreB, zScoreB, xScoreB, eqScoreB, dofB, pValueB, scoreBVox;
		mat maxScore(nContrasts, nBootstrap + 1);
		maxScore.fill(-datum::inf);
		
		mat tfceScore, tfceScoreB, maxTfceScore, eqZScoreB, oneMinusUncPTfce, oneMinusFwePTfce;
		if (doTfce) {
			tfceScore.set_size(nContrasts, nVox);
			tfceScoreB.set_size(1, nVox);
			eqZScoreB.set_size(1,nVox);
			maxTfceScore.set_size(nContrasts, nBootstrap + 1);
			maxTfceScore.fill(-datum::inf);
		}
		
		field<uvec> clusterExtentSizes(nContrasts);
		field<vec> clusterMassSizes(nContrasts);
		field<uvec> clusterExtentLabels(nContrasts);
		field<uvec> clusterMassLabels(nContrasts);
		uvec clusterExtentLabelsB;
		uvec clusterMassLabelsB;
		uvec clusterExtentSizesB;
		vec clusterMassSizesB;
		mat maxClusterExtentSizes, maxClusterMassSizes;
		if (clusterThresholdT > 0 || clusterThresholdF > 0) maxClusterExtentSizes.zeros(nContrasts, nBootstrap + 1);
		if (clusterMassThresholdT > 0 || clusterMassThresholdF > 0) maxClusterMassSizes.zeros(nContrasts, nBootstrap + 1);
		dof.zeros(nContrasts, nVox);

		// estimate the model
		if (voxelWiseDesign){ // need to do the job voxel by voxel
			mat oneMinusUncP(nContrasts, nVox);
			mat oneMinusFweP(nContrasts, nVox);
			oneMinusUncP.fill(nBootstrap);
			oneMinusFweP.fill(nBootstrap);
			scoreB.set_size(nContrasts, nVox);
			scoreBVox.set_size(nContrasts, nVox);
			field<mat> cope(nContrasts);
			if (doTfce) {
				oneMinusUncPTfce.set_size(nContrasts, nVox);
				oneMinusFwePTfce.set_size(nContrasts, nVox);
				oneMinusUncPTfce.fill(nBootstrap);
				oneMinusFwePTfce.fill(nBootstrap);
			}
			mat tmpScoreB, tmpDofB;
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) cope(j).set_size(1, nVox);
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++) cope(j + startFC).set_size(sizeFcontrast(j), nVox);
			}
			if (verbose) cout << "Computing original scores: " << flush;
			uvec v = linspace<uvec>(0, nVox - 1, 101);
			it4 = 0; // counter for indicating percentage
			
			// set up relevant variables
			eqScoreB.set_size(1, nVox);
			pValueB.set_size(1, nVox);
			dofB.set_size(nContrasts, nVox);

			for(uword iVox = 0; iVox < nVox; iVox++) {
				adjustDesign(iVox); // change the design matrix and its pseudoinverse
				pe.col(iVox) = computePe(data.col(iVox));
				residuals = computeAdjustedResiduals(data.col(iVox), pe.col(iVox));
				
				//compute the SwE and dof for this voxel
				computeSweAll(swe, dofB, residuals, iVox); // need to get the dof
				
				// original statistics for this voxel
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++) {
						cope(j).col(iVox) = tc.row(j) * pe.col(iVox);
						scoreB(j, iVox) = as_scalar(cope(j).col(iVox) / sqrt(swe(j).col(iVox)));
					}
				}
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						if (sizeFcontrast(j) == 1) {
							cope(j + startFC).col(iVox) = fullFContrast(j) * pe.col(iVox);
							scoreB(j + startFC, iVox) = as_scalar(square(cope(j + startFC).col(iVox)) /swe(j).col(iVox));
						}
						else {
							cope(j + startFC).col(iVox) = fullFContrast(j) * pe.col(iVox);
							tmp.set_size(sizeFcontrast(j), sizeFcontrast(j));
							uvec ind = find(trimatl(ones<umat>(sizeFcontrast(j), sizeFcontrast(j)))== 1); // indices for the lower triangular of tmp2
							tmp(ind) = swe(j + startFC).col(iVox);
							scoreB(j + startFC, iVox) = as_scalar(cope(j + startFC).col(iVox).t() * solve(symmatl(tmp), cope(j + startFC).col(iVox))) / sizeFcontrast(j); // will be corrected later with degrees of freedom if clusterwise inference is needed
						}
					}
				}
				if(any(v==iVox) && verbose){ // progression display
					if (it4 == 0) cout << "0%" << flush;
					if (it4 > 0 && it4 <= 10) cout << "\b\b" << it4 << "%" << flush;
					if (it4 > 10) cout << "\b\b\b" << it4 << "%" << flush;
					it4++;
				}
			} // iVox loop
			
			if(verbose) cout << "\b\b\b\bdone" << endl;
			if(verbose) cout << "Computing original equivalent scores and saving original images: " << flush;
			
			if (outputGlm) saveImage(pe, out_fileroot  + "_" + LABEL_PE);
			// compute original scores
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					string statLabel = "_" + LABEL_TSTAT + num2str(j + 1);
					if (outputGlm) saveImage(cope(j), out_fileroot + "_" + LABEL_COPE + statLabel);
					if (outputRaw){
						tmp = scoreB.row(j); // convert as saveImage does not seem to work directely with score.row(j)
						saveImage(tmp, out_fileroot  + statLabel);
					}
					if (outputGlm) saveImage(swe(j), out_fileroot + "_" + LABEL_VARCOPE + statLabel);
					
					// convert into equivalent scores and parametric p-values
					T2ZP(tmp, eqScoreB, pValueB, dofB.row(j), false);
					
					// compute the max equivalentScore
					maxScore(j, 0) = eqScoreB.max();
					
					// save the equivalent score where needed
					equivalentScore.row(j) = eqScoreB;
					if (outputEquivalent) saveImage(eqScoreB, out_fileroot + "_" + LABEL_EQUIVALENT_Z + "_" + LABEL_TSTAT + num2str(j + 1));
					
					if (clusterThresholdT > 0){
						if (clusterThresholdT < 1) // pValue threshold
							computeClusterStatsT(pValueB, clusterExtentLabels(j), clusterExtentSizes(j));
						else // equivalent z-score threshold
							computeClusterStatsT(eqScoreB, clusterExtentLabels(j), clusterExtentSizes(j));
						
						tmp2.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterExtentLabels(j)(i) > 0) tmp2(i) = clusterExtentSizes(j)(clusterExtentLabels(j)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERE + statLabel);
						if (clusterExtentSizes(j).n_elem > 0 ) maxClusterExtentSizes(j, 0) = clusterExtentSizes(j).max();
					}
					if (clusterMassThresholdT > 0){
						if (clusterMassThresholdT < 1) // pValue threshold
							computeClusterStatsT(pValueB, eqScoreB, clusterMassLabels(j), clusterMassSizes(j));
						else // equivalent z-score threshold
							computeClusterStatsT(eqScoreB, clusterMassLabels(j), clusterMassSizes(j));
						
						tmp2.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterMassLabels(j)(i) > 0) tmp2(i) = clusterMassSizes(j)(clusterMassLabels(j)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERM + statLabel);
						if (clusterMassSizes(j).n_elem > 0 ) maxClusterMassSizes(j, 0) = clusterMassSizes(j).max();
					}
					if (doTfce) {
						if (!tfceDeltaOverride)
							tfceDelta(j) = eqScoreB.max() / 100.0;  // i.e. 100 subdivisions of the max input stat height
						if ( tfceDelta(j) <= 0 ) {
							cout << "Warning: the original statistic image for contrast t" << j << " contains no positive values, and cannot be processed with TFCE. A blank output image will be created." << endl;
							tfceScoreB.zeros();
							maxTfceScore(j, 0) = 0;
							oneMinusUncPTfce.row(j).zeros();
							oneMinusFwePTfce.row(j).zeros();
						}
						else {
							tfce(eqScoreB, tfceScoreB, j);
							maxTfceScore(j, 0) = tfceScoreB.max();
						}
						// save original tfce scores
						tfceScore.row(j) = tfceScoreB;
						saveImage(tfceScoreB, out_fileroot + "_" + LABEL_TFCE + statLabel);
					}
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++) {
					string statLabel = "_" + LABEL_FSTAT + num2str(j + 1);
					if (outputGlm) saveImage(cope(j + startFC), out_fileroot + "_" + LABEL_COPE + statLabel);
					if (outputRaw) {
						tmp = scoreB.row(j + startFC); // convert as saveImage does not seem to work directely with score.row(j)
						saveImage(tmp, out_fileroot  + statLabel);
					}
					if (outputGlm) saveImage(swe(j + startFC), out_fileroot + "_" + LABEL_VARCOPE + statLabel);
					// convert into equivalent scores and parametric p-values
					if (sizeFcontrast(j) > 1) {
						tmp = tmp % ((dofB.row(j + startFC) - sizeFcontrast(j) + 1) / dofB.row(j + startFC));
						uvec tmpInd = find(tmp < 0);
						tmp(tmpInd) = zeros<mat>(1, tmpInd.n_elem);
					}
					
					// convert into parametric scores or p-values (and also Z-scores if tfce needed)
					if (doTfce)
						F2XZP(tmp, eqScoreB, eqZScoreB, pValueB, sizeFcontrast(j), dofB.row(j + startFC), false);
					else
						F2XP(tmp, eqScoreB, pValueB, sizeFcontrast(j), dofB.row(j + startFC), false);
					
					// compute the max equivalentScore
					maxScore(j + startFC, 0) = eqScoreB.max();
					
					// save the equivalent score where needed
					equivalentScore.row(j + startFC) = eqScoreB;
					if (outputEquivalent) saveImage(eqScoreB, out_fileroot + "_" + LABEL_EQUIVALENT_CHISQUARE + "_" + LABEL_FSTAT + num2str(j + 1));
					if (clusterThresholdF > 0){
						if (clusterThresholdF < 1) // pValue threshold
							computeClusterStatsF(pValueB, clusterExtentLabels(j + startFC), clusterExtentSizes(j + startFC));
						else // equivalent chi^2-score threshold
							computeClusterStatsF(eqScoreB, clusterExtentLabels(j + startFC), clusterExtentSizes(j + startFC));
						
						tmp2.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterExtentLabels(j + startFC)(i) > 0) tmp2(i) = clusterExtentSizes(j + startFC)(clusterExtentLabels(j + startFC)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERE + statLabel);
						if (clusterExtentSizes(j + startFC).n_elem > 0) maxClusterExtentSizes(j + startFC, 0) = clusterExtentSizes(j + startFC).max();
					}
					if (clusterMassThresholdF > 0){
						if (clusterMassThresholdF < 1) // pValue threshold
							computeClusterStatsF(pValueB, eqScoreB, clusterMassLabels(j + startFC), clusterMassSizes(j + startFC));
						else // equivalent chi^2-score threshold
							computeClusterStatsF(eqScoreB, clusterMassLabels(j + startFC), clusterMassSizes(j + startFC));

						tmp2.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterMassLabels(j + startFC)(i) > 0) tmp2(i) = clusterMassSizes(j + startFC)(clusterMassLabels(j + startFC)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERM + statLabel);
						if (clusterMassSizes(j + startFC).n_elem > 0) maxClusterMassSizes(j + startFC, 0) = clusterMassSizes(j + startFC).max();
					}
					if (doTfce) {
						if (!tfceDeltaOverride)
							tfceDelta(j + startFC) = eqZScoreB.max() / 100.0;  // i.e. 100 subdivisions of the max input stat height
						if ( tfceDelta(j + startFC) <= 0 ) {
							cout << "Warning: the original statistic image for contrast t" << j << " contains no positive values, and cannot be processed with TFCE. A blank output image will be created." << endl;
							tfceScoreB.zeros();
							maxTfceScore(j + startFC, 0) = 0;
							oneMinusUncPTfce.row(j + startFC).zeros();
							oneMinusFwePTfce.row(j + startFC).zeros();
						}
						else {
							tfce(eqZScoreB, tfceScoreB, j + startFC);
							maxTfceScore(j + startFC, 0) = tfceScoreB.max();
						}
						// save original tfce scores
						tfceScore.row(j + startFC) = tfceScoreB;
						saveImage(tfceScoreB, out_fileroot + "_" + LABEL_TFCE + statLabel);
					}
				}
			}
			if(verbose) cout << "done" << endl;
			
			if (!voxelwiseOutput && !outputUncorr && !(clusterThresholdT > 0 || clusterThresholdF > 0 || clusterMassThresholdT > 0 || clusterMassThresholdF > 0))
				return; // stop here, no need to go further if the user did not require p-values
			
			// set up cluster-wise variables
			field<mat> oneMinusFwePClusterExtent(nContrasts);
			field<mat> oneMinusFwePClusterMass(nContrasts);
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					if (clusterThresholdT > 0) {
						oneMinusFwePClusterExtent(j).set_size(clusterExtentSizes(j).n_elem, 1);
						oneMinusFwePClusterExtent(j).fill(nBootstrap);
					}
					if (clusterMassThresholdT > 0) {
						oneMinusFwePClusterMass(j).set_size(clusterMassSizes(j).n_elem, 1);
						oneMinusFwePClusterMass(j).fill(nBootstrap);
					}
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++) {
					if (clusterThresholdF > 0) {
						oneMinusFwePClusterExtent(j + startFC).set_size(clusterExtentSizes(j + startFC).n_elem, 1);
						oneMinusFwePClusterExtent(j + startFC).fill(nBootstrap);
					}
					if (clusterMassThresholdF > 0) {
						oneMinusFwePClusterMass(j + startFC).set_size(clusterMassSizes(j + startFC).n_elem, 1);
						oneMinusFwePClusterMass(j + startFC).fill(nBootstrap);
					}
				}
			}

			// WB procedure
			if(verbose) cout << "Computing non-parametric p-values for all contrasts: bootstrap " << flush;
			for (uword iB = 0; iB < nBootstrap; iB++) {
				if(verbose) cout << iB + 1 << flush;
				
				// do the WB voxel per voxel (due to the voxelwise design)
				uvec v = linspace<uvec>(0, nVox - 1, 101);
				it4 = 0; // counter for indicating percentage
				for(uword iVox = 0; iVox < nVox; iVox++) {
					adjustDesign(iVox); // change the design matrix and its pseudoinverse
					residuals = computeAdjustedResiduals(data.col(iVox), pe.col(iVox));
					if (!doFOnly && (voxelwiseOutput || outputUncorr || clusterThresholdT > 0 || clusterMassThresholdT > 0 || doTfce)) {
						for (uword j = 0; j < tc.n_rows; j++) {
							restrictedMatrix = solve(designMatrix.t() * designMatrix, tc.row(j).t());
							restrictedMatrix = designMatrix - designMatrix * (restrictedMatrix * solve(tc.row(j) * restrictedMatrix, tc.row(j)));
							restrictedFittedData = restrictedMatrix * pe.col(iVox);
							restrictedResiduals = computeAdjustedRestrictedResiduals(data.col(iVox), restrictedFittedData, restrictedMatrix);
							resampledData = restrictedFittedData + restrictedResiduals % resamplingMult.col(iB);
							
							if (clusterThresholdT > 0 || clusterMassThresholdT > 0) {
								computeT(resampledData, tmpScoreB, copeB, varcopeB, tmpDofB, j);
								scoreB(j, iVox) = as_scalar(tmpScoreB);
								dofB(j, iVox) = as_scalar(tmpDofB);
								if (voxelwiseOutput || outputUncorr) scoreBVox(j, iVox) = scoreB(j, iVox);
							}
							else {
								computeT(resampledData, tmpScoreB, copeB, varcopeB, j);
								scoreBVox(j, iVox) = as_scalar(tmpScoreB);
							}
						}
					} // end !doFOnly
					// for F-contrasts
					if (!doTOnly && (voxelwiseOutput || outputUncorr || clusterThresholdF > 0 || clusterMassThresholdF > 0 || doTfce)) {
						for (uword j = 0; j < fc.n_rows; j++){
							restrictedMatrix = solve(designMatrix.t() * designMatrix, fullFContrast(j).t());
							restrictedMatrix = designMatrix - designMatrix * (restrictedMatrix * solve(fullFContrast(j) * restrictedMatrix, fullFContrast(j)));
							restrictedFittedData = restrictedMatrix * pe.col(iVox);
							restrictedResiduals = computeAdjustedRestrictedResiduals(data.col(iVox), restrictedFittedData, restrictedMatrix);
							resampledData = restrictedFittedData + restrictedResiduals % resamplingMult.col(iB);
							
							if (clusterThresholdF > 0 || clusterMassThresholdF > 0) {
								computeF(resampledData, tmpScoreB, copeB, varcopeB, tmpDofB, j);
								scoreB(j + startFC, iVox) = as_scalar(tmpScoreB); // if contrast rank > 1, will be corrected with degrees of freedom
								dofB(j + startFC, iVox) = as_scalar(tmpDofB);
								if (voxelwiseOutput || outputUncorr) {
									if (sizeFcontrast(j) == 1) {
										scoreBVox(j+ startFC, iVox) = scoreB(j + startFC, iVox);
									}
									else {
										if (scoreB(j + startFC, iVox) == 0) { // need to check if some score = 0
											computeF(resampledData, tmpScoreB, copeB, varcopeB, j);
											scoreBVox(j+ startFC, iVox) = as_scalar(tmpScoreB);
										}
										else {
											scoreBVox(j+ startFC, iVox) = scoreB(j+ startFC, iVox) * (dofB(j + startFC, iVox) / (dofB(j + startFC, iVox) - sizeFcontrast(j) + 1));
										}
									}
								}
							}
							else {
								computeF(resampledData, tmpScoreB, copeB, varcopeB, j);
								scoreBVox(j + startFC, iVox) = as_scalar(tmpScoreB);
							}
						}
					}
					if(any(v==iVox) && verbose){ // progression display
						if (it4 == 0) cout << "   (0%)" << flush;
						else if (it4 > 0 && it4 < 10) cout << "\b\b\b" << it4 << "%)" << flush;
						else if (it4 == 10) cout << "\b\b\b\b\b(" << it4 << "%)" << flush;
						else if (it4 >= 10 && it4 < 100) cout << "\b\b\b\b" << it4 << "%)" << flush;
						else cout << "\b\b\b\b\b\b(" << it4 << "%)" << flush;
						it4++;
					}
				} // end iVox loop

				// compute equivalent score + cluster-wise inference
				if (!doFOnly) {
					for (uword j = 0; j < tc.n_rows; j++){
						tmp = scoreB.row(j);
						T2ZP(tmp, eqScoreB, pValueB, dofB.row(j), false);
						if (clusterThresholdT > 0){
							if (clusterThresholdT < 1) // pValue threshold
								computeClusterStatsT(pValueB, clusterExtentLabelsB, clusterExtentSizesB);
							else // equivalent z-score threshold
								computeClusterStatsT(eqScoreB, clusterExtentLabelsB, clusterExtentSizesB);
						
							if (clusterExtentSizesB.n_elem > 0) maxClusterExtentSizes(j, iB + 1) = clusterExtentSizesB.max();
							if (clusterExtentSizes(j).n_elem > 0) oneMinusFwePClusterExtent(j) = oneMinusFwePClusterExtent(j) - (conv_to<vec>::from(clusterExtentSizes(j)) - TOLERANCE < maxClusterExtentSizes(j, iB + 1));
						}
						if (clusterMassThresholdT > 0){
							if (clusterMassThresholdT < 1) // pValue threshold
								computeClusterStatsT(pValueB, eqScoreB, clusterMassLabelsB, clusterMassSizesB);
							else // equivalent z-score threshold
								computeClusterStatsT(eqScoreB, clusterMassLabelsB, clusterMassSizesB);
							
							if (clusterMassSizesB.n_elem > 0) maxClusterMassSizes(j, iB + 1) = clusterMassSizesB.max();
							if (clusterMassSizes(j).n_elem > 0) oneMinusFwePClusterMass(j) = oneMinusFwePClusterMass(j) - (clusterMassSizes(j) - TOLERANCE < maxClusterMassSizes(j, iB + 1));
						}
						if (doTfce) {
							if ( tfceDelta(j) <= 0 ) {
								tfceScoreB.zeros();
								maxTfceScore(j, iB + 1) = 0;
							}
							else {
								tfce(eqScoreB, tfceScoreB, j);
								oneMinusUncPTfce.row(j) = oneMinusUncPTfce.row(j) - (tfceScore.row(j) - TOLERANCE < tfceScoreB);
								maxTfceScore(j, iB + 1) = tfceScoreB.max();
								oneMinusFwePTfce.row(j) = oneMinusFwePTfce.row(j) - (tfceScore.row(j) - TOLERANCE < maxTfceScore(j, iB + 1));
							}
						}
					}
				}
				if (!doTOnly) {
					for (uword j = 0; j < fc.n_rows; j++){
						// convert into equivalent scores and parametric p-values
						tmp = scoreB.row(j + startFC);
						if (doTfce)
							F2XZP(tmp, eqScoreB, eqZScoreB, pValueB, sizeFcontrast(j), dofB.row(j + startFC), false);
						else
							F2XP(tmp, eqScoreB, pValueB, sizeFcontrast(j), dofB.row(j + startFC), false);

						if (clusterThresholdF > 0){
							if (clusterThresholdF < 1) // pValue threshold
								computeClusterStatsF(pValueB, clusterExtentLabelsB, clusterExtentSizesB);
							else // equivalent chi^2-score threshold
								computeClusterStatsF(eqScoreB, clusterExtentLabelsB, clusterExtentSizesB);
							
							if (clusterExtentSizesB.n_elem > 0) maxClusterExtentSizes(j + startFC, iB + 1) = clusterExtentSizesB.max();
							if (clusterExtentSizes(j + startFC).n_elem > 0)	oneMinusFwePClusterExtent(j + startFC) = oneMinusFwePClusterExtent(j + startFC) - (conv_to<vec>::from(clusterExtentSizes(j + startFC)) - TOLERANCE < maxClusterExtentSizes(j + startFC, iB + 1));
						}
						if (clusterMassThresholdF > 0){
							if (clusterMassThresholdF < 1) // pValue threshold
								computeClusterStatsF(pValueB, eqScoreB, clusterMassLabelsB, clusterMassSizesB);
							else // equivalent chi^2-score threshold
								computeClusterStatsF(eqScoreB, clusterMassLabelsB, clusterMassSizesB);
							
							if (clusterMassSizesB.n_elem > 0) maxClusterMassSizes(j + startFC, iB + 1) = clusterMassSizesB.max();
							if (clusterMassSizes(j + startFC).n_elem > 0)	oneMinusFwePClusterMass(j + startFC) = oneMinusFwePClusterMass(j + startFC) - ( (clusterMassSizes(j + startFC) - TOLERANCE) < maxClusterMassSizes(j + startFC, iB + 1));
						}
						if (doTfce) {
							if ( tfceDelta(j + startFC) <= 0 ) {
								tfceScoreB.zeros();
								maxTfceScore(j + startFC, iB + 1) = 0;
							}
							else {
								tfce(eqZScoreB, tfceScoreB, j + startFC);
								oneMinusUncPTfce.row(j + startFC) = oneMinusUncPTfce.row(j + startFC) - (tfceScore.row(j + startFC) - TOLERANCE < tfceScoreB);
								maxTfceScore(j + startFC, iB + 1) = tfceScoreB.max();
								oneMinusFwePTfce.row(j + startFC) = oneMinusFwePTfce.row(j + startFC) - (tfceScore.row(j + startFC) - TOLERANCE < maxTfceScore(j + startFC, iB + 1));
							}
						}
					}
				}
				// voxelwise inference
				if (voxelwiseOutput || outputUncorr) {
					oneMinusUncP = oneMinusUncP - (eqScoreB > equivalentScore - TOLERANCE); // decrease 1-p if the bootsrap score is higher than the original
					maxScore.col(iB + 1) = max(eqScoreB, 1);
					for (uword j = 0; j < nContrasts; j++) oneMinusFweP.row(j) = oneMinusFweP.row(j) - (equivalentScore.row(j) - TOLERANCE < maxScore(j, iB + 1));
				}
				
				if(verbose){
					if (iB < 9) cout << "\b\b\b\b\b\b\b\b" << flush;
					else if (iB >= 9 && iB < 99) cout << "\b\b\b\b\b\b\b\b\b" << flush;
					else if (iB >= 99 && iB < 999) cout << "\b\b\b\b\b\b\b\b\b\b" << flush;
					else if (iB >= 999 && iB < 9999) cout << "\b\b\b\b\b\b\b\b\b\b\b" << flush;
					else cout << "\b\b\b\b\b\b\b\b\b\b\b\b" << flush;
				}
				
			} // end iB loop
			
			if(verbose) cout << "Saving Results: " << flush;
			oneMinusUncP = oneMinusUncP / (nBootstrap + 1);
			oneMinusFweP = oneMinusFweP / (nBootstrap + 1);
			
			if(doTfce) {
				oneMinusUncPTfce = oneMinusUncPTfce / (nBootstrap + 1);
				oneMinusFwePTfce = oneMinusFwePTfce / (nBootstrap + 1);
			}
			
			if(!doFOnly){
				for (uword j = 0; j < tc.n_rows; j++){
					string statLabel = "_" + LABEL_TSTAT + num2str(j + 1);
					if(logP){
						if (outputUncorr) {
							tmp = -log10(1-oneMinusUncP.row(j));
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P  + statLabel);
						}
						if (voxelwiseOutput) {
							tmp = -log10(1-oneMinusFweP.row(j));
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
						if (doTfce) {
							tmp = -log10(1-oneMinusUncPTfce.row(j));
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_P  + statLabel);
							tmp = -log10(1-oneMinusFwePTfce.row(j));
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
					}
					else {
						if (outputUncorr) {
							tmp = oneMinusUncP.row(j);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_P + statLabel);
						}
						if (voxelwiseOutput) {
							tmp = oneMinusFweP.row(j);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
						if (doTfce) {
							tmp = oneMinusUncPTfce.row(j);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_P  + statLabel);
							tmp = oneMinusFwePTfce.row(j);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
					}
					if (outputTextNull && voxelwiseOutput) {
						tmpVec = maxScore.row(j).t();
						printMaxStats(tmpVec, "_" + LABEL_VOX, statLabel);
					}
					if (clusterThresholdT > 0) {
						tmp = oneMinusFwePClusterExtent(j) / (nBootstrap + 1);
						saveClusterWisePValues(tmp, clusterExtentLabels(j), "_" + LABEL_CLUSTERE, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterExtentSizes.row(j).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERE, statLabel);
						}
					}
					if (clusterMassThresholdT > 0) {
						tmp = oneMinusFwePClusterMass(j) / (nBootstrap + 1);
						saveClusterWisePValues(tmp, clusterMassLabels(j), "_" + LABEL_CLUSTERM, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterMassSizes.row(j).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERM, statLabel);
						}
					}
					if (doTfce && outputTextNull) {
						tmpVec = maxTfceScore.row(j).t();
						printMaxStats(tmpVec, "_" + LABEL_TFCE, statLabel);
					}
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					string statLabel = "_" + LABEL_FSTAT + num2str(j + 1);
					if(logP){
						if (outputUncorr) {
							tmp = -log10(1-oneMinusUncP.row(j + startFC));
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P  + statLabel);
						}
						if (voxelwiseOutput) {
							tmp = -log10(1-oneMinusFweP.row(j + startFC));
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
						if (doTfce) {
							tmp = -log10(1-oneMinusUncPTfce.row(j + startFC));
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_P  + statLabel);
							tmp = -log10(1-oneMinusFwePTfce.row(j + startFC));
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
					}
					else {
						if (outputUncorr) {
							tmp = oneMinusUncP.row(j + startFC);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_P + statLabel);
						}
						if (voxelwiseOutput) {
							tmp = oneMinusFweP.row(j + startFC);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
						if (doTfce) {
							tmp = oneMinusUncPTfce.row(j + startFC);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_P  + statLabel);
							tmp = oneMinusFwePTfce.row(j + startFC);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
					}
					if (outputTextNull && voxelwiseOutput) {
						tmpVec = maxScore.row(j + startFC).t();
						printMaxStats(tmpVec, "_" + LABEL_VOX, statLabel);
					}
					if (clusterThresholdF > 0) {
						tmp = oneMinusFwePClusterExtent(j + startFC) / (nBootstrap + 1);
						saveClusterWisePValues(tmp, clusterExtentLabels(j + startFC), "_" + LABEL_CLUSTERE, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterExtentSizes.row(j + startFC).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERE, statLabel);
						}
					}
					if (clusterMassThresholdF > 0) {
						tmp = oneMinusFwePClusterMass(j + startFC) / (nBootstrap + 1);
						saveClusterWisePValues(tmp, clusterMassLabels(j + startFC), "_" + LABEL_CLUSTERM, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterMassSizes.row(j + startFC).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERM, statLabel);
						}
					}
					if (doTfce && outputTextNull) {
						tmpVec = maxTfceScore.row(j + startFC).t();
						printMaxStats(tmpVec, "_" + LABEL_TFCE, statLabel);
					}
				}
			}
			if (verbose) cout << "done" << endl;
		}
		else { // same design at every voxel -> do everything in one go
			mat cope;
			if (verbose) cout << "Computing original equivalent scores: " << flush;
			
			pe = computePe(data);
			residuals = computeAdjustedResiduals(data, pe);
			scoreB.set_size(1, residuals.n_cols);
			copeB.set_size(1, residuals.n_cols);
			varcopeB.set_size(1, residuals.n_cols);
			zScoreB.set_size(1, residuals.n_cols);
			xScoreB.set_size(1, residuals.n_cols);
			pValueB.set_size(1, residuals.n_cols);
			dofB.set_size(1, residuals.n_cols);
			computeSweAll(swe, dof, residuals); // need to get the dof

			residuals.reset(); // to release memory (TBC)
			if (outputGlm) saveImage(pe, out_fileroot  + "_" + LABEL_PE);

			if (doTfce) {
				oneMinusUncPTfce.set_size(1, nVox);
				oneMinusFwePTfce.set_size(1, nVox);
			}
			// compute original scores
			if (!doFOnly) {
				for (uword j = 0; j < tc.n_rows; j++) {
					cope = tc.row(j) * pe;
					scoreB = cope /sqrt(swe(j));
					if (outputGlm) saveImage(cope, out_fileroot + "_" + LABEL_COPE + "_" + LABEL_TSTAT + num2str(j + 1));
					if (outputRaw) saveImage(scoreB, out_fileroot + "_" + LABEL_TSTAT + num2str(j + 1));
					if (outputGlm) saveImage(swe(j), out_fileroot + "_" + LABEL_VARCOPE + "_" + LABEL_TSTAT + num2str(j + 1));
					// compute the equivalent scores and p-values
					T2ZP(scoreB, zScoreB, pValueB, dof.row(j), false);
					// get the max equivalent score
					maxScore(j, 0) = zScoreB.max();
					// save the equivalent score
					equivalentScore.row(j) = zScoreB;
					if (outputEquivalent) saveImage(zScoreB, out_fileroot + "_" + LABEL_EQUIVALENT_Z + "_" + LABEL_TSTAT + num2str(j + 1));
					if (clusterThresholdT > 0){
						if (clusterThresholdT < 1) // pValue threshold
							computeClusterStatsT(pValueB, clusterExtentLabels(j), clusterExtentSizes(j));
						else // equivalent z-score threshold
							computeClusterStatsT(zScoreB, clusterExtentLabels(j), clusterExtentSizes(j));
						
						tmp.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterExtentLabels(j)(i) > 0) tmp(i) = clusterExtentSizes(j)(clusterExtentLabels(j)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERE + "_" + LABEL_TSTAT + num2str(j + 1));
						if (clusterExtentSizes(j).n_elem > 0 ) maxClusterExtentSizes(j, 0) = clusterExtentSizes(j).max();
					}
					if (clusterMassThresholdT > 0){
						if (clusterMassThresholdT < 1) // pValue threshold
							computeClusterStatsT(pValueB, zScoreB, clusterMassLabels(j), clusterMassSizes(j));
						else // equivalent z-score threshold
							computeClusterStatsT(zScoreB, clusterMassLabels(j), clusterMassSizes(j));

						tmp.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterMassLabels(j)(i) > 0) tmp(i) = clusterMassSizes(j)(clusterMassLabels(j)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERM + "_" + LABEL_TSTAT + num2str(j + 1));
						if (clusterMassSizes(j).n_elem > 0 ) maxClusterMassSizes(j, 0) = clusterMassSizes(j).max();
					}
					if (doTfce) {
						if (!tfceDeltaOverride)
							tfceDelta(j) = zScoreB.max() / 100.0;  // i.e. 100 subdivisions of the max input stat height
						if ( tfceDelta(j) <= 0 ) {
							cout << "Warning: the original statistic image for contrast t" << j << " contains no positive values, and cannot be processed with TFCE. A blank output image will be created." << endl;
							tfceScoreB.zeros();
							maxTfceScore(j, 0) = 0;
							oneMinusUncPTfce.zeros();
							oneMinusFwePTfce.zeros();
						}
						else {
							tfce(zScoreB, tfceScoreB, j);
							maxTfceScore(j, 0) = tfceScoreB.max();
						}
						// save original tfce scores
						tfceScore.row(j) = tfceScoreB;
						saveImage(tfceScoreB, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_TSTAT + num2str(j + 1));
					}
				}
			}
			if (!doTOnly) {
				for (uword j = 0; j < fc.n_rows; j++){
					cope = fullFContrast(j) * pe;
					if (sizeFcontrast(j) == 1) {
						scoreB = square(cope) / swe(j + startFC);
					}
					else {
						tmp.set_size(sizeFcontrast(j), sizeFcontrast(j));
						uvec ind = find(trimatl(ones<umat>(sizeFcontrast(j), sizeFcontrast(j)))== 1); // indices for the lower triangular
						for (uword iVox = 0; iVox < nVox; iVox++) { // need to do it voxel per voxel
							tmp(ind) = swe(j + startFC).col(iVox);
							scoreB(iVox) = as_scalar(cope.col(iVox).t() * solve(symmatl(tmp), cope.col(iVox))) / sizeFcontrast(j);
						}
					}
					if (outputGlm) saveImage(cope, out_fileroot + "_" + LABEL_COPE + "_" + LABEL_FSTAT + num2str(j + 1));
					if (outputRaw) saveImage(scoreB, out_fileroot + "_" + LABEL_FSTAT + num2str(j + 1));
					if (outputGlm) saveImage(swe(j + startFC), out_fileroot + "_" + LABEL_VARCOPE + "_" + LABEL_FSTAT + num2str(j + 1));
					
					if (sizeFcontrast(j) > 1) {
						scoreB = scoreB % ((dof.row(j + startFC) - sizeFcontrast(j) + 1) / dof.row(j + startFC)); // need to correct with dof
						uvec tmpInd = find(scoreB < 0);
						scoreB(tmpInd) = zeros<mat>(1, tmpInd.n_elem);
					}
					// compute the equivalent scores and p-values
					if (doTfce)
						F2XZP(scoreB, xScoreB, eqZScoreB, pValueB, sizeFcontrast(j), dof.row(j + startFC), false);
					else
						F2XP(scoreB, xScoreB, pValueB, sizeFcontrast(j), dof.row(j + startFC), false);
					// get the max equivalent score
					maxScore(j + startFC, 0) = xScoreB.max();
					// save the equivalent scores
					equivalentScore.row(j + startFC) = xScoreB;
					if (outputEquivalent) saveImage(xScoreB, out_fileroot + "_" + LABEL_EQUIVALENT_CHISQUARE + "_" + LABEL_FSTAT + num2str(j + 1));
					
					if (clusterThresholdF > 0){
						if (clusterThresholdF < 1) // pValue threshold
							computeClusterStatsF(pValueB, clusterExtentLabels(j + startFC), clusterExtentSizes(j + startFC));
						else // equivalent chi^2-score threshold
							computeClusterStatsF(xScoreB, clusterExtentLabels(j + startFC), clusterExtentSizes(j + startFC));
						
						tmp.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterExtentLabels(j + startFC)(i) > 0) tmp(i) = clusterExtentSizes(j + startFC)(clusterExtentLabels(j + startFC)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERE + "_" + LABEL_FSTAT + num2str(j + 1));
						if (clusterExtentSizes(j + startFC).n_elem > 0 ) maxClusterExtentSizes(j + startFC, 0) = clusterExtentSizes(j + startFC).max();
					}
					if (clusterMassThresholdF > 0){
						if (clusterMassThresholdF < 1) // pValue threshold
							computeClusterStatsF(pValueB, xScoreB, clusterMassLabels(j + startFC), clusterMassSizes(j + startFC));
						else // equivalent chi^2-score threshold
							computeClusterStatsF(xScoreB, clusterMassLabels(j + startFC), clusterMassSizes(j + startFC));

						tmp.zeros(1, nVox);
						for (uword i = 0; i < nVox; i++) {
							if (clusterMassLabels(j + startFC)(i) > 0) tmp(i) = clusterMassSizes(j + startFC)(clusterMassLabels(j + startFC)(i) - 1);
						}
						saveImage(tmp, out_fileroot + "_" + LABEL_CLUSTERM + "_" + LABEL_FSTAT + num2str(j + 1));
						if (clusterMassSizes(j + startFC).n_elem > 0 ) maxClusterMassSizes(j + startFC, 0) = clusterMassSizes(j + startFC).max();
					}
					if (doTfce) {
						if (!tfceDeltaOverride)
							tfceDelta(j + startFC) = eqZScoreB.max() / 100.0;  // i.e. 100 subdivisions of the max input stat height
						if ( tfceDelta(j + startFC) <= 0 ) {
							cout << "Warning: the original statistic image for contrast t" << j << " contains no positive values, and cannot be processed with TFCE. A blank output image will be created." << endl;
							tfceScoreB.zeros();
							maxTfceScore(j + startFC, 0) = 0;
							oneMinusUncPTfce.zeros();
							oneMinusFwePTfce.zeros();
						}
						else {
							tfce(eqZScoreB, tfceScoreB, j + startFC);
							maxTfceScore(j + startFC, 0) = tfceScoreB.max();
						}
						// save original tfce scores
						tfceScore.row(j + startFC) = tfceScoreB;
						saveImage(tfceScoreB, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_FSTAT + num2str(j + 1));
					}
				}
			}
			if (verbose) cout << "done" << endl;
			
			// Start the Wild Bootstrap procedure contrast per contrast
			mat oneMinusUncP(1, nVox);
			mat oneMinusFweP(1, nVox);
			mat oneMinusFwePClusterExtent, oneMinusFwePClusterMass;
			
			if (!doFOnly && (voxelwiseOutput || outputUncorr || clusterThresholdT > 0 || clusterMassThresholdT > 0 || doTfce)) {
				for (uword j = 0; j < tc.n_rows; j++) {
					if(verbose) cout << "Computing non-parametric p-values for t-constrat " << j + 1 << ": " << flush;
					oneMinusUncP.fill(nBootstrap);
					oneMinusFweP.fill(nBootstrap);
					restrictedMatrix = solve(designMatrix.t() * designMatrix, tc.row(j).t());
					restrictedMatrix = designMatrix - designMatrix * (restrictedMatrix * solve(tc.row(j) * restrictedMatrix, tc.row(j)));
					restrictedFittedData = restrictedMatrix * pe;
					restrictedResiduals = computeAdjustedRestrictedResiduals(data, restrictedFittedData, restrictedMatrix);
					if (clusterThresholdT > 0) {
						oneMinusFwePClusterExtent.set_size(clusterExtentSizes(j).n_elem, 1);
						oneMinusFwePClusterExtent.fill(nBootstrap);
					}
					if (clusterMassThresholdT > 0) {
						oneMinusFwePClusterMass.set_size(clusterMassSizes(j).n_elem, 1);
						oneMinusFwePClusterMass.fill(nBootstrap);
					}
					if (doTfce) {
						if ( tfceDelta(j) > 0) {
							oneMinusUncPTfce.fill(nBootstrap);
							oneMinusFwePTfce.fill(nBootstrap);
						}
					}
					if(verbose) cout << "bootstrap " << flush;
					for (uword iB = 0; iB < nBootstrap; iB++) {
						if(verbose) cout << iB + 1 << flush;
						resampledData = restrictedFittedData + restrictedResiduals % arma::repmat(resamplingMult.col(iB), 1, restrictedResiduals.n_cols);
						
						computeT(resampledData, scoreB, copeB, varcopeB, dofB, j);
						// convert into equivalent z-scores and parametric p-values
						T2ZP(scoreB, zScoreB, pValueB, dofB, false);
						
						oneMinusUncP = oneMinusUncP - (zScoreB > equivalentScore.row(j) - TOLERANCE); // decrease 1-p if the bootsrap score is higher than the original
						maxScore(j, iB + 1) = zScoreB.max();
						oneMinusFweP = oneMinusFweP - (equivalentScore.row(j) - TOLERANCE < maxScore(j, iB + 1));
						
						// cluster-wise inference
						if (clusterThresholdT > 0){
							if (clusterThresholdT < 1) // pValue threshold
								computeClusterStatsT(pValueB, clusterExtentLabelsB, clusterExtentSizesB);
							else // equivalent z-score threshold
								computeClusterStatsT(zScoreB, clusterExtentLabelsB, clusterExtentSizesB);
							
							if (clusterExtentSizesB.n_elem > 0) maxClusterExtentSizes(j, iB + 1) = clusterExtentSizesB.max();
							oneMinusFwePClusterExtent = oneMinusFwePClusterExtent - (conv_to<vec>::from(clusterExtentSizes(j)) - TOLERANCE < maxClusterExtentSizes(j, iB + 1));
						}
						if (clusterMassThresholdT > 0){
							if (clusterMassThresholdT < 1) // pValue threshold
								computeClusterStatsT(pValueB, zScoreB, clusterMassLabelsB, clusterMassSizesB);
							else // equivalent z-score threshold
								computeClusterStatsT(zScoreB, clusterMassLabelsB, clusterMassSizesB);
							if (clusterMassSizesB.n_elem > 0) maxClusterMassSizes(j, iB + 1) = clusterMassSizesB.max();
							oneMinusFwePClusterMass = oneMinusFwePClusterMass - (clusterMassSizes(j) - TOLERANCE < maxClusterMassSizes(j, iB + 1));
						}
						if (doTfce) {
							if ( tfceDelta(j) <= 0 ) {
								tfceScoreB.zeros();
								maxTfceScore(j, iB + 1) = 0;
							}
							else {
								tfce(zScoreB, tfceScoreB, j);
								oneMinusUncPTfce = oneMinusUncPTfce - (tfceScore.row(j) - TOLERANCE < tfceScoreB);
								maxTfceScore(j, iB + 1) = tfceScoreB.max();
								oneMinusFwePTfce = oneMinusFwePTfce - (tfceScore.row(j) - TOLERANCE < maxTfceScore(j, iB + 1));
							}
						}
						if(verbose){
							if (iB < 9) cout << "\b" << flush;
							else if (iB >= 9 && iB < 99) cout << "\b\b" << flush;
							else if (iB >= 99 && iB < 999) cout << "\b\b\b" << flush;
							else if (iB >= 999 && iB < 9999) cout << "\b\b\b\b" << flush;
							else cout << "\b\b\b\b\b" << flush;
						}
					} // end iB loop
					
					oneMinusUncP = oneMinusUncP / (nBootstrap + 1);
					oneMinusFweP = oneMinusFweP / (nBootstrap + 1);
					if(logP){
						if (outputUncorr) {
							tmp = -log10(1-oneMinusUncP);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P  + "_" + LABEL_TSTAT + num2str(j + 1));
						}
						if (voxelwiseOutput) {
							tmp = -log10(1-oneMinusFweP);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
						}
					}
					else {
						if (outputUncorr) {
							tmp = oneMinusUncP;
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
						}
						if (voxelwiseOutput) {
							tmp = oneMinusFweP;
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_FWE_CORRECTED + LABEL_P + "_" + LABEL_TSTAT + num2str(j + 1));
						}
					}
					string statLabel = "_" + LABEL_TSTAT + num2str(j + 1);
					if (outputTextNull && voxelwiseOutput) {
						tmpVec = maxScore.row(j).t();
						printMaxStats(tmpVec, "_" + LABEL_VOX, statLabel);
					}
					if (clusterThresholdT > 0) {
						oneMinusFwePClusterExtent = oneMinusFwePClusterExtent / (nBootstrap + 1);
						saveClusterWisePValues(oneMinusFwePClusterExtent, clusterExtentLabels(j), "_" + LABEL_CLUSTERE, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterExtentSizes.row(j).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERE, statLabel);
						}
					}
					if (clusterMassThresholdT > 0) {
						oneMinusFwePClusterMass = oneMinusFwePClusterMass / (nBootstrap + 1);
						saveClusterWisePValues(oneMinusFwePClusterMass, clusterMassLabels(j), "_" + LABEL_CLUSTERM, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterMassSizes.row(j).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERM, statLabel);
						}
					}
					if (doTfce) {
						oneMinusUncPTfce = oneMinusUncPTfce / (nBootstrap + 1);
						oneMinusFwePTfce = oneMinusFwePTfce / (nBootstrap + 1);
						if(logP){
							tmp = -log10(1-oneMinusUncPTfce);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_P  + statLabel);
							tmp = -log10(1-oneMinusFwePTfce);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						} 
						else {
							saveImage(oneMinusUncPTfce, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_P  + statLabel);
							saveImage(oneMinusFwePTfce, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_FWE_CORRECTED + LABEL_P + statLabel);
						}
						if (outputTextNull) {
							tmpVec = maxTfceScore.row(j).t();
							printMaxStats(tmpVec, "_" + LABEL_TFCE, statLabel);
						}
					}
					if(verbose) cout << endl;
				}
			} // end of !doFOnly
			// for F-contrasts
			if (!doTOnly && (voxelwiseOutput || outputUncorr || clusterThresholdF > 0 || clusterMassThresholdF > 0 || doTfce)) {
				scoreBVox.set_size(1, nVox);
				for (uword j = 0; j < fc.n_rows; j++){
					if(verbose) cout << "Computing non-parametric p-values for f-constrat " << j + 1 << ": " << flush;
					oneMinusUncP.fill(nBootstrap);
					oneMinusFweP.fill(nBootstrap);
					restrictedMatrix = solve(designMatrix.t() * designMatrix, fullFContrast(j).t());
					restrictedMatrix = designMatrix - designMatrix * (restrictedMatrix * solve(fullFContrast(j) * restrictedMatrix, fullFContrast(j)));
					restrictedFittedData = restrictedMatrix * pe;
					restrictedResiduals = computeAdjustedRestrictedResiduals(data, restrictedFittedData, restrictedMatrix);
					if (clusterThresholdF > 0) {
						oneMinusFwePClusterExtent.set_size(clusterExtentSizes(j + startFC).n_elem);
						oneMinusFwePClusterExtent.fill(nBootstrap);
					}
					if (clusterMassThresholdF > 0) {
						oneMinusFwePClusterMass.set_size(clusterMassSizes(j + startFC).n_elem);
						oneMinusFwePClusterMass.fill(nBootstrap);
					}
					if (doTfce) {
						if ( tfceDelta(j + startFC) > 0) {
							oneMinusUncPTfce.fill(nBootstrap);
							oneMinusFwePTfce.fill(nBootstrap);
						}
					}
					if(verbose) cout << "bootstrap " << flush;
					for (uword iB = 0; iB < nBootstrap; iB++) {
						if(verbose) cout << iB + 1 << flush;
						resampledData = restrictedFittedData + restrictedResiduals % arma::repmat(resamplingMult.col(iB), 1, restrictedResiduals.n_cols);
						
						computeF(resampledData, scoreB, copeB, varcopeB, dofB, j);
						// convert into equivalent chi^2-scores (and z-scores if tfce) and parametric p-values
						if (doTfce)
							F2XZP(scoreB, xScoreB, eqZScoreB, pValueB, sizeFcontrast(j), dofB, false);
						else
							F2XP(scoreB, xScoreB, pValueB, sizeFcontrast(j), dofB, false);
						
						oneMinusUncP = oneMinusUncP - (xScoreB > equivalentScore.row(j + startFC) - TOLERANCE); // decrease 1-p if the bootsrap score is higher than the original
						maxScore(j + startFC, iB + 1) = xScoreB.max();
						oneMinusFweP = oneMinusFweP - (equivalentScore.row(j + startFC) - TOLERANCE < maxScore(j + startFC, iB + 1));
						// cluster-wise inference
						if (clusterThresholdF > 0){
							if (clusterThresholdF < 1) // pValue threshold
								computeClusterStatsF(pValueB, clusterExtentLabelsB, clusterExtentSizesB);
							else // equivalent chi^2-score threshold
								computeClusterStatsF(xScoreB, clusterExtentLabelsB, clusterExtentSizesB);
							
							if (clusterExtentSizesB.n_elem > 0) maxClusterExtentSizes(j + startFC, iB + 1) = clusterExtentSizesB.max();
							oneMinusFwePClusterExtent = oneMinusFwePClusterExtent - (conv_to<vec>::from(clusterExtentSizes(j + startFC)) - TOLERANCE < maxClusterExtentSizes(j + startFC, iB + 1));
						}
						if (clusterMassThresholdF > 0){
							if (clusterMassThresholdF < 1) // pValue threshold
								computeClusterStatsF(pValueB, xScoreB, clusterMassLabelsB, clusterMassSizesB);
							else // equivalent chi^2-score threshold
								computeClusterStatsF(xScoreB, clusterMassLabelsB, clusterMassSizesB);

							if (clusterMassSizesB.n_elem > 0) maxClusterMassSizes(j + startFC, iB + 1) = clusterMassSizesB.max();
							oneMinusFwePClusterMass = oneMinusFwePClusterMass - (clusterMassSizes(j + startFC) - TOLERANCE < maxClusterMassSizes(j + startFC, iB + 1));
						}
						if (doTfce) {
							if ( tfceDelta(j + startFC) <= 0 ) {
								tfceScoreB.zeros();
								maxTfceScore(j + startFC, iB + 1) = 0;
							}
							else {
								tfce(eqZScoreB, tfceScoreB, j + startFC);
								oneMinusUncPTfce = oneMinusUncPTfce - (tfceScore.row(j + startFC) - TOLERANCE < tfceScoreB);
								maxTfceScore(j + startFC, iB + 1) = tfceScoreB.max();
								oneMinusFwePTfce = oneMinusFwePTfce - (tfceScore.row(j + startFC) - TOLERANCE < maxTfceScore(j + startFC, iB + 1));
							}
						}						
						if(verbose){
							if (iB < 9) cout << "\b" << flush;
							else if (iB >= 9 && iB < 99) cout << "\b\b" << flush;
							else if (iB >= 99 && iB < 999) cout << "\b\b\b" << flush;
							else if (iB >= 999 && iB < 9999) cout << "\b\b\b\b" << flush;
							else cout << "\b\b\b\b\b" << flush;
						}
					} // end of the WB loop
					
					oneMinusUncP = oneMinusUncP / (nBootstrap + 1);
					oneMinusFweP = oneMinusFweP / (nBootstrap + 1);
					if(logP) {
						if (outputUncorr) {
							tmp = -log10(1-oneMinusUncP);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_P  + "_" + LABEL_FSTAT + num2str(j + 1));
						}
						if (voxelwiseOutput) {
							tmp = -log10(1-oneMinusFweP);
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
						}
					}
					else {
						if (outputUncorr) {
							tmp = oneMinusUncP;
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
						}
						if (voxelwiseOutput) {
							tmp = oneMinusFweP;
							saveImage(tmp, out_fileroot + "_" + LABEL_VOX + "_" + LABEL_FWE_CORRECTED + LABEL_P + "_" + LABEL_FSTAT + num2str(j + 1));
						}
					}
					string statLabel = "_" + LABEL_FSTAT + num2str(j + 1);
					if (outputTextNull && voxelwiseOutput) {
						tmpVec = maxScore.row(j +  startFC).t();
						printMaxStats(tmpVec, "_" + LABEL_VOX, statLabel);
					}
					if (clusterThresholdF > 0) {
						oneMinusFwePClusterExtent = oneMinusFwePClusterExtent / (nBootstrap + 1);
						saveClusterWisePValues(oneMinusFwePClusterExtent, clusterExtentLabels(j + startFC), "_" + LABEL_CLUSTERE, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterExtentSizes.row(j +  startFC).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERE, statLabel);
						}
					}
					if (clusterMassThresholdF > 0) {
						oneMinusFwePClusterMass = oneMinusFwePClusterMass / (nBootstrap + 1);
						saveClusterWisePValues(oneMinusFwePClusterMass, clusterMassLabels(j + startFC), "_" + LABEL_CLUSTERM, statLabel);
						if (outputTextNull) {
							tmpVec = maxClusterMassSizes.row(j +  startFC).t();
							printMaxStats(tmpVec, "_" + LABEL_CLUSTERM, statLabel);
						}
					}
					if (doTfce) {
						oneMinusUncPTfce = oneMinusUncPTfce / (nBootstrap + 1);
						oneMinusFwePTfce = oneMinusFwePTfce / (nBootstrap + 1);
						if(logP){
							tmp = -log10(1-oneMinusUncPTfce);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_P  + statLabel);
							tmp = -log10(1-oneMinusFwePTfce);
							saveImage(tmp, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_LOG + LABEL_FWE_CORRECTED + LABEL_P  + statLabel);
						} 
						else {
							saveImage(oneMinusUncPTfce, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_P  + statLabel);
							saveImage(oneMinusFwePTfce, out_fileroot + "_" + LABEL_TFCE + "_" + LABEL_FWE_CORRECTED + LABEL_P  + statLabel);
						}
						if (outputTextNull) {
							tmpVec = maxTfceScore.row(j +  startFC).t();
							printMaxStats(tmpVec, "_" + LABEL_TFCE, statLabel);
						}
					}
					if(verbose) cout << endl;
				} // end of the F-contrasts loop
			}
		}
	}
}


