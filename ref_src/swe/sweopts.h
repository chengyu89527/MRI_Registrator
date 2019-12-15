/*  sweopts.h
    Bryan Guillaume & Tom Nichols
    Copyright (C) 2019 University of Oxford  */

#if !defined(sweopts_h)
#define sweopts_h

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "utils/options.h"
#include "utils/log.h"

using namespace Utilities;

namespace SWE {
	
	class sweopts {
	 public:
		static sweopts& getInstance();
		~sweopts() { delete gopt; }
		
		Option<string> in_fileroot;
		Option<string> maskname;
		Option<string> out_fileroot;
		Option<string> dm_file;
		Option<string> tc_file;
		Option<string> fc_file;
		Option<string> subj_file;
		Option<string> resamplingMatrix_file;
		Option<bool> modifiedSwE;
		Option<bool> wb;
		Option<bool> logP;
		Option<int> n_boot;
		Option<bool> voxelwiseOutput;
		Option<bool> doFOnly;
		Option<bool> tfce;
		Option<bool> tfce2D;
		Option<float> cluster_thresh;
		Option<float> clustermass_thresh;
		Option<float> f_thresh;
		Option<float> fmass_thresh;
		Option<float> tfce_height;
		Option<float> tfce_delta;
		Option<float> tfce_size;
		Option<int> tfce_connectivity;
		Option<bool> help;
		Option<bool> verbose;
		Option<bool> outputRaw;
		Option<bool> outputEquivalent;
		Option<bool> outputDof;
		Option<bool> outputUncorr;
//		Option<bool> outputTextWB;
		Option<bool> outputTextNull;
		Option<bool> disableNonConstantMask;
		Option<int> randomSeed;
		Option<vector<int> > voxelwise_ev_numbers;
		Option<vector<string> > voxelwise_ev_filenames;
		Option<bool> outputGlm;
		
		void parse_command_line(int argc, char** argv, Log& logger);
		
	 private:
		sweopts();
		const sweopts& operator=(sweopts&);
		sweopts(sweopts&);
			
		OptionParser options;
			
		static sweopts* gopt;
	};
	
 inline sweopts& sweopts::getInstance(){
	 if(gopt == NULL)
		 gopt = new sweopts();
	 
	 return *gopt;
 }
	
 inline sweopts::sweopts() :
	in_fileroot(string("-i"), "",
							string("~<input>\t4D input image"),
							true, requires_argument),
	maskname(string("-m,--mask"), "",
					 string("~<mask>\tmask image"),
					 false, requires_argument),
	out_fileroot(string("-o"), string(""),
							 string("~<out_root>\toutput file-rootname"),
							 true, requires_argument),
	dm_file(string("-d"), string(""),
					string("~<design.mat>\tdesign matrix file"),
					true, requires_argument),
	tc_file(string("-t"), string(""),
					string("~<design.con>\tt contrasts file"),
					true, requires_argument),
	fc_file(string("-f"), string(""),
					string("~<design.fts>\tf contrasts file"),
					false, requires_argument),
	subj_file(string("-s,--subj"), string(""),
						string("~<design.sub>\tsubjects file"),
						true, requires_argument),
	resamplingMatrix_file(string("--resamplingMatrix"), string(""),
												string("~<resamplingMatrix>\tresampling matrix file"),
												false, requires_argument, false),
	modifiedSwE(string("--modified"), false,
							string("use the modified \"Homogeneous\" SwE instead of the classic \"Heterogeneous\" SwE"),
							false, no_argument),
	wb(string("--wb"), false,
							string("\tinference is done using a non-parametric Wild Bootstrap procedure instead of a parametric procedure"),
							false, no_argument),
	logP(string("--logp"), false,
		 string("\treturn -log_10(p) images instead of 1-p images"),
		 false, no_argument),
	n_boot(string("-n"), 999,
				string("~<n_boot>\tnumber of bootstraps (default 999)"),
				false, requires_argument),
	voxelwiseOutput(string("-x,--corrp"),false,
									string("output voxelwise corrected p-value images"),
									false, no_argument),
	doFOnly(string("--fonly"), false,
					string("\tcalculate f-statistics only"),
					false, no_argument),
	tfce(string("-T"), false,
			string("\tcarry out Threshold-Free Cluster Enhancement"),
			false, no_argument),
	tfce2D(string("--T2"), false,
				string("\tcarry out Threshold-Free Cluster Enhancement with 2D optimisation (e.g. for TBSS data); H=2, E=1, C=26"),
				false, no_argument),
	cluster_thresh(string("-c"), -1,
								 string("~<thresh>\tcarry out cluster-extent-based inference for t-contrasts with the supplied cluster-forming threshold (supplied as an equivalent z-score if thresh >= 1 or as an uncorrected p-value if thresh < 1)"),
								 false, requires_argument),
	clustermass_thresh(string("-C"), -1,
								 string("~<thresh>\tcarry out cluster-mass-based inference for t-contrasts with the supplied cluster-forming threshold (supplied as an equivalent z-score if thresh >= 1 or as an uncorrected p-value if thresh < 1)"),
									false, requires_argument),
	f_thresh(string("-F"), -1,
					 string("~<thresh>\tcarry out cluster-extent-based inference for f-contrasts with the supplied cluster-forming threshold (supplied as an equivalent one-degree-of-freedom chi-squared-score if thresh >= 1 or as an uncorrected p-value if thresh < 1)"),
					 false, requires_argument),
	fmass_thresh(string("-S"), -1,
							 string("~<thresh>\tcarry out cluster-mass-based inference for f-contrasts with the supplied cluster-forming threshold (supplied as an equivalent one-degree-of-freedom chi-squared-score if thresh >= 1 or as an uncorrected p-value if thresh < 1)"),
							 false, requires_argument),
	tfce_height(string("--tfce_H"), 2, string("~<H>\tTFCE height parameter (default=2)"), false, requires_argument),
	tfce_delta(string("--tfce_D"), 1, string("~<H>\tTFCE delta parameter overide"), false, requires_argument),
	tfce_size(string("--tfce_E"), 0.5, string("~<E>\tTFCE extent parameter (default=0.5)"), false, requires_argument),
	tfce_connectivity(string("--tfce_C"), 6, string("~<C>\tTFCE connectivity (6 or 26; default=6)"), false, requires_argument),
	help(string("-h,--help"), false,
			 string("display this message"),
			 false, no_argument),
	verbose(string("--quiet"), true,
					string("\tswitch off diagnostic messages"),
					false, no_argument),
	outputRaw(string("-R,--raw"), false,
						string("output raw voxelwise statistic images"),
						false, no_argument),
	outputEquivalent(string("-E,--equivalent"), false,
						string("output equivalent z (for t-contrast) or one-degree-of-freedom chi-squared (for f-contrast) statistic images"),
						false, no_argument),
	outputDof(string("-D,--dof"), false,
									 string("output effective number of degrees of freedom images"),
									 false, no_argument),
	outputUncorr(string("--uncorrp"), false,
							 string("output uncorrected p-value images"),
							 false, no_argument),
//	outputTextWB(string("-P"), false,
//								 string("\toutput Wild Boostrap vector text file"),
//								 false, no_argument),
	outputTextNull(string("-N"), false,
								 string("\toutput null distribution text files"),
								 false, no_argument),
	disableNonConstantMask(string("--norcmask"), false,
												 string("don't remove constant voxels from mask"),
												 false, no_argument),
	randomSeed(string("--seed"),0,
						 string("~<seed>\tspecific integer seed for random number generator"),
						 false, requires_argument),
	voxelwise_ev_numbers(string("--vxl"), vector<int>(),
											 string("\tlist of numbers indicating voxelwise EVs position in the design matrix (list order corresponds to files in vxf option)."),
											 false, requires_argument, false),
	voxelwise_ev_filenames(string("--vxf"), vector<string>(),
												 string("\tlist of 4D images containing voxelwise EVs (list order corresponds to numbers in vxl option)."),
												 false, requires_argument, false),
	outputGlm(string("--glm_output"), false,
						string("output glm information (pe, cope & varcope)"),
						false, no_argument),
	options("swe v1.0.2", "swe -i <input> -o <output> -d <design.mat> -t <design.con> -s <design.sub> [options]")
	{
		
		try {
//			options.add(demean_data);
			options.add(in_fileroot);
			options.add(maskname);
			options.add(out_fileroot);
			options.add(dm_file);
			options.add(tc_file);
			options.add(fc_file);
			options.add(subj_file);
			options.add(resamplingMatrix_file);
			options.add(modifiedSwE);
			options.add(wb);
			options.add(logP);
			options.add(n_boot);
			options.add(voxelwiseOutput);
			options.add(doFOnly);
			options.add(tfce);
			options.add(tfce2D);
			options.add(cluster_thresh);
			options.add(clustermass_thresh);
			options.add(f_thresh);
			options.add(fmass_thresh);
			options.add(help);
			options.add(verbose);
			options.add(outputRaw);
			options.add(outputEquivalent);
			options.add(outputDof);
			options.add(outputUncorr);
//			options.add(outputTextWB);
			options.add(outputTextNull);
			options.add(disableNonConstantMask);
			options.add(randomSeed);
			options.add(tfce_height);
			options.add(tfce_delta);
			options.add(tfce_size);
			options.add(tfce_connectivity);
			options.add(voxelwise_ev_numbers);
			options.add(voxelwise_ev_filenames);
			options.add(outputGlm);
		}
		catch(X_OptionError& e) {
			options.usage();
			cerr << endl << e.what() << endl;
		}
		catch(std::exception &e) {
			cerr << e.what() << endl;
		}
	}
}

#endif

