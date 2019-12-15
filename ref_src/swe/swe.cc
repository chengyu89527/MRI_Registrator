/*  swe.cc
    Bryan Guillaume & Tom Nichols
    Copyright (C) 2019 University of Oxford  */

#include "newimage/newimageall.h"
#include "libprob.h"
#include "sweopts.h"
#include "swefns.h"

using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace Utilities;
using namespace SWE;
using namespace arma;

// main function
int main(int argc, char *argv[]) {

  Log& logger = LogSingleton::getInstance();
	sweopts& opts = sweopts::getInstance();
	opts.parse_command_line(argc, argv, logger);
	
	if (opts.verbose.value()) {
		cout << "swe options: ";
		for (int i=1; i<argc; i++) cout << argv[i] << " ";
		cout << endl;
	}
    try {
			
			// setting random seed if needed
			if (opts.randomSeed.set()) srand(opts.randomSeed.value());
			if (opts.randomSeed.set() && opts.verbose.value()) cout << "Seeding with " << opts.randomSeed.value() << endl;
			
			// setting up the model
			if (opts.verbose.value()) cout << "Setting up the model: " << flush;
			SweModel sweModel;
			sweModel.setup(opts);
			if (opts.verbose.value()) cout << "done" << endl;

			// issue some warning about the outputs if relevant
			if (!sweModel.outputRaw && (sweModel.wb || !sweModel.outputEquivalent) && (!sweModel.wb || !sweModel.voxelwiseOutput) && !sweModel.outputUncorr) {
				if (sweModel.wb && !sweModel.voxelwiseOutput && (sweModel.clusterThresholdT > 0 || sweModel.clusterThresholdF > 0 || sweModel.clusterMassThresholdT > 0 || sweModel.clusterMassThresholdF > 0)){
					cerr << "Warning! No voxelwise output options selected. Only clusterwise outputs will be generated." << endl;
				}
				else {
					cerr << "Warning! No relevant output options selected. Exiting." << endl;
					return 1;
				}
			}
			
			// loading the data
			if (opts.verbose.value()) cout << "Loading data: " << flush;
			mat data;
			sweModel.loadData(data, opts);
			if (data.n_rows == 0 || data.n_cols == 0)
				throw Exception("No data voxels present.");
			if (opts.verbose.value()) cout << "done" << endl;

			// check the inputs
			if (opts.verbose.value()) cout << "Checking the inputs: " << flush;
			sweModel.checkModel(data.n_rows);
			if (opts.verbose.value()) cout << "done" << endl;
			
			// main computation
			if (sweModel.wb) { // non-parametric inferences
					sweModel.computeStatsWb(data);
			}
			else { // parametric inferences
				if (sweModel.modified) {
					sweModel.computeStatsModifiedSwe(data);
				}
				else {
					sweModel.computeStatsClassicSwe(data);
				}
			}
    }
    catch(Exception& e)
    {
        cerr << "ERROR: Program failed: " <<  e.what() << endl << endl << "Exiting" << endl;
        return 1;
    }
    catch(...)
    {
        cerr << "ERROR: Program failed, unknown exception" << endl << endl << "Exiting" << endl;
        return 1;
    }
    if (opts.verbose.value())
		  cout << "Finished, exiting :-)." << endl;
    return 0;
}
