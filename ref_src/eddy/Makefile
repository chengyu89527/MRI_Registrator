#
# This is the makefile for the eddy project. It can be invoked as
# make, make cuda=1, make cpu=1 or make cuda=1 cpu=1
# 
# In order to build versions for different versions of the
# CUDA tools use
# make CUDA_INSTALLATION=/Developer/NVIDIA/CUDA-x.x
# on a Mac, and where x.x is the desired toolbox version. Or
# make CUDA_INSTALLATION=/opt/cuda-x.x
# or
# make CUDA_INSTALLATION=/usr/local/cuda-x.x
# depending on how you installed the cuda toolbox.
#
# If you combine that with CUDAVER=x.x you will also
# get your executable "labeled correctly"
# 
# By default it will make a fat binary with code for
# all the compute capabilities allowed for by a given
# CUDA version. If one only need 3.0--3.5 (for testing) 
# one can build with the fastbuild=1 option
# which saves a lot of compilation time.
#
include ${FSLCONFDIR}/default.mk

PROJNAME=eddy
debug=a
USRCXXFLAGS=-std=c++11 
TMPCXXFLAGS_1=
TMPCXXFLAGS_2=
TMPNAME_1=
TMPNAME_2=

ifndef cuda
   ifndef cpu #Default single cpu, no threading
	PARALLELFLAGS =
	cpu = 1
   endif
endif

ifeq ($(cuda),1) 
# Additions for CUDAfication of MoveBySusc
	CBF_DIR=../cudabasisfield
	BASISFIELD_DIR=../basisfield
	INC_BASISFIELD=$(BASISFIELD_DIR)

	CUDACXXFLAGS=-DCOMPILE_GPU
	CUDAOBJS=CBFKernelDefinitions.o CBFSparseDiagonalMatrix.o CBFSplineField.o LSResampler_cuda.o DiffusionGP_cuda.o PostEddyCF_cuda.o EddyGpuUtils.o EddyInternalGpuUtils.o CudaVolume.o EddyMatrixKernels.o EddyKernels.o GpuPredictorChunk.o StackResampler.o
	CUDALDFLAGS= -Xlinker -rpath $(LIB_CUDA) -L$(LIB_CUDA) -lcudart
	ifeq ($(fastbuild),1)
		GENCODE_FLAGS    := -gencode arch=compute_30,code=sm_30
	endif
	CUDAEXENAME=eddy_cuda${CUDAVER}
endif

ifeq ($(cpu),1)
	CPUCXXFLAGS=${PARALLELFLAGS}
	CPULDFLAGS=${PARALLELFLAGS}
        CPUOBJS= $(OBJS:%.o=cpubuild/%.o) cpubuild/LSResampler.o cpubuild/PostEddyCF.o
        TMPNAME_2=
        ifdef PARALLELFLAGS	
            TMPNAME_2=_openmp
        endif
	EXENAME=eddy${TMPNAME_1}${TMPNAME_2}
endif
USRLDFLAGS=-L${LIB_NEWMAT} -Wl,-rpath,/opt/fmrib/fsl/lib -rdynamic
USRINCFLAGS=-I. -I${INC_NEWMAT} -I${INC_PROB} -I${INC_BOOST} -I${INC_CUDA} -I$(CBF_DIR) -I$(INC_BASISFIELD)

OBJS=eddy.o BiasFieldEstimatorImpl.o MoveBySuscCF.o PostEddyAlignShellsFunctions.o EddyCommandLineOptions.o ECModels.o KMatrix.o HyParEstimator.o ECScanClasses.o EddyUtils.o EddyHelperClasses.o DiffusionGP.o b0Predictor.o
TESTOBJS=test_s2v_model.o PostEddyAlignShellsFunctions.o EddyCommandLineOptions.o ECModels.o KMatrix.o HyParEstimator.o ECScanClasses.o EddyUtils.o EddyHelperClasses.o DiffusionGP.o b0Predictor.o 
TESTMIBINSOBJS=test_MI_bins.o PostEddyAlignShellsFunctions.o EddyCommandLineOptions.o ECModels.o KMatrix.o HyParEstimator.o ECScanClasses.o EddyUtils.o EddyHelperClasses.o DiffusionGP.o b0Predictor.o 
MASSAGEOBJS=massage_fieldmap.o
TOPUPOBJS=../topup/topup_matrices.o ../topup/topup_file_io.o ../topup/displacement_vector.o 
LIBS=-lwarpfns -lmeshclass -lbasisfield -lnewimage -lmiscmaths -lprob -lNewNifti -lznz -lutils -lnewmat -lm -lz
ifdef NVCC11
	NVCC=${NVCC11}
endif
NVCCFLAGS=-c -DCOMPILE_GPU -O3 -m 64 -std=c++11
ifeq ($(CLOBBER_CLANG),1)
   NVCCFLAGS+= -DCLOBBER_CLANG
endif

XFILES=${CUDAEXENAME} ${EXENAME}

ifeq ($(FSLMASTERBUILD),1)
all:
#	for params in ${EDDYBUILDPARAMETERS}; do echo Starting sub-build: $${params}; $(MAKE) FSLMASTERBUILD=0 --always-make $${params}; done
install:
#	for params in ${EDDYBUILDPARAMETERS}; do echo Starting sub-install: $${params}; $(MAKE) install FSLMASTERBUILD=0 $${params}; done
	for params in ${EDDYBUILDPARAMETERS}; do echo Starting sub-install: $${params}; $(MAKE) install FSLMASTERBUILD=0 $${params} distclean; done
else
all: ${XFILES}
endif

CBFKernelDefinitions.o : $(CBF_DIR)/CBFKernelDefinitions.cu \
                           $(CBF_DIR)/CBFKernels.cuh \
                           $(CBF_DIR)/CBFKernelHelpers.cuh
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o CBFKernelDefinitions.o $(CBF_DIR)/CBFKernelDefinitions.cu ${AccumulatedIncFlags}

CBFSparseDiagonalMatrix.o : $(CBF_DIR)/CBFSparseDiagonalMatrix.cu \
                             $(CBF_DIR)/CBFSparseDiagonalMatrix.cuh
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o CBFSparseDiagonalMatrix.o $(CBF_DIR)/CBFSparseDiagonalMatrix.cu ${AccumulatedIncFlags}

CBFSplineField.o : $(CBF_DIR)/CBFSplineField.cu \
                    $(CBF_DIR)/CBFSplineField.cuh \
                    $(CBF_DIR)/CBFMemory.h \
                    $(CBF_DIR)/CBFSparseDiagonalMatrix.cuh \
                    $(CBF_DIR)/CBFKernels.cuh \
                    $(CBF_DIR)/CBFKernelHelpers.cuh
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o CBFSplineField.o $(CBF_DIR)/CBFSplineField.cu ${AccumulatedIncFlags}

GpuPredictorChunk.o: cuda/GpuPredictorChunk.h cuda/GpuPredictorChunk.cu EddyHelperClasses.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o GpuPredictorChunk.o cuda/GpuPredictorChunk.cu ${AccumulatedIncFlags}

StackResampler.o: cuda/StackResampler.h cuda/StackResampler.cu EddyHelperClasses.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o StackResampler.o cuda/StackResampler.cu ${AccumulatedIncFlags}

EddyKernels.o: cuda/EddyKernels.h cuda/EddyKernels.cu 
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o EddyKernels.o cuda/EddyKernels.cu ${AccumulatedIncFlags}

EddyMatrixKernels.o: cuda/EddyMatrixKernels.h cuda/eddy_matrix_kernels_internal.h cuda/EddyMatrixKernels.cu 
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o EddyMatrixKernels.o cuda/EddyMatrixKernels.cu ${AccumulatedIncFlags}

CudaVolume.o: cuda/CudaVolume.h cuda/CudaVolume.cu cuda/EddyKernels.h cuda/EddyFunctors.h EddyHelperClasses.h 
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o CudaVolume.o cuda/CudaVolume.cu ${AccumulatedIncFlags}

EddyInternalGpuUtils.o: cuda/EddyInternalGpuUtils.h cuda/EddyInternalGpuUtils.cu cuda/CudaVolume.h cuda/EddyKernels.h \
	cuda/EddyFunctors.h cuda/StackResampler.h EddyHelperClasses.h DiffusionGP.h b0Predictor.h ECScanClasses.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o EddyInternalGpuUtils.o cuda/EddyInternalGpuUtils.cu ${AccumulatedIncFlags}

EddyGpuUtils.o: cuda/EddyGpuUtils.h cuda/EddyGpuUtils.cu cuda/EddyInternalGpuUtils.h cuda/EddyKernels.h \
	EddyHelperClasses.h DiffusionGP.h b0Predictor.h ECScanClasses.h EddyUtils.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o EddyGpuUtils.o cuda/EddyGpuUtils.cu ${AccumulatedIncFlags}

DiffusionGP_cuda.o : cuda/DiffusionGP.cu DiffusionGP.h EddyHelperClasses.h DWIPredictionMaker.h EddyUtils.h cuda/CudaVolume.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o DiffusionGP_cuda.o cuda/DiffusionGP.cu ${AccumulatedIncFlags}

LSResampler_cuda.o : cuda/LSResampler.cu LSResampler.h EddyHelperClasses.h EddyUtils.h ECScanClasses.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o LSResampler_cuda.o cuda/LSResampler.cu ${AccumulatedIncFlags}

PostEddyCF_cuda.o : cuda/PostEddyCF.cu PostEddyCF.h cuda/CudaVolume.h EddyUtils.h cuda/EddyInternalGpuUtils.h
	${NVCC} ${NVCCFLAGS} ${GENCODE_FLAGS} -o PostEddyCF_cuda.o cuda/PostEddyCF.cu ${AccumulatedIncFlags}

${EXENAME}: ${CPUOBJS} ${TOPUPOBJS}
	${CXX11} ${CPUCXXFLAGS} ${LDFLAGS} ${CPULDFLAGS} -o ${EXENAME} ${CPUOBJS} ${TOPUPOBJS} ${LIBS} 

${CUDAEXENAME}: USRCXXFLAGS+=-DCOMPILE_GPU
${CUDAEXENAME}: ${OBJS} ${CUDAOBJS} ${TOPUPOBJS}
	${CXX11} ${CUDACXXFLAGS} ${LDFLAGS} -o ${CUDAEXENAME} ${OBJS} ${TOPUPOBJS} ${CUDAOBJS} ${LIBS} ${CUDALDFLAGS}

cpubuild/%.o:%.cpp
	@mkdir -p cpubuild
	$(CXX11) -c $(CPPFLAGS) ${CPUCXXFLAGS} $(CXXFLAGS) $< -o $@

%.o:%.cpp
	$(CXX11) -c $(CPPFLAGS) $(CXXFLAGS) $< -o $@

test_s2v_model: ${TESTOBJS} ${CUDAOBJS} ${TOPUPOBJS}
	${CXX11} ${CUDACXXFLAGS} ${LDFLAGS} -o test_s2v_model ${TESTOBJS} ${TOPUPOBJS} ${CUDAOBJS} ${LIBS} ${CUDALDFLAGS}

test_MI_bins: ${TESTMIBINSOBJS} ${CUDAOBJS} ${TOPUPOBJS}
	${CXX11} ${CUDACXXFLAGS} ${LDFLAGS} -o test_MI_bins ${TESTMIBINSOBJS} ${TOPUPOBJS} ${CUDAOBJS} ${LIBS} ${CUDALDFLAGS}

massage_fieldmap: ${MASSAGEOBJS} ${CUDAOBJS} ${TOPUPOBJS}
	${CXX11} ${CUDACXXFLAGS} ${LDFLAGS} -o massage_fieldmap ${MASSAGEOBJS} ${TOPUPOBJS} ${CUDAOBJS} ${LIBS} ${CUDALDFLAGS}

print-% : ; @echo $* = $($*)

clean:
	${RM} -f /tmp/fslgrot *.o *.a *.exe core depend.mk eddy eddy_cuda* eddy_openmp* `[ ! -d cpubuild ] || find cpubuild -name "*" -type f`
	@[ ! -d cpubuild ] || rmdir cpubuild

cleanish:
	${RM} -f /tmp/fslgrot *.o *.a *.exe core depend.mk `[ ! -d cpubuild ] || find cpubuild -name "*" -type f`
	@[ ! -d cpubuild ] || rmdir cpubuild
