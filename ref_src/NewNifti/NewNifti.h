/*
  Matthew Webster (WIN@FMRIB)
  Copyright (C) 2018 University of Oxford  */
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
#if !defined(__newnifti_h)
#define __newnifti_h

#include <string>
#include <vector>

#if __cplusplus >= 201103L || __clang__
  #include <array>
using std::array;
#else
  #include <tr1/array>
using std::tr1::array;
#endif

#include "nifti2.h"
#include "legacyFunctions.h"
#include "znzlib/znzlib.h"

namespace NiftiIO {

  enum HeaderType { analyze, nifti1, nifti2, other=-1};


  struct NiftiException : public std::exception
  {
    std::string errorMessage;
    NiftiException(const std::string& error) : errorMessage(error) {}
    ~NiftiException() throw() {}
    const char* what() const throw() { return errorMessage.c_str(); }
  };


  class NiftiExtension
  {
  public:
    int32_t    esize ;
    int32_t    ecode ;
    std::vector<char> edata ;
    size_t extensionSize() const { return( sizeof(esize) + sizeof(ecode) + edata.size() ); }
  };


  class NiftiHeader
  {
  private:
    void initialise(void);
  public:
  NiftiHeader();
  //Common Header parameters
  int sizeof_hdr;
  int64_t vox_offset;
  std::string magic;
  short datatype;
  int bitsPerVoxel;
  array<int64_t,8> dim;
  array<double,8> pixdim;
  int intentCode;
  double intent_p1;
  double intent_p2;
  double intent_p3;
  std::string intentName;
  double sclSlope;
  double sclInter;
  double cal_max;
  double cal_min;
  double sliceDuration;
  double toffset;
  int64_t sliceStart;
  int64_t sliceEnd;
  std::string description;
  std::string auxillaryFile;
  int qformCode;
  int sformCode;
  double qB;
  double qC;
  double qD;
  double qX;
  double qY;
  double qZ;
  array<double,4> sX;
  array<double,4> sY;
  array<double,4> sZ;
  int sliceCode;
  int units;
  char sliceOrdering;
  LegacyFields legacyFields;
  //Useful extras
  std::string fileType() const;
  std::string niftiOrientationString( const int orientation ) const;
  std::string niftiSliceString() const;
  std::string niftiIntentString() const;
  std::string niftiTransformString(const int transform) const;
  std::string unitsString(const int units) const;
  std::string datatypeString() const;
  std::string originalOrder() const;
  void sanitise();
  char freqDim() const { return sliceOrdering & 3; }
  char phaseDim() const { return (sliceOrdering >> 2) & 3; }
  char sliceDim() const { return (sliceOrdering >> 4) & 3; }
  char niftiVersion() const { return NIFTI2_VERSION(*this); }
  bool isAnalyze() const { return ( (NIFTI2_VERSION(*this) == 1) && (NIFTI_VERSION(*this) == 0) ); }
  bool wasWrongEndian;
  bool singleFile() const { return NIFTI_ONEFILE(*this); }
  size_t nElements() const { size_t elements(dim[1]); for (int dims=2;dims<=dim[0];dims++) elements*=dim[dims]; return elements; }
  int bpvOfDatatype(void);
  mat44 getQForm() const;
  void setQForm(const mat44& qForm);
  std::string qFormName() const { return niftiTransformString(qformCode);}
  mat44 getSForm() const;
  void setSForm(const mat44& sForm);
  std::string sFormName() const { return niftiTransformString(qformCode);}
  int leftHanded() const { return( (pixdim[0] < 0.0) ? -1.0 : 1.0 ) ; }
  int64_t nominalVoxOffset() { return( ( singleFile() && vox_offset < 352 ) ? 352 : vox_offset ); }
  void setNiftiVersion( const char niftiVersion, const bool isSingleFile );
  void report() const;
  template<class T>
    void readAllFieldsFromRaw(const T& rawHeader);
  template<class T>
    void readCommonFieldsFromRaw(const T& rawHeader);
  template<class T>
    void readNiftiFieldsFromRaw(const T& rawNiftiHeader);
  template<class T>
    void writeAllFieldsToRaw(T& rawHeader) const;
  template<class T>
    void writeCommonFieldsToRaw(T& rawHeader) const;
  template<class T>
    void writeNiftiFieldsToRaw(T& rawNiftiHeader) const;
  };

    NiftiHeader loadExtensions(const std::string filename, std::vector<NiftiExtension>& extensions);
    NiftiHeader loadHeader(const std::string filename);
    NiftiHeader loadImage( std::string filename, char*& buffer, std::vector<NiftiExtension>& extensions, bool allocateBuffer=true);
    NiftiHeader loadImageROI( std::string filename, char*& buffer, std::vector<NiftiExtension>& extensions, int64_t xmin=-1, int64_t xmax=-1, int64_t ymin=-1, int64_t ymax=-1, int64_t zmin=-1, int64_t zmax=-1, int64_t tmin=-1, int64_t tmax=-1, int64_t d5min=-1, int64_t d5max=-1, int64_t d6min=-1, int64_t d6max=-1, int64_t d7min=-1, int64_t d7max=-1);
    void reportHeader(const analyzeHeader& header);
    void reportHeader(const nifti_1_header& header);
    void reportHeader(const nifti_2_header& header);
    void saveImage(const std::string filename, const char* buffer, const std::vector<NiftiExtension>& extensions, const NiftiHeader header, const bool useCompression=true);
    template<class T>
      void byteSwap(T& rawNiftiHeader);
    void byteSwap(const size_t elementLength, void* buffer,const unsigned long nElements=1);
    void byteSwapAnalyzeFields(analyzeHeader& rawHeader);
    template<class T>
      void byteSwapImageFields(T& rawHeader);
    template<class T>
      void byteSwapLegacyFields(T& rawHeader);
    template<class T>
      void byteSwapNiftiFields(T& rawNiftiHeader);
    template<class T>
    HeaderType headerType(const T& header);
    void reportLegacyHeader(const nifti_1_header& header);


  //fileIO
  //This class actually does the reading/writing to/from files
  //An instance _must_ be constructed with a file
  //The file will be closed on destruction of the instance
  //To open a new file, use the assignment operator, e.g.
  // instance = fileIO(newFile)
  class fileIO
  {
  public:
    bool debug;

    fileIO(const std::string& filename, const bool reading, const bool useCompression=true);
    ~fileIO() { znzclose(fileHandle); }
    fileIO& operator=(fileIO rhs);
    void reset() {znzclose(fileHandle);}
    void readData(const NiftiHeader& header,void* buffer);
    void readExtensions(const NiftiHeader& header, std::vector<NiftiExtension>& extensions );
    NiftiHeader readHeader();
    HeaderType readHeaderType();
    size_t readRawBytes( void *buffer, size_t length );
    template <class T>
      void readRawHeader(NiftiHeader& header);
    void seek(size_t nBytes, int mode) { znzseek(fileHandle, nBytes, mode); }
    void writeData(const NiftiHeader& header,const void* buffer);
    void writeExtensions(const NiftiHeader& header, const std::vector<NiftiExtension>& extensions );
    void writeHeader(const NiftiHeader& header);
    void writeRawBytes( const void *buffer, size_t length );
    template<class T>
      void writeRawHeader(const NiftiHeader& header);
  private:
    znzFile fileHandle;
  };


  //Explicit specialisations declared here
  template<> void byteSwap(nifti_1_header& rawHeader);
  template<> void byteSwap(nifti_2_header& rawHeader);
  template<> void byteSwap(analyzeHeader& rawHeader);
}

#endif
