/*! \file EddyHelperClasses.h
    \brief Contains declaration of classes that implements useful functionality for the eddy project.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of classes that implements useful
// concepts for the eddy current project.
// 
// EddyHelperClasses.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#ifndef EddyHelperClasses_h
#define EddyHelperClasses_h

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sys/time.h>
#include <boost/current_function.hpp>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"

#ifndef TicToc
#define TicToc(task) { timeval tim;		\
  gettimeofday(&tim,NULL); \
  task; \
  timeval tim2; \
  gettimeofday(&tim2,NULL); \
  cout << "Call to " #task " took " << 1000000*(tim2.tv_sec-tim.tv_sec) + tim2.tv_usec - tim.tv_usec << " usec" << endl; }
#endif


#define UseEddyTry

#ifdef UseEddyTry
  #ifndef EddyTry
    #define EddyTry try
    #define EddyCatch catch (const exception& e) { cout << e.what() << endl; throw EDDY::EddyException(string(__FILE__) + ":::  " + string(BOOST_CURRENT_FUNCTION) + ":  Exception thrown"); }
  #endif
#else 
  #ifndef EddyTry
    #define EddyTry
    #define EddyCatch
  #endif
# endif 

namespace EDDY {

enum Parameters { ZERO_ORDER_MOVEMENT, MOVEMENT, EC, ALL };
enum ECModel { NoEC, Linear, Quadratic, Cubic, Unknown };
enum SecondLevelECModel { No_2nd_lvl_mdl, Linear_2nd_lvl_mdl, Quadratic_2nd_lvl_mdl, Unknown_2nd_lvl_mdl };
enum OffsetModel { LinearOffset, QuadraticOffset, UnknownOffset };
enum OLType { SliceWise, GroupWise, Both };
enum ScanType { DWI, B0 , ANY };
enum FinalResampling { JAC, LSR, UNKNOWN_RESAMPLING};

 
class EddyException: public std::exception
{
public:
  EddyException(const std::string& msg) noexcept : m_msg(msg) {}
  ~EddyException() noexcept {}
  virtual const char * what() const noexcept { return string("EDDY:::  " + m_msg).c_str(); }
private:
  std::string m_msg;
};

 
class JsonReader
{
public:
  JsonReader();
  JsonReader(const std::string& fname) EddyTry : _fname(fname) { common_read(); } EddyCatch
  void Read(const std::string& fname) EddyTry { _fname = fname; common_read(); } EddyCatch
  
  NEWMAT::Matrix SliceOrdering() const;
  
  NEWMAT::ColumnVector PEVector() const;
  
  double TotalReadoutTime() const { return(0.05); }
private:
  void common_read();
  std::string _fname;
  std::string _content;
};

 
class DiffPara
{
public:
  
  DiffPara() EddyTry { _bvec.ReSize(3); _bvec(1)=1; _bvec(2)=0; _bvec(3)=0; _bval = 0; } EddyCatch
  
  DiffPara(double bval) EddyTry : _bval(bval) { _bvec.ReSize(3); _bvec(1)=1; _bvec(2)=0; _bvec(3)=0; } EddyCatch
  
  DiffPara(const NEWMAT::ColumnVector&   bvec,
	   double                        bval) EddyTry : _bvec(bvec), _bval(bval)
  {
    if (_bvec.Nrows() != 3) throw EddyException("DiffPara::DiffPara: b-vector must be three elements long");
    if (_bval < 0) throw EddyException("DiffPara::DiffPara: b-value must be non-negative");
    if (_bval) _bvec /= _bvec.NormFrobenius();
  } EddyCatch
  
  friend ostream& operator<<(ostream& op, const DiffPara& dp) EddyTry { op << "b-vector: " << dp._bvec.t() << endl << "b-value:  " << dp._bval << endl; return(op); } EddyCatch
  
  bool operator==(const DiffPara& rhs) const;
  
  bool operator!=(const DiffPara& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  
  bool operator>(const DiffPara& rhs) const EddyTry { return(this->bVal()>rhs.bVal()); } EddyCatch
  
  bool operator<(const DiffPara& rhs) const EddyTry { return(this->bVal()<rhs.bVal()); } EddyCatch
  
  NEWMAT::ColumnVector bVec() const EddyTry { return(_bvec); } EddyCatch
  
  double bVal() const { return(_bval); } 
private:
  NEWMAT::ColumnVector _bvec;
  double               _bval;
};

 
class AcqPara
{
public:
  
  AcqPara(const NEWMAT::ColumnVector&   pevec,
          double                        rotime);
  
  friend ostream& operator<<(ostream& op, const AcqPara& ap) EddyTry { op << "Phase-encode vector: " << ap._pevec.t() << endl << "Read-out time: " << ap._rotime; return(op); } EddyCatch
  
  bool operator==(const AcqPara& rh) const;
  
  bool operator!=(const AcqPara& rh) const EddyTry { return(!(*this == rh)); } EddyCatch
  
  NEWMAT::ColumnVector PhaseEncodeVector() const EddyTry { return(_pevec); } EddyCatch
  
  std::vector<unsigned int> BinarisedPhaseEncodeVector() const;
  
  double ReadOutTime() const { return(_rotime); }
private:
  NEWMAT::ColumnVector _pevec;
  double               _rotime;
};

class PolationPara
{
public:
  PolationPara() EddyTry : _int(NEWIMAGE::spline), _ext(NEWIMAGE::periodic), _evip(true), _s2v_int(NEWIMAGE::spline) {} EddyCatch
  PolationPara(NEWIMAGE::interpolation ip, NEWIMAGE::extrapolation ep, bool evip, NEWIMAGE::interpolation s2v_ip) EddyTry {
    SetInterp(ip); SetExtrap(ep); SetExtrapValidity(evip); SetS2VInterp(s2v_ip);
  } EddyCatch
  NEWIMAGE::interpolation GetInterp() const { return(_int); }
  NEWIMAGE::interpolation GetS2VInterp() const { return(_s2v_int); } 
  NEWIMAGE::extrapolation GetExtrap() const { return(_ext); } 
  bool GetExtrapValidity() const { return(_evip); } 
  void SetInterp(NEWIMAGE::interpolation ip) EddyTry { 
    if (ip!=NEWIMAGE::trilinear && ip!=NEWIMAGE::spline) throw EddyException("PolationPara::SetInterp: Invalid interpolation"); 
    _int = ip;
  } EddyCatch
  void SetS2VInterp(NEWIMAGE::interpolation ip) EddyTry { 
    if (ip!=NEWIMAGE::trilinear && ip!=NEWIMAGE::spline) throw EddyException("PolationPara::SetS2VInterp: Invalid interpolation"); 
    _s2v_int = ip;
  } EddyCatch
  void SetExtrap(NEWIMAGE::extrapolation ep) EddyTry {
    if (ep!=NEWIMAGE::mirror && ep!=NEWIMAGE::periodic) throw EddyException("PolationPara::SetExtrap: Invalid extrapolation"); 
    if (ep!=NEWIMAGE::periodic && _evip) throw EddyException("PolationPara::SetExtrap: Invalid extrapolation and validity combo"); 
    _ext = ep;
  } EddyCatch
  void SetExtrapValidity(bool evip) EddyTry {
    if (evip && _ext!=NEWIMAGE::periodic) throw EddyException("PolationPara::SetExtrapValidity: Invalid extrapolation and validity combo"); 
    _evip = evip;
  } EddyCatch
  
  friend std::ostream& operator<<(std::ostream& out, const PolationPara& pp) EddyTry {
    out << "PolationPara:" << endl;
    if (pp._int == NEWIMAGE::trilinear) out << "Interpolation: trilinear" << endl;
    else out << "Interpolation: spline" << endl;
    if (pp._ext == NEWIMAGE::mirror) out << "Extrapolation: mirror" << endl;
    else out << "Extrapolation: periodic" << endl;
    if (pp._evip) out << "Extrapolation along EP is valid" << endl;
    else out << "Extrapolation along EP is not valid" << endl;
    if (pp._s2v_int == NEWIMAGE::trilinear) out << "Slice-to-vol interpolation: trilinear" << endl;
    else out << "Slice-to-vol interpolation: spline" << endl;
    return(out); 
  } EddyCatch
private:
  NEWIMAGE::interpolation _int;        
  NEWIMAGE::extrapolation _ext;        
  bool                    _evip;       
  NEWIMAGE::interpolation _s2v_int;    
};

class JacMasking
{
public:
  JacMasking(bool doit, double ll, double ul) : _doit(doit), _ll(ll), _ul(ul) {} 
  ~JacMasking() {}
  bool DoIt() const { return(_doit); } 
  double LowerLimit() const { return(_ll); } 
  double UpperLimit() const { return(_ul); }
private:
  bool    _doit;
  double  _ll;
  double  _ul;
};

class ReferenceScans
{
public:
  ReferenceScans() EddyTry : _loc_ref(0), _b0_loc_ref(0), _b0_shape_ref(0), _dwi_loc_ref(0), _shell_loc_ref(1,0), _shell_shape_ref(1,0) {} EddyCatch
  ReferenceScans(std::vector<unsigned int> b0indx, std::vector<std::vector<unsigned int> > shindx) EddyTry : _loc_ref(0), _shell_loc_ref(shindx.size()), _shell_shape_ref(shindx.size()) {
    _b0_loc_ref = (b0indx.size() > 0 ? b0indx[0] : 0); _b0_shape_ref = (b0indx.size() > 0 ? b0indx[0] : 0);
    _dwi_loc_ref = ((shindx.size() && shindx[0].size()) ? shindx[0][0] : 0);
    for (unsigned int i=0; i<shindx.size(); i++) { _shell_loc_ref[i] = shindx[i][0]; _shell_shape_ref[i] = shindx[i][0]; }
  } EddyCatch
  unsigned int GetLocationReference() const { return(_loc_ref); }
  unsigned int GetB0LocationReference() const { return(_b0_loc_ref); } 
  unsigned int GetB0ShapeReference() const { return(_b0_shape_ref); } 
  unsigned int GetDWILocationReference() const { return(_dwi_loc_ref); } 
  unsigned int GetShellLocationReference(unsigned int si) const EddyTry { 
    if (si>=_shell_loc_ref.size()) throw EddyException("ReferenceScans::GetShellLocationReference: Shell index out of range"); 
    else return(_shell_loc_ref[si]);
  } EddyCatch
  unsigned int GetShellShapeReference(unsigned int si) const EddyTry { 
    if (si>=_shell_shape_ref.size()) throw EddyException("ReferenceScans::GetShellShapeReference: Shell index out of range"); 
    else return(_shell_shape_ref[si]);
  } EddyCatch
  void SetLocationReference(unsigned int indx) { _loc_ref=indx; }
  void SetB0LocationReference(unsigned int indx) { _b0_loc_ref=indx; } 
  void SetB0ShapeReference(unsigned int indx) { _b0_shape_ref=indx; } 
  void SetDWILocationReference(unsigned int indx) { _dwi_loc_ref=indx; } 
  void SetShellLocationReference(unsigned int si, unsigned int indx) EddyTry { 
    if (si>=_shell_loc_ref.size()) throw EddyException("ReferenceScans::SetShellLocationReference: Shell index out of range"); 
    _shell_loc_ref[si] = indx;
  } EddyCatch
  void SetShellShapeReference(unsigned int si, unsigned int indx) EddyTry { 
    if (si>=_shell_shape_ref.size()) throw EddyException("ReferenceScans::SetShellShapeReference: Shell index out of range"); 
    _shell_shape_ref[si] = indx;
  } EddyCatch
private:
  
  unsigned int                _loc_ref;          
  unsigned int                _b0_loc_ref;       
  unsigned int                _b0_shape_ref;     
  unsigned int                _dwi_loc_ref;      
  std::vector<unsigned int>   _shell_loc_ref;    
  std::vector<unsigned int>   _shell_shape_ref;  
};










class MaskManager
{
public:
  MaskManager(const NEWIMAGE::volume<float>& mask) EddyTry : _mask(mask) {} EddyCatch
  MaskManager(int xs, int ys, int zs) EddyTry : _mask(xs,ys,zs) { _mask = 1.0; } EddyCatch
  void ResetMask() EddyTry { _mask = 1.0; } EddyCatch
  void SetMask(const NEWIMAGE::volume<float>& mask) EddyTry { if (!NEWIMAGE::samesize(_mask,mask)) throw EddyException("EDDY::MaskManager::SetMask: Wrong dimension"); else _mask = mask;} EddyCatch
  void UpdateMask(const NEWIMAGE::volume<float>& mask) EddyTry { if (!NEWIMAGE::samesize(_mask,mask)) throw EddyException("EDDY::MaskManager::UpdateMask: Wrong dimension"); else _mask *= mask;} EddyCatch
  const NEWIMAGE::volume<float>& GetMask() const EddyTry { return(_mask); } EddyCatch
private:
  NEWIMAGE::volume<float> _mask;
};

 
class DiffStats
{
public:
  DiffStats() {}
  
  DiffStats(const NEWIMAGE::volume<float>& diff, const NEWIMAGE::volume<float>& mask);
  
  double MeanDifference() const EddyTry { return(mean_stat(_md)); } EddyCatch 
  
  double MeanDifference(int sl) const EddyTry { if (index_ok(sl)) return(_md[sl]); else return(0.0); } EddyCatch
  
  NEWMAT::RowVector MeanDifferenceVector() const EddyTry { return(get_vector(_md)); } EddyCatch
  
  double MeanSqrDiff() const EddyTry { return(mean_stat(_msd)); } EddyCatch
  
  double MeanSqrDiff(int sl) const EddyTry { if (index_ok(sl)) return(_msd[sl]); else return(0.0); } EddyCatch
  
  NEWMAT::RowVector MeanSqrDiffVector() const EddyTry { return(get_vector(_msd)); } EddyCatch
  
  unsigned int NVox() const EddyTry { unsigned int n=0; for (int i=0; i<int(_n.size()); i++) n+=_n[i]; return(n); } EddyCatch
  
  unsigned int NVox(int sl) const EddyTry { if (index_ok(sl)) return(_n[sl]); else return(0); } EddyCatch
  
  NEWMAT::RowVector NVoxVector() const EddyTry { return(get_vector(_n)); } EddyCatch
  
  unsigned int NSlice() const EddyTry { return(_n.size()); } EddyCatch
private:
  std::vector<double>        _md;  
  std::vector<double>        _msd; 
  std::vector<unsigned int>  _n;   

  bool index_ok(int sl) const EddyTry
  { if (sl<0 || sl>=int(_n.size())) throw EddyException("DiffStats::index_ok: Index out of range"); return(true); } EddyCatch

  double mean_stat(const std::vector<double>& stat) const EddyTry
  { double ms=0; for (int i=0; i<int(_n.size()); i++) ms+=_n[i]*stat[i]; ms/=double(NVox()); return(ms); } EddyCatch

  template<class T>
  NEWMAT::RowVector get_vector(const std::vector<T>& stat) const EddyTry
  { NEWMAT::RowVector ov(stat.size()); for (unsigned int i=0; i<stat.size(); i++) ov(i+1) = double(stat[i]); return(ov); } EddyCatch
};

 
class MultiBandGroups
{
public:
  MultiBandGroups(unsigned int nsl, unsigned int mb=1, int offs=0);
  
  MultiBandGroups(const std::string& fname);
  
  MultiBandGroups(const NEWMAT::Matrix& mat);
  void SetTemporalOrder(const std::vector<unsigned int>& to) EddyTry { 
    if (to.size() != _to.size()) throw EddyException("MultiBandGroups::SetTemporalOrder: to size mismatch"); else _to=to; 
  } EddyCatch
  unsigned int MBFactor() const { return(_mb); } 
  unsigned int NGroups() const EddyTry { return(_grps.size()); } EddyCatch
  const std::vector<unsigned int>& SlicesInGroup(unsigned int grp_i) const EddyTry {
    if (grp_i >= _grps.size()) throw EddyException("MultiBandGroups::SlicesInGroup: Group index out of range");
    else return(_grps[grp_i]);
  } EddyCatch
  const std::vector<unsigned int>& SlicesAtTimePoint(unsigned int time_i) const EddyTry {
    if (time_i >= _grps.size()) throw EddyException("MultiBandGroups::SlicesAtTimePoint: Time index out of range");
    else return(_grps[_to[time_i]]);
  } EddyCatch
  friend ostream& operator<<(ostream& os, const MultiBandGroups& mbg) EddyTry
  {
    for (unsigned int i=0; i<mbg._grps.size(); i++) {
      for (unsigned int j=0; j<mbg._grps[i].size(); j++) os << std::setw(5) << mbg._grps[i][j]; 
      os << endl;
    }
    return(os);
  } EddyCatch
private:
  unsigned int                            _nsl;  
  unsigned int                            _mb;   
  int                                     _offs; 
  std::vector<std::vector<unsigned int> > _grps; 
  
  std::vector<unsigned int>               _to;

  
  void assert_grps();   
};

 
class DiffStatsVector
{
public:
  
  DiffStatsVector(unsigned int n) EddyTry : _n(n) { _ds = new DiffStats[_n]; } EddyCatch
  
  DiffStatsVector(const DiffStatsVector& rhs) EddyTry : _n(rhs._n) { _ds = new DiffStats[_n]; for (unsigned int i=0; i<_n; i++) _ds[i] = rhs._ds[i]; } EddyCatch
  ~DiffStatsVector() EddyTry { delete [] _ds; } EddyCatch
  
  DiffStatsVector& operator=(const DiffStatsVector& rhs) EddyTry {
    delete [] _ds; _n = rhs._n; _ds = new DiffStats[_n]; for (unsigned int i=0; i<_n; i++) _ds[i] = rhs._ds[i]; return(*this);
  } EddyCatch
  
  const DiffStats& operator[](unsigned int i) const EddyTry { throw_if_oor(i); return(_ds[i]); } EddyCatch
  
  DiffStats& operator[](unsigned int i) EddyTry { throw_if_oor(i); return(_ds[i]); } EddyCatch
  
  unsigned int NScan() const { return(_n); }
  
  unsigned int NSlice() const EddyTry { return(_ds[0].NSlice()); } EddyCatch
  
  double MeanDiff(unsigned int sc, unsigned int sl) const EddyTry { throw_if_oor(sc); return(_ds[sc].MeanDifference(int(sl))); } EddyCatch
  
  double MeanSqrDiff(unsigned int sc, unsigned int sl) const EddyTry { throw_if_oor(sc); return(_ds[sc].MeanSqrDiff(int(sl))); } EddyCatch
  
  unsigned int NVox(unsigned int sc, unsigned int sl) const EddyTry { throw_if_oor(sc); return(_ds[sc].NVox(int(sl))); } EddyCatch
  
  void Write(const std::string& bfname) const;
private:
  unsigned int _n;   
  DiffStats    *_ds; 

  void throw_if_oor(unsigned int i) const EddyTry { if (i >= _n) throw EddyException("DiffStatsVector::throw_if_oor: Index out of range"); } EddyCatch
};

 
class OutlierDefinition {
public:
  OutlierDefinition(double        nstdev,    
		    unsigned int  minn,      
		    bool          pos,       
		    bool          sqr)       
  : _nstdev(nstdev), _minn(minn), _pos(pos), _sqr(sqr) {}
  OutlierDefinition() : _nstdev(4.0), _minn(250), _pos(false), _sqr(false) {}
  double NStdDev() const { return(_nstdev); }
  unsigned int MinVoxels() const { return(_minn); }
  bool ConsiderPosOL() const { return(_pos); }
  bool ConsiderSqrOL() const { return(_sqr); }
private:
  double        _nstdev;    
  unsigned int  _minn;      
  bool          _pos;       
  bool          _sqr;       
};

 
class ReplacementManager {
public:
  ReplacementManager(unsigned int              nscan,  
		     unsigned int              nsl,    
		     const OutlierDefinition&  old,    
		     unsigned int              etc,    
		     OLType                    olt,   
		     const MultiBandGroups&    mbg)    
  EddyTry : _old(old), _etc(etc), _olt(olt), _mbg(mbg), _sws(nsl,nscan), _gws(mbg.NGroups(),nscan), _swo(nsl,nscan), _gwo(mbg.NGroups(),nscan) 
  {
  if (_etc != 1 && _etc != 2) throw  EddyException("ReplacementManager::ReplacementManager: etc must be 1 or 2");
  } EddyCatch
  ~ReplacementManager() EddyTry {} EddyCatch
  unsigned int NSlice() const EddyTry { return(_swo._ovv.size()); } EddyCatch
  unsigned int NScan() const EddyTry { unsigned int rval = (_swo._ovv.size()) ? _swo._ovv[0].size() : 0; return(rval); } EddyCatch
  unsigned int NGroup() const EddyTry { return(_mbg.NGroups()); } EddyCatch
  void Update(const DiffStatsVector& dsv);
  std::vector<unsigned int> OutliersInScan(unsigned int scan) const;
  bool ScanHasOutliers(unsigned int scan) const;
  bool IsAnOutlier(unsigned int slice, unsigned int scan) const EddyTry { return(_swo._ovv[slice][scan]); } EddyCatch
  void WriteReport(const std::vector<unsigned int>& i2i,
		   const std::string&               bfname) const;
  void WriteMatrixReport(const std::vector<unsigned int>& i2i,
			 unsigned int                     nscan,
			 const std::string&               om_fname,
			 const std::string&               nstdev_fname,
			 const std::string&               n_sqr_stdev_fname) const;
  
  void DumpOutlierMaps(const std::string& fname) const;
  
  struct OutlierInfo {
    std::vector<std::vector<bool> >           _ovv;     
    std::vector<std::vector<double> >         _nsv;     
    std::vector<std::vector<double> >         _nsq;     
    OutlierInfo(unsigned int nsl, unsigned int nscan) EddyTry : _ovv(nsl), _nsv(nsl), _nsq(nsl) { 
      for (unsigned int i=0; i<nsl; i++) { _ovv[i].resize(nscan,false); _nsv[i].resize(nscan,0.0); _nsq[i].resize(nscan,0.0); }
    } EddyCatch
  };
  
  struct StatsInfo {
    std::vector<std::vector<unsigned int> >   _nvox;    
    std::vector<std::vector<double> >         _mdiff;   
    std::vector<std::vector<double> >         _msqrd;   
    StatsInfo(unsigned int nsl, unsigned int nscan) EddyTry : _nvox(nsl), _mdiff(nsl), _msqrd(nsl) { 
      for (unsigned int i=0; i<nsl; i++) { _nvox[i].resize(nscan,0); _mdiff[i].resize(nscan,0.0); _msqrd[i].resize(nscan,0.0); }
    } EddyCatch
  };
private:
  OutlierDefinition                 _old;     
  unsigned int                      _etc;     
  OLType                            _olt;     
  MultiBandGroups                   _mbg;     
  StatsInfo                         _sws;     
  StatsInfo                         _gws;     
  OutlierInfo                       _swo;     
  OutlierInfo                       _gwo;     

  void throw_if_oor(unsigned int scan) const EddyTry { if (scan >= this->NScan()) throw EddyException("ReplacementManager::throw_if_oor: Scan index out of range"); } EddyCatch
  double sqr(double a) const EddyTry { return(a*a); } EddyCatch
  std::pair<double,double> mean_and_std(const EDDY::ReplacementManager::StatsInfo& sws, unsigned int minvox, unsigned int etc, 
					const std::vector<std::vector<bool> >& ovv, std::pair<double,double>& stdev) const;
};

 
class ImageCoordinates
{
public:
  ImageCoordinates(const NEWIMAGE::volume<float>& ima) 
  EddyTry : ImageCoordinates(static_cast<unsigned int>(ima.xsize()),static_cast<unsigned int>(ima.ysize()),static_cast<unsigned int>(ima.zsize())) {} EddyCatch
  ImageCoordinates(unsigned int xn, unsigned int yn, unsigned int zn) 
  EddyTry : _xn(xn), _yn(yn), _zn(zn)
  {
    _x = new float[_xn*_yn*_zn];
    _y = new float[_xn*_yn*_zn];
    _z = new float[_xn*_yn*_zn];
    for (unsigned int k=0, indx=0; k<_zn; k++) {
      for (unsigned int j=0; j<_yn; j++) {
	for (unsigned int i=0; i<_xn; i++) {
	  _x[indx] = float(i);
	  _y[indx] = float(j);
	  _z[indx++] = float(k);
	}
      }
    }
  } EddyCatch
  ImageCoordinates(const ImageCoordinates& inp) 
  EddyTry : _xn(inp._xn), _yn(inp._yn), _zn(inp._zn) 
  {
    _x = new float[_xn*_yn*_zn]; std::memcpy(_x,inp._x,_xn*_yn*_zn*sizeof(float));
    _y = new float[_xn*_yn*_zn]; std::memcpy(_y,inp._y,_xn*_yn*_zn*sizeof(float));
    _z = new float[_xn*_yn*_zn]; std::memcpy(_z,inp._z,_xn*_yn*_zn*sizeof(float));   
  } EddyCatch
  ImageCoordinates(ImageCoordinates&& inp)
  EddyTry : _xn(inp._xn), _yn(inp._yn), _zn(inp._zn)
  {
    _x = inp._x; inp._x = nullptr;
    _y = inp._y; inp._y = nullptr;
    _z = inp._z; inp._z = nullptr;
  } EddyCatch
  ~ImageCoordinates() EddyTry { delete[] _x; delete[] _y; delete[] _z; } EddyCatch
  ImageCoordinates& operator=(const ImageCoordinates& rhs) EddyTry {
    if (this == &rhs) return(*this);
    delete[] _x; delete[] _y; delete[] _z;
    _xn = rhs._xn; _yn = rhs._yn; _zn = rhs._zn; 
    _x = new float[_xn*_yn*_zn]; std::memcpy(_x,rhs._x,_xn*_yn*_zn*sizeof(float));
    _y = new float[_xn*_yn*_zn]; std::memcpy(_y,rhs._y,_xn*_yn*_zn*sizeof(float));
    _z = new float[_xn*_yn*_zn]; std::memcpy(_z,rhs._z,_xn*_yn*_zn*sizeof(float));   
    return(*this);
  } EddyCatch
  ImageCoordinates& operator=(ImageCoordinates&& rhs) EddyTry {
    if (this != &rhs) {
      delete[] _x; delete[] _y; delete[] _z;
      _xn = rhs._xn; _yn = rhs._yn; _zn = rhs._zn; 
      _x = rhs._x; rhs._x = nullptr;
      _y = rhs._y; rhs._y = nullptr;
      _z = rhs._z; rhs._z = nullptr;
    }
    return(*this);
  } EddyCatch
  ImageCoordinates& operator+=(const ImageCoordinates& rhs) EddyTry {
    if (_xn != rhs._xn || _yn != rhs._yn || _zn != rhs._zn) throw EddyException("ImageCoordinates::operator-= size mismatch");
    for (unsigned int i=0; i<_xn*_yn*_zn; i++) { _x[i]+=rhs._x[i]; _y[i]+=rhs._y[i]; _z[i]+=rhs._z[i]; }
    return(*this);
  } EddyCatch
  ImageCoordinates& operator-=(const ImageCoordinates& rhs) EddyTry {
    if (_xn != rhs._xn || _yn != rhs._yn || _zn != rhs._zn) throw EddyException("ImageCoordinates::operator-= size mismatch");
    for (unsigned int i=0; i<_xn*_yn*_zn; i++) { _x[i]-=rhs._x[i]; _y[i]-=rhs._y[i]; _z[i]-=rhs._z[i]; }
    return(*this);
  } EddyCatch
  ImageCoordinates operator+(const ImageCoordinates& rhs) const EddyTry {
    return(ImageCoordinates(*this)+=rhs);
  } EddyCatch
  ImageCoordinates operator-(const ImageCoordinates& rhs) const EddyTry {
    return(ImageCoordinates(*this)-=rhs);
  } EddyCatch
  ImageCoordinates& operator/=(double div) EddyTry {
    if (div==0) throw EddyException("ImageCoordinates::operator/= attempt to divide by zero");
    for (unsigned int i=0; i<_xn*_yn*_zn; i++) { _x[i]/=div; _y[i]/=div; _z[i]/=div; }
    return(*this);
  } EddyCatch
  NEWIMAGE::volume<float> operator*(const NEWIMAGE::volume4D<float>& vol) EddyTry {
    if (int(_xn) != vol.xsize() || int(_yn) != vol.ysize() || int(_zn) != vol.zsize() || vol.tsize() != 3) {
      throw EddyException("ImageCoordinates::operator* size mismatch");
    }
    NEWIMAGE::volume<float> ovol = vol[0];
    for (unsigned int k=0, indx=0; k<_zn; k++) {
      for (unsigned int j=0; j<_yn; j++) {
	for (unsigned int i=0; i<_xn; i++) {
          ovol(i,j,k) = _x[indx]*vol(i,j,k,0) + _y[indx]*vol(i,j,k,1) + _z[indx]*vol(i,j,k,2);
          indx++;
	}
      }
    }
    return(ovol);
  } EddyCatch
  void Transform(const NEWMAT::Matrix& M) EddyTry {
    if (M.Nrows() != 4 || M.Ncols() != 4) throw EddyException("ImageCoordinates::Transform: Matrix M must be 4x4");
    float M11 = M(1,1); float M12 = M(1,2); float M13 = M(1,3); float M14 = M(1,4); 
    float M21 = M(2,1); float M22 = M(2,2); float M23 = M(2,3); float M24 = M(2,4); 
    float M31 = M(3,1); float M32 = M(3,2); float M33 = M(3,3); float M34 = M(3,4); 
    float *xp = _x; float *yp = _y; float *zp = _z;
    for (unsigned int i=0; i<N(); i++) {
      float ox = M11 * *xp + M12 * *yp + M13 * *zp + M14;
      float oy = M21 * *xp + M22 * *yp + M23 * *zp + M24;
      float oz = M31 * *xp + M32 * *yp + M33 * *zp + M34;
      *xp = ox; *yp = oy; *zp = oz;
      xp++; yp++; zp++;
    }
  } EddyCatch
  void Transform(const std::vector<NEWMAT::Matrix>&             M,    
		 const std::vector<std::vector<unsigned int> >& grps) 
  EddyTry {
    if (M.size() != grps.size()) throw EddyException("ImageCoordinates::Transform: Mismatch between M and grps");
    for (unsigned int grp=0; grp<grps.size(); grp++) {
      if (M[grp].Nrows() != 4 || M[grp].Ncols() != 4) throw EddyException("ImageCoordinates::Transform: All Matrices M must be 4x4");
      std::vector<unsigned int> slices = grps[grp];
      float M11 = M[grp](1,1); float M12 = M[grp](1,2); float M13 = M[grp](1,3); float M14 = M[grp](1,4); 
      float M21 = M[grp](2,1); float M22 = M[grp](2,2); float M23 = M[grp](2,3); float M24 = M[grp](2,4); 
      float M31 = M[grp](3,1); float M32 = M[grp](3,2); float M33 = M[grp](3,3); float M34 = M[grp](3,4); 
      for (unsigned int i=0; i<slices.size(); i++) {
	for (unsigned int indx=slstart(slices[i]); indx<slend(slices[i]); indx++) {
	  float ox = M11 * _x[indx] + M12 * _y[indx] + M13 * _z[indx] + M14;
	  float oy = M21 * _x[indx] + M22 * _y[indx] + M23 * _z[indx] + M24;
	  float oz = M31 * _x[indx] + M32 * _y[indx] + M33 * _z[indx] + M34;
	  _x[indx] = ox; _y[indx] = oy; _z[indx] = oz;
	}
      }
    }
  } EddyCatch
  ImageCoordinates MakeTransformed(const NEWMAT::Matrix& M) const EddyTry {
    ImageCoordinates rval = *this;
    rval.Transform(M);
    return(rval);
  } EddyCatch
  ImageCoordinates MakeTransformed(const std::vector<NEWMAT::Matrix>&             M,
				   const std::vector<std::vector<unsigned int> >& grps) const EddyTry {
    ImageCoordinates rval = *this;
    rval.Transform(M,grps);
    return(rval);    
  } EddyCatch
  void Write(const std::string& fname) const EddyTry
  {
    NEWMAT::Matrix omat(N(),3);
    for (unsigned int i=0; i<N(); i++) {
      omat(i+1,1) = x(i); omat(i+1,2) = y(i); omat(i+1,3) = z(i); 
    }
    MISCMATHS::write_ascii_matrix(fname,omat);
  } EddyCatch

  unsigned int N() const EddyTry { return(_xn*_yn*_zn); } EddyCatch
  unsigned int NX() const EddyTry { return(_xn); } EddyCatch
  unsigned int NY() const EddyTry { return(_yn); } EddyCatch
  unsigned int NZ() const EddyTry { return(_zn); } EddyCatch
  bool IsInBounds(unsigned int i) const EddyTry { return(_x[i] >= 0 && _x[i] <= (_xn-1) && _y[i] >= 0 && _y[i] <= (_yn-1) && _z[i] >= 0 && _z[i] <= (_zn-1)); }
EddyCatch  
  const float& x(unsigned int i) const EddyTry { return(_x[i]); } EddyCatch
  const float& y(unsigned int i) const EddyTry { return(_y[i]); } EddyCatch
  const float& z(unsigned int i) const EddyTry { return(_z[i]); } EddyCatch
  float& x(unsigned int i) EddyTry { return(_x[i]); } EddyCatch
  float& y(unsigned int i) EddyTry { return(_y[i]); } EddyCatch
  float& z(unsigned int i) EddyTry { return(_z[i]); } EddyCatch
   
private:
  unsigned int _xn;
  unsigned int _yn;
  unsigned int _zn;
  float *_x;
  float *_y;
  float *_z;
  unsigned int slstart(unsigned int sl) const EddyTry { return(sl*_xn*_yn); } EddyCatch
  unsigned int slend(unsigned int sl) const EddyTry { return((sl+1)*_xn*_yn); } EddyCatch
};

 
class MutualInfoHelper
{
public:
  MutualInfoHelper(unsigned int nbins) EddyTry : _nbins(nbins), _lset(false) { 
    _mhist1 = new double[_nbins];
    _mhist2 = new double[_nbins];
    _jhist = new double[_nbins*_nbins];
  } EddyCatch
  MutualInfoHelper(unsigned int nbins, float min1, float max1, float min2, float max2) EddyTry
    : _nbins(nbins), _min1(min1), _max1(max1), _min2(min2), _max2(max2), _lset(true) { 
    _mhist1 = new double[_nbins];
    _mhist2 = new double[_nbins];
    _jhist = new double[_nbins*_nbins];
  } EddyCatch
  virtual ~MutualInfoHelper() EddyTry { delete[] _mhist1; delete[] _mhist2; delete[] _jhist; } EddyCatch
  void SetLimits(float min1, float max1, float min2, float max2) EddyTry {
    _min1 = min1; _max1 = max1; _min2 = min2; _max2 = max2; _lset = true;
  } EddyCatch
  double MI(const NEWIMAGE::volume<float>& ima1,
	    const NEWIMAGE::volume<float>& ima2,
	    const NEWIMAGE::volume<float>& mask) const;
  double SoftMI(const NEWIMAGE::volume<float>& ima1,
		const NEWIMAGE::volume<float>& ima2,
		const NEWIMAGE::volume<float>& mask) const;
private:
  double plogp(double p) const EddyTry { if (p) return( - p*std::log(p)); else return(0.0); } EddyCatch
  unsigned int val_to_indx(float val, float min, float max, unsigned int nbins) const EddyTry
  {
    int tmp = static_cast<int>((val-min)*static_cast<float>(nbins-1)/(max-min) + 0.5);
    if (tmp < 0) tmp = 0;
    else if (static_cast<unsigned int>(tmp) > (nbins-1)) tmp = nbins-1;
    return(static_cast<unsigned int>(tmp));
  } EddyCatch
  unsigned int val_to_floor_indx(float val, float min, float max, unsigned int nbins, float *rem) const EddyTry
  {
    unsigned int rval=0;
    float x = (val-min)*static_cast<float>(nbins)/(max-min); 
    if (x <= 0.5) { *rem = 0.0; rval = 0; }
    else if (x >= static_cast<float>(nbins-0.5)) { *rem = 0.0; rval = nbins - 1; }
    else { rval = static_cast<unsigned int>(x-0.5); *rem = x - 0.5 - static_cast<float>(rval); }
    return(rval);
  } EddyCatch

  unsigned int          _nbins;   
  float                 _min1;    
  float                 _max1;    
  float                 _min2;    
  float                 _max2;    
  bool                  _lset;    
  mutable double        *_mhist1; 
  mutable double        *_mhist2; 
  mutable double        *_jhist;  
};

} 

#endif 









 




