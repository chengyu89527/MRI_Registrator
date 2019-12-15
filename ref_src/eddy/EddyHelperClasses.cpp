// Definitions of classes that implements useful
// concepts for the eddy current project.
// 
// EddyHelperClasses.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford 
//

#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"

using namespace EDDY;

NEWMAT::Matrix JsonReader::SliceOrdering() const EddyTry
{
  std::vector<double> slice_times;
  
  size_t spos = _content.find("SliceTiming");
  std::string st_cont;
  if (spos != std::string::npos) {
    st_cont = _content.substr(spos);
    size_t spos2 = st_cont.find("[");
    st_cont = st_cont.substr(spos2+1);
    while (!isdigit(st_cont[0])) st_cont = st_cont.substr(1);
    spos2 = st_cont.find("]");
    st_cont = st_cont.substr(0,spos2);
    while (!isdigit(st_cont.back())) st_cont.pop_back();
  } 
  else throw EddyException("JsonReader::SliceOrdering: Failed to extract string \"SliceTiming\"");

  
  std::vector<double> times;
  if (st_cont.size()) {
    std::istringstream ss(st_cont);
    std::string line;
    while (!ss.eof()) { 
      std::getline(ss,line);
      std::istringstream ls(line);
      std::string elem;
      while (!ls.eof()) { 
	std::getline(ls,elem,',');
	while (elem.size() && !isdigit(elem.back())) elem.pop_back();
        if (elem.size()) {
	  times.push_back(atof(elem.c_str()));
	}
      }
    }
  }
  else throw EddyException("JsonReader::SliceOrdering: Failed to parse acquisition times");

  
  std::vector<std::pair<unsigned int,double> > indextime(times.size());
  for (unsigned int i=0; i<times.size(); i++) { indextime[i] = std::make_pair(i,times[i]); }
  std::sort(indextime.begin(),indextime.end(),[](std::pair<unsigned int,double> p1, std::pair<unsigned int,double> p2) 
	    { return((p1.second == p2.second) ? (p1.first < p2.first) : (p1.second < p2.second)); });

  
  unsigned int mbf = 1;
  for (unsigned int i=1; i<indextime.size(); i++) { 
    if (indextime[i].second > indextime[i-1].second) break;
    else mbf++;
  }
  double ctime = indextime[0].second;
  unsigned int cnt = 1;
  for (unsigned int i=1; i<indextime.size(); i++) {
    if (indextime[i].second == ctime) cnt++;
    else {
      if (cnt != mbf) throw EddyException("JsonReader::SliceOrdering: Inconsistent MB groups");
      else {
	ctime = indextime[i].second;
	cnt = 1;
      }
    }
  }
  
  
  NEWMAT::Matrix M(indextime.size()/mbf,mbf);
  unsigned int ii=0;
  for (int ri=0; ri<M.Nrows(); ri++) {
    for (int ci=0; ci<M.Ncols(); ci++) M(ri+1,ci+1) = std::round(static_cast<double>(indextime[ii++].first));
  }  

  return(M);
} EddyCatch

NEWMAT::ColumnVector JsonReader::PEVector() const EddyTry
{
  NEWMAT::ColumnVector pe(3); pe = 0;
  
  size_t spos = _content.find("\"PhaseEncodingDirection\":");
  std::string st_cont;
  if (spos != std::string::npos) {
    st_cont = _content.substr(spos);
    size_t spos2 = st_cont.find(":");
    st_cont = st_cont.substr(spos2+1);
    spos2 = st_cont.find("\"");
    st_cont = st_cont.substr(spos2+1);
    cout << "st_cont[0] = " << st_cont[0] << endl;
    if (st_cont[0] == 'i') {
      if (st_cont[1] == '-') pe(1) = -1;
      else pe(1) = 1;
    }
    else if (st_cont[0] == 'j') {
      if (st_cont[1] == '-') pe(2) = -1;
      else pe(2) = 1;
    }
    else throw EddyException("JsonReader::SliceOrdering: Failed to decode \"PhaseEncodingDirection\"");
  } 
  else throw EddyException("JsonReader::SliceOrdering: Failed to extract string \"PhaseEncodingDirection\"");

  return(pe);
} EddyCatch

void JsonReader::common_read() EddyTry
{
  std::ifstream infile;
  std::stringstream buffer;
  infile.open(_fname);
  if (infile.is_open()) {
    buffer << infile.rdbuf();
    infile.close();
  }
  else throw EddyException("JsonReader::common_read(): Could not open file " + _fname);
  _content = buffer.str();
} EddyCatch










bool DiffPara::operator==(const DiffPara& rhs) const EddyTry
{ 
  return(EddyUtils::AreInSameShell(*this,rhs) && EddyUtils::HaveSameDirection(*this,rhs)); 
} EddyCatch










AcqPara::AcqPara(const NEWMAT::ColumnVector&   pevec,
                 double                        rotime) EddyTry
: _pevec(pevec), _rotime(rotime)
{
  if (pevec.Nrows() != 3) throw EddyException("AcqPara::AcqPara: Wrong dimension pe-vector");
  if (rotime < 0.01 || rotime > 0.2) throw EddyException("AcqPara::AcqPara: Unrealistic read-out time");
  int cc = 0; 
  for (int i=0; i<3; i++) { if (fabs(pevec(i+1)) > 1e-6) cc++; }
  if (!cc) throw EddyException("AcqPara::AcqPara: Zero Phase-encode vector");
  if (cc > 1) throw EddyException("AcqPara::AcqPara: Oblique pe-vectors not yet implemented");
} EddyCatch

bool AcqPara::operator==(const AcqPara& rh) const EddyTry
{
  if (fabs(this->_rotime-rh._rotime) > 1e-6) return(false);
  for (int i=0; i<3; i++) {
    if (fabs(this->_pevec(i+1)-rh._pevec(i+1)) > 1e-6) return(false);
  }
  return(true);
} EddyCatch

std::vector<unsigned int> AcqPara::BinarisedPhaseEncodeVector() const EddyTry
{
  std::vector<unsigned int> rval(3,0);
  for (unsigned int i=0; i<3; i++) rval[i] = (_pevec(i+1) == 0) ? 0 : 1;
  return(rval);
} EddyCatch

DiffStats::DiffStats(const NEWIMAGE::volume<float>& diff, const NEWIMAGE::volume<float>& mask) EddyTry
: _md(diff.zsize(),0.0), _msd(diff.zsize(),0.0), _n(mask.zsize(),0)
{
#pragma omp parallel for 
  for (int k=0; k<diff.zsize(); k++) {
    for (int j=0; j<diff.ysize(); j++) {
      for (int i=0; i<diff.xsize(); i++) {
	if (mask(i,j,k)) {
          _md[k] += diff(i,j,k);
          _msd[k] += diff(i,j,k)*diff(i,j,k);
	  _n[k] += 1;
	}
      }
    }
    if (_n[k]) { _md[k] /= double(_n[k]); _msd[k] /= double(_n[k]); }
  }  
} EddyCatch

MultiBandGroups::MultiBandGroups(unsigned int nsl, 
				 unsigned int mb,
				 int          offs) EddyTry
: _nsl(nsl), _mb(mb), _offs(offs)
{
  if (std::abs(_offs) > 1) throw EddyException("MultiBandGroups::MultiBandGroups: offs out of range");
  if (((int(_nsl)+std::abs(_offs)) % int(_mb)) || (_mb==1 && _offs!=0)) throw EddyException("MultiBandGroups::MultiBandGroups: Incompatible nsl, mb and offs");
  unsigned int ngrp = (_nsl+static_cast<unsigned int>(std::abs(_offs))) / _mb;
  _grps.resize(ngrp);
  for (unsigned int grp=0; grp<ngrp; grp++) {
    int sindx = (offs == 1) ? -1 : 0;
    for (int i=sindx+grp; i<int(_nsl); i+=ngrp) {
      if (i >= 0 && i < int(_nsl)) _grps[grp].push_back(static_cast<unsigned int>(i));
    }
  }
  _to.resize(ngrp); for (unsigned int grp=0; grp<ngrp; grp++) _to[grp]=grp; 
} EddyCatch

MultiBandGroups::MultiBandGroups(const std::string& fname) EddyTry
{
  
  std::string line;
  std::ifstream ifs(fname.c_str());
  if (ifs.is_open()) {
    while (std::getline(ifs,line)) {
      std::vector<unsigned int> tmp_vec; 
      unsigned int tmp_ui;
      std::stringstream ss(line);
      while (ss >> tmp_ui) tmp_vec.push_back(tmp_ui);
      _grps.push_back(tmp_vec); 
    }
    if (ifs.eof()) ifs.close();
    else throw EddyException("MultiBandGroups::MultiBandGroups: Problem reading file");
  }
  else throw EddyException("MultiBandGroups::MultiBandGroups: Unable to open file");
  assert_grps(); 
} EddyCatch

MultiBandGroups::MultiBandGroups(const NEWMAT::Matrix& slices) EddyTry
{
  _grps.resize(slices.Nrows());
  for (unsigned int i=0; i<_grps.size(); i++) {
    _grps[i].resize(slices.Ncols());
    for (unsigned int j=0; j<_grps[i].size(); j++) _grps[i][j] = static_cast<unsigned int>(std::round(slices(i+1,j+1)));
  }
  assert_grps(); 
} EddyCatch

void MultiBandGroups::assert_grps() EddyTry
{
  
  _nsl = 0;
  _mb = 0;
  for (unsigned int grp=0; grp<_grps.size(); grp++) {
    _mb = std::max(static_cast<unsigned int>(_grps.size()),_mb);
    for (unsigned int sl=0; sl<_grps[grp].size(); sl++) {
      _nsl = std::max(_grps[grp][sl],_nsl);
    }
  }
  _nsl++;
  _offs = 0; 
  
  std::vector<unsigned int> check_vec(_nsl,0);
  for (unsigned int grp=0; grp<_grps.size(); grp++) {
    for (unsigned int sl=0; sl<_grps[grp].size(); sl++) {
      check_vec[_grps[grp][sl]] += 1;
    }
  }
  for (unsigned int i=0; i<check_vec.size(); i++) {
    if (check_vec[i] != 1) throw EddyException("MultiBandGroups::MultiBandGroups: Logical error in file");
  }
  
  _to.resize(_grps.size());
  for (unsigned int i=0; i<_grps.size(); i++) _to[i] = i;
} EddyCatch


void DiffStatsVector::Write(const std::string& bfname) const EddyTry
{
  std::string fname = bfname + std::string(".MeanDifference");
  std::ofstream file;
  file.open(fname.c_str(),ios::out|ios::trunc);
  for (unsigned int i=0; i<_n; i++) file << _ds[i].MeanDifferenceVector() << endl;
  file.close();

  fname = bfname + std::string(".MeanSquaredDifference");
  file.open(fname.c_str(),ios::out|ios::trunc);
  for (unsigned int i=0; i<_n; i++) file << _ds[i].MeanSqrDiffVector() << endl;
  file.close();

  fname = bfname + std::string(".NoOfVoxels");
  file.open(fname.c_str(),ios::out|ios::trunc);
  for (unsigned int i=0; i<_n; i++) file << _ds[i].NVoxVector() << endl;
  file.close();
} EddyCatch


void ReplacementManager::Update(const DiffStatsVector& dsv) EddyTry
{
  if (dsv.NSlice() != this->NSlice() || dsv.NScan() != this->NScan()) {
    throw EddyException("ReplacementManager::Update: Mismatched DiffStatsVector object");
  }
  
  for (unsigned int scan=0; scan<NScan(); scan++) {
    for (unsigned int sl=0; sl<NSlice(); sl++) { 
      _sws._nvox[sl][scan] = dsv.NVox(scan,sl); 
      _sws._mdiff[sl][scan] = dsv.MeanDiff(scan,sl); 
      _sws._msqrd[sl][scan] = dsv.MeanSqrDiff(scan,sl); 
    }
  }
  
  for (unsigned int scan=0; scan<NScan(); scan++) {
    for (unsigned int grp=0; grp<NGroup(); grp++) {
      std::vector<unsigned int> sl_in_grp = _mbg.SlicesInGroup(grp);
      _gws._nvox[grp][scan] = 0; _gws._mdiff[grp][scan] = 0.0;
      for (unsigned int i=0; i<sl_in_grp.size(); i++) { 
	_gws._nvox[grp][scan] += _sws._nvox[sl_in_grp[i]][scan]; 
	_gws._mdiff[grp][scan] += _sws._nvox[sl_in_grp[i]][scan] * _sws._mdiff[sl_in_grp[i]][scan]; 
	_gws._msqrd[grp][scan] += _sws._nvox[sl_in_grp[i]][scan] * _sws._msqrd[sl_in_grp[i]][scan]; 
      }
      _gws._mdiff[grp][scan] /= static_cast<double>(_gws._nvox[grp][scan]);
      _gws._msqrd[grp][scan] /= static_cast<double>(_gws._nvox[grp][scan]);
    }
  }
  
  std::pair<double,double> sstd(0.0,0.0);     
  std::pair<double,double> gstd(0.0,0.0);     
  std::pair<double,double> smd = mean_and_std(_sws,_old.MinVoxels(),_etc,_swo._ovv,sstd); 
  std::pair<double,double> gmd = mean_and_std(_gws,_old.MinVoxels(),_etc,_gwo._ovv,gstd); 
  
  
  for (unsigned int scan=0; scan<NScan(); scan++) {
    for (unsigned int grp=0; grp<NGroup(); grp++) {
      if (_gws._nvox[grp][scan] >= _old.MinVoxels()) { 
	double sf = (_etc == 1) ? 1.0 / std::sqrt(double(_gws._nvox[grp][scan])) : 1.0;
	_gwo._nsv[grp][scan] = (_gws._mdiff[grp][scan] - gmd.first) / (sf * gstd.first);
	_gwo._nsq[grp][scan] = (_gws._msqrd[grp][scan] - gmd.second) / (sf * gstd.second);
	_gwo._ovv[grp][scan] = (-_gwo._nsv[grp][scan] > _old.NStdDev()) ? true : false;
	if (_old.ConsiderPosOL()) _gwo._ovv[grp][scan] = (_gwo._nsv[grp][scan] > _old.NStdDev()) ? true : _gwo._ovv[grp][scan];
	if (_old.ConsiderSqrOL()) _gwo._ovv[grp][scan] = (_gwo._nsq[grp][scan] > _old.NStdDev()) ? true : _gwo._ovv[grp][scan];
      }
    }
  }
  
  if (_olt==EDDY::GroupWise || _olt==EDDY::Both) { 
    
    for (unsigned int grp=0; grp<NGroup(); grp++) {
      std::vector<unsigned int> sl_in_grp = _mbg.SlicesInGroup(grp);
      for (unsigned int scan=0; scan<NScan(); scan++) {
	for (unsigned int i=0; i<sl_in_grp.size(); i++) {
	  _swo._nsv[sl_in_grp[i]][scan] = _gwo._nsv[grp][scan];
	  _swo._nsq[sl_in_grp[i]][scan] = _gwo._nsq[grp][scan];
	  _swo._ovv[sl_in_grp[i]][scan] = _gwo._ovv[grp][scan];
	}
      }
    }
    if (_olt==EDDY::Both) { 
      
      for (unsigned int grp=0; grp<NGroup(); grp++) {
	std::vector<unsigned int> sl_in_grp = _mbg.SlicesInGroup(grp);
	for (unsigned int scan=0; scan<NScan(); scan++) {
	  bool group_kosher = true;
	  for (unsigned int i=0; i<sl_in_grp.size(); i++) {
	    if (_sws._nvox[sl_in_grp[i]][scan] >= _old.MinVoxels()) { 
	      double sf = (_etc == 1) ? 1.0 / std::sqrt(double(_sws._nvox[sl_in_grp[i]][scan])) : 1.0;	      
	      double tmp_ns = (_sws._mdiff[sl_in_grp[i]][scan] - smd.first) / (sf * sstd.first);
	      group_kosher = (-tmp_ns > (_old.NStdDev()-1.0)) ? true : false;
	      if (_old.ConsiderPosOL()) group_kosher = (tmp_ns > (_old.NStdDev()-1.0) && _gwo._nsv[grp][scan] > 0.0) ? true : group_kosher;
	      if (_old.ConsiderSqrOL()) {
		tmp_ns = (_sws._msqrd[sl_in_grp[i]][scan] - smd.second) / (sf * sstd.second);
		group_kosher = (tmp_ns > (_old.NStdDev()-1.0)) ? true : group_kosher;
	      }
	    }
	    if (!group_kosher) break;
	  }
	  if (!group_kosher) { 
	    for (unsigned int i=0; i<sl_in_grp.size(); i++) {
	      _swo._nsv[sl_in_grp[i]][scan] = 0.0;    
	      _swo._nsq[sl_in_grp[i]][scan] = 0.0;    
	      _swo._ovv[sl_in_grp[i]][scan] = false;
	    }
	  }
	}
      }
      
      for (unsigned int scan=0; scan<NScan(); scan++) {
	for (unsigned int sl=0; sl<NSlice(); sl++) {
	  if (_sws._nvox[sl][scan] >= _old.MinVoxels()) { 
	    double sf = (_etc == 1) ? 1.0 / std::sqrt(double(_sws._nvox[sl][scan])) : 1.0;
	    double tmp_ns = (_sws._mdiff[sl][scan] - smd.first) / (sf * sstd.first);
	    _swo._nsv[sl][scan] = (std::abs(tmp_ns) > std::abs(_swo._nsv[sl][scan])) ? tmp_ns : _swo._nsv[sl][scan];
	    tmp_ns = (_sws._msqrd[sl][scan] - smd.second) / (sf * sstd.second);
	    _swo._nsq[sl][scan] = (std::abs(tmp_ns) > std::abs(_swo._nsq[sl][scan])) ? tmp_ns : _swo._nsq[sl][scan];
	    _swo._ovv[sl][scan] = (-_swo._nsv[sl][scan] > _old.NStdDev()) ? true : false;
	    if (_old.ConsiderPosOL()) _swo._ovv[sl][scan] = (_swo._nsv[sl][scan] > _old.NStdDev()) ? true : _swo._ovv[sl][scan];
	    if (_old.ConsiderSqrOL()) _swo._ovv[sl][scan] = (_swo._nsq[sl][scan] > _old.NStdDev()) ? true : _swo._ovv[sl][scan];
	  }
	}
      }
    }
  }
  else { 
    for (unsigned int scan=0; scan<NScan(); scan++) {
      for (unsigned int sl=0; sl<NSlice(); sl++) {
	if (_sws._nvox[sl][scan] >= _old.MinVoxels()) { 
	  double sf = (_etc == 1) ? 1.0 / std::sqrt(double(_sws._nvox[sl][scan])) : 1.0;
	  _swo._nsv[sl][scan] = (_sws._mdiff[sl][scan] - smd.first) / (sf * sstd.first);
	  _swo._nsq[sl][scan] = (_sws._msqrd[sl][scan] - smd.second) / (sf * sstd.second);
	  _swo._ovv[sl][scan] = (-_swo._nsv[sl][scan] > _old.NStdDev()) ? true : false;
	  if (_old.ConsiderPosOL()) _swo._ovv[sl][scan] = (_swo._nsv[sl][scan] > _old.NStdDev()) ? true : _swo._ovv[sl][scan];
	  if (_old.ConsiderSqrOL()) _swo._ovv[sl][scan] = (_swo._nsq[sl][scan] > _old.NStdDev()) ? true : _swo._ovv[sl][scan];
	}
      }
    }
  }
} EddyCatch

std::vector<unsigned int> ReplacementManager::OutliersInScan(unsigned int scan) const EddyTry
{
  throw_if_oor(scan);
  std::vector<unsigned int> ol;
  for (unsigned int sl=0; sl<NSlice(); sl++) if (_swo._ovv[sl][scan]) ol.push_back(sl);
  return(ol);
} EddyCatch

bool ReplacementManager::ScanHasOutliers(unsigned int scan) const EddyTry
{
  throw_if_oor(scan);
  for (unsigned int sl=0; sl<NSlice(); sl++) if (_swo._ovv[sl][scan]) return(true);
  return(false);
} EddyCatch

void ReplacementManager::WriteReport(const std::vector<unsigned int>& i2i,
				     const std::string&               fname) const EddyTry
{
  std::ofstream fout;
  fout.open(fname.c_str(),ios::out|ios::trunc);
  if (fout.fail()) {
    cout << "Failed to open outlier report file " << fname << endl;
    return;
  }
  for (unsigned int sl=0; sl<NSlice(); sl++) {
    for (unsigned int s=0; s<NScan(); s++) {
      if (_swo._ovv[sl][s]) {
	fout << "Slice " << sl << " in scan " << i2i[s] << " is an outlier with mean " << _swo._nsv[sl][s] << " standard deviations off, and mean squared " << _swo._nsq[sl][s] << " standard deviations off." << endl;
      }
    }
  }
  fout.close();
  return;
} EddyCatch

void ReplacementManager::WriteMatrixReport(const std::vector<unsigned int>& i2i,
					   unsigned int                     nscan,
					   const std::string&               om_fname,
					   const std::string&               nstdev_fname,
					   const std::string&               n_sqr_stdev_fname) const EddyTry
{
  std::ofstream fout;
  if (!om_fname.empty()) {
    fout.open(om_fname.c_str(),ios::out|ios::trunc);
    fout << "One row per scan, one column per slice. Outlier: 1, Non-outlier: 0" << endl;
    
    NEWMAT::Matrix rpovv(nscan,NSlice()); rpovv = 0.0;
    for (unsigned int scan=0; scan<NScan(); scan++) {
      for (unsigned int slice=0; slice<NSlice(); slice++) {
	if (_swo._ovv[slice][scan]) rpovv(i2i[scan]+1,slice+1) = 1.0;
      }
    }
    
    for (unsigned int scan=0; scan<nscan; scan++) {
      for (unsigned int slice=0; slice<NSlice(); slice++) {
	if (rpovv(scan+1,slice+1) > 0) fout << "1 ";
	else fout << "0 ";
      }
      fout << endl;
    }
    fout.close();
  }
  
  if (!nstdev_fname.empty()) {
    fout.open(nstdev_fname.c_str(),ios::out|ios::trunc);
    fout << "One row per scan, one column per slice. b0s set to zero" << endl;
    
    NEWMAT::Matrix rpnsv(nscan,NSlice()); rpnsv = 0.0;
    for (unsigned int scan=0; scan<NScan(); scan++) {
      for (unsigned int slice=0; slice<NSlice(); slice++) {
	rpnsv(i2i[scan]+1,slice+1) = _swo._nsv[slice][scan];
      }
    }
    
    for (unsigned int scan=0; scan<nscan; scan++) {
      for (unsigned int slice=0; slice<NSlice(); slice++) {
	double tmp = rpnsv(scan+1,slice+1);
	fout << tmp << " ";
      }
      fout << endl;
    }
    fout.close();
  }
  
  if (!n_sqr_stdev_fname.empty()) {
    fout.open(n_sqr_stdev_fname.c_str(),ios::out|ios::trunc);
    fout << "One row per scan, one column per slice. b0s set to zero" << endl;
    
    NEWMAT::Matrix rpnsv(nscan,NSlice()); rpnsv = 0.0;
    for (unsigned int scan=0; scan<NScan(); scan++) {
      for (unsigned int slice=0; slice<NSlice(); slice++) {
	rpnsv(i2i[scan]+1,slice+1) = _swo._nsq[slice][scan];
      }
    }
    
    for (unsigned int scan=0; scan<nscan; scan++) {
      for (unsigned int slice=0; slice<NSlice(); slice++) {
	double tmp = rpnsv(scan+1,slice+1);
	fout << tmp << " ";
      }
      fout << endl;
    }
    fout.close();
  }
} EddyCatch

void ReplacementManager::DumpOutlierMaps(const std::string& bfname) const EddyTry
{
  std::string fname = bfname + ".SliceWiseOutlierMap";
  std::ofstream fout;
  fout.open(fname.c_str(),ios::out|ios::trunc);
  fout << "One row per scan, one column per slice. Outlier: 1, Non-outlier: 0" << endl;
  for (unsigned scan=0; scan<NScan(); scan++) {
    for (unsigned int slice=0; slice<NSlice(); slice++) {
      fout << _swo._ovv[slice][scan] << " ";
    }
    fout << endl;
  } 
  fout.close();

  fname = bfname + ".SliceWiseNoOfStdevMap";
  fout.open(fname.c_str(),ios::out|ios::trunc);
  fout << "One row per scan, one column per slice." << endl;
  for (unsigned scan=0; scan<NScan(); scan++) {
    for (unsigned int slice=0; slice<NSlice(); slice++) {
      fout << _gwo._nsv[slice][scan] << " ";
    }
    fout << endl;
  } 
  fout.close();

  fname = bfname + ".GroupWiseOutlierMap";
  fout.open(fname.c_str(),ios::out|ios::trunc);
  fout << "One row per scan, one column per slice. Outlier: 1, Non-outlier: 0" << endl;
  for (unsigned scan=0; scan<NScan(); scan++) {
    for (unsigned int slice=0; slice<NSlice(); slice++) {
      fout << _gwo._ovv[slice][scan] << " ";
    }
    fout << endl;
  } 
  fout.close();

  fname = bfname + ".GroupWiseNoOfStdevMap";
  fout.open(fname.c_str(),ios::out|ios::trunc);
  fout << "One row per scan, one column per slice." << endl;
  for (unsigned scan=0; scan<NScan(); scan++) {
    for (unsigned int slice=0; slice<NSlice(); slice++) {
      fout << _swo._nsv[slice][scan] << " ";
    }
    fout << endl;
  } 
  fout.close();
} EddyCatch

std::pair<double,double> ReplacementManager::mean_and_std(
							  const EDDY::ReplacementManager::StatsInfo& stats,   
							  unsigned int                               minvox,  
							  unsigned int                               etc,     
							  const std::vector<std::vector<bool> >&     ovv,     
							  
							  std::pair<double,double>&                  stdev) const EddyTry
{
  
  std::pair<double,double> mval(0.0,0.0);
  unsigned int ntot = 0;
  for (unsigned int sl_gr=0; sl_gr<stats._mdiff.size();  sl_gr++) {
    for (unsigned int scan=0; scan<stats._mdiff[sl_gr].size(); scan++) {
      if (!ovv[sl_gr][scan] && stats._nvox[sl_gr][scan] >= minvox) {
	if (etc == 1) {
	  mval.first += stats._nvox[sl_gr][scan] * stats._mdiff[sl_gr][scan];
	  mval.second += stats._nvox[sl_gr][scan] * stats._msqrd[sl_gr][scan];
	  ntot += stats._nvox[sl_gr][scan];
	}
	else {
	  mval.first += stats._mdiff[sl_gr][scan];
	  mval.second += stats._msqrd[sl_gr][scan];
	  ntot += 1;
	}
      }
    } 
  }
  mval.first /= double(ntot); mval.second /= double(ntot);
  
  
  
  
  ntot = 0; stdev.first = 0.0; stdev.second = 0.0;
  for (unsigned int sl_gr=0; sl_gr<stats._mdiff.size();  sl_gr++) {
    for (unsigned int scan=0; scan<stats._mdiff[sl_gr].size(); scan++) {
      if (!ovv[sl_gr][scan] && stats._nvox[sl_gr][scan] >= minvox) {
	if (etc == 1) {
	  stdev.first += stats._nvox[sl_gr][scan] * this->sqr(stats._mdiff[sl_gr][scan] - mval.first);
	  stdev.second += stats._nvox[sl_gr][scan] * this->sqr(stats._msqrd[sl_gr][scan] - mval.second);
	}
	else {
	  stdev.first += this->sqr(stats._mdiff[sl_gr][scan] - mval.first);
	  stdev.second += this->sqr(stats._msqrd[sl_gr][scan] - mval.second);
	}
	ntot += 1;
      }
    } 
  }
  stdev.first /= double(ntot - 1); stdev.second /= double(ntot - 1);
  stdev.first = std::sqrt(stdev.first); stdev.second = std::sqrt(stdev.second);
  
  return(mval);
} EddyCatch

double MutualInfoHelper::MI(const NEWIMAGE::volume<float>& ima1,
			    const NEWIMAGE::volume<float>& ima2,
			    const NEWIMAGE::volume<float>& mask) const EddyTry
{
  
  memset(_mhist1,0,_nbins*sizeof(double));
  memset(_mhist2,0,_nbins*sizeof(double));
  memset(_jhist,0,_nbins*_nbins*sizeof(double));
  float min1, max1, min2, max2;
  if (!_lset) { min1 = ima1.min(); max1 = ima1.max(); min2 = ima2.min(); max2 = ima2.max(); }
  else { min1 = _min1; max1 = _max1; min2 = _min2; max2 = _max2; }
  unsigned int nvox=0;
  for (int z=0; z<ima1.zsize(); z++) {
    for (int y=0; y<ima1.ysize(); y++) {
      for (int x=0; x<ima1.xsize(); x++) {
	if (mask(x,y,z)) {
	  unsigned int i1 = this->val_to_indx(ima1(x,y,z),min1,max1,_nbins);
	  unsigned int i2 = this->val_to_indx(ima2(x,y,z),min2,max2,_nbins);
	  _mhist1[i1] += 1.0;
	  _mhist2[i2] += 1.0;
	  _jhist[i2*_nbins + i1] += 1.0;
	  nvox++;
	}
      }
    }
  }
  
  double je=0.0; double me1=0.0; double me2=0.0;
  for (unsigned int i1=0; i1<_nbins; i1++) {
    me1 += this->plogp(_mhist1[i1]/static_cast<double>(nvox));
    me2 += this->plogp(_mhist2[i1]/static_cast<double>(nvox));
    for (unsigned int i2=0; i2<_nbins; i2++) {
      je += this->plogp(_jhist[i2*_nbins + i1]/static_cast<double>(nvox));
    }
  }
  
  return(me1+me2-je);
} EddyCatch

double MutualInfoHelper::SoftMI(const NEWIMAGE::volume<float>& ima1,
				const NEWIMAGE::volume<float>& ima2,
				const NEWIMAGE::volume<float>& mask) const EddyTry
{
  
  memset(_mhist1,0,_nbins*sizeof(double));
  memset(_mhist2,0,_nbins*sizeof(double));
  memset(_jhist,0,_nbins*_nbins*sizeof(double));
  float min1, max1, min2, max2;
  if (!_lset) { min1 = ima1.min(); max1 = ima1.max(); min2 = ima2.min(); max2 = ima2.max(); }
  else { min1 = _min1; max1 = _max1; min2 = _min2; max2 = _max2; }
  double nvox=0.0;
  for (int z=0; z<ima1.zsize(); z++) {
    for (int y=0; y<ima1.ysize(); y++) {
      for (int x=0; x<ima1.xsize(); x++) {
	if (mask(x,y,z)) {
	  float mv = mask(x,y,z);
	  float r1, r2;
	  unsigned int i1 = this->val_to_floor_indx(ima1(x,y,z),min1,max1,_nbins,&r1);
	  unsigned int i2 = this->val_to_floor_indx(ima2(x,y,z),min2,max2,_nbins,&r2);
	  _mhist1[i1] += mv*(1.0 - r1);
	  if (r1) _mhist1[i1+1] += mv*r1;
	  _mhist2[i2] += mv*(1.0 - r2);
	  if (r2) _mhist2[i2+1] += mv*r2; 
	  _jhist[i2*_nbins + i1] += mv*(1.0 - r1)*(1.0 - r2);
	  if (r1) _jhist[i2*_nbins + i1+1] += mv*r1*(1.0 - r2);
	  if (r2) _jhist[(i2+1)*_nbins + i1] += mv*(1.0 - r1)*r2;
	  if (r1 && r2) _jhist[(i2+1)*_nbins + i1+1] += mv*r1*r2;
	  nvox += mv;
	}
      }
    }
  }
  
  double je=0.0; double me1=0.0; double me2=0.0;
  for (unsigned int i1=0; i1<_nbins; i1++) {
    me1 += this->plogp(_mhist1[i1]/nvox);
    me2 += this->plogp(_mhist2[i1]/nvox);
    for (unsigned int i2=0; i2<_nbins; i2++) {
      je += this->plogp(_jhist[i2*_nbins + i1]/nvox);
    }
  }
  
  return(me1+me2-je);
} EddyCatch



