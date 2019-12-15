//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Provide an interface matching BASISFIELD::splinefield, specifically for use with
///        EDDY::MoveBySuscCF class
/// \details Essentially this is a wrapper around a BASISFIELD::splinefield object, which
///          re-implements some of the more expensive calculations on the GPU, whilst simply
///          passing other calculations through to the underlying BASISFIELD::splinefield
///          object. As such, expect limited functionality as compared to the BASISFIELD
///          version - just enough to get EDDY::MoveBySuscCF running faster. Additionally,
///          we are using the pimpl idiom here.
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////
#ifndef CBF_SPLINE_FIELD_CUH
#define CBF_SPLINE_FIELD_CUH

#include "newimage/newimageall.h"
#include "newmat.h"
#include "miscmaths/bfmatrix.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"

#include <boost/shared_ptr.hpp>

#include <vector>
#include <memory>


namespace CBF
{
  class CBFSplineField
  {
    public:
      
      ~CBFSplineField();
      
      CBFSplineField(CBFSplineField&& rhs);
      
      CBFSplineField& operator=(CBFSplineField&& rhs);
      
      CBFSplineField(const CBFSplineField& rhs);
      
      CBFSplineField& operator=(const CBFSplineField& rhs);

      
      
      CBFSplineField(const std::vector<unsigned int>& psz, const std::vector<double>& pvxs,
          const std::vector<unsigned int>& pksp, int porder=3);

      unsigned int CoefSz_x() const;
      unsigned int CoefSz_y() const;
      unsigned int CoefSz_z() const;
      
      void AsVolume(NEWIMAGE::volume<float>& vol, BASISFIELD::FieldIndex fi=BASISFIELD::FIELD);
      
      void SetCoef(const NEWMAT::ColumnVector& pcoef);
      
      double BendEnergy() const; 
      
      NEWMAT::ReturnMatrix BendEnergyGrad() const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> BendEnergyHess(
          MISCMATHS::BFMatrixPrecisionType prec) const;
      
      NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&  ima1,
                               const NEWIMAGE::volume<float>&  ima2,
                               const NEWIMAGE::volume<char>    *mask)
                               const;
      
      NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                               const NEWIMAGE::volume<float>&    ima1,
                               const NEWIMAGE::volume<float>&    ima2,
                               const NEWIMAGE::volume<char>      *mask)
                               const;
      
      NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&    ima,
                               const NEWIMAGE::volume<char>      *mask)
                               const;
      
      NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                               const NEWIMAGE::volume<float>&    ima,
                               const NEWIMAGE::volume<char>      *mask)
                               const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima1,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv,
                                                 const NEWIMAGE::volume<float>& ima,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv, 
                                                 const NEWIMAGE::volume<float>& ima1,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>& deriv1,
                                                 const NEWIMAGE::volume<float>& ima1,
                                                 const std::vector<unsigned int>& deriv2,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;
      
      boost::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>& ima1,
                                                 const BASISFIELD::basisfield& bf2,
                                                 const NEWIMAGE::volume<float>& ima2,
                                                 const NEWIMAGE::volume<char> *mask,
                                                 MISCMATHS::BFMatrixPrecisionType prec)
                                                 const;

    private:
      
      class Impl;
      
      std::unique_ptr<Impl> pimpl_;
  }; 
} 
#endif 

