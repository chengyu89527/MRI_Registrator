/////////////////////////////////////////////////////////////////////
///
/// \file EddyFunctors.h
/// \brief Declarations of functors that I use for the CUDA implementation of Eddy
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford 
///


#ifndef EddyFunctors_h
#define EddyFunctors_h

#include <cuda.h>
#include <thrust/functional.h>
#include <thrust/random.h>

namespace EDDY {






template<typename T>
class Binarise : public unary_function<T,T>
{
public:
  Binarise(const T& thr) : _ll(thr), _ul(std::numeric_limits<T>::max()) {}
  Binarise(const T& ll, const T& ul) : _ll(ll), _ul(ul) {}
  __host__ __device__ T operator()(const T& x) const { return(static_cast<T>(x > _ll && x < _ul)); }
private:
  const T _ll;
  const T _ul;
  Binarise() : _ll(static_cast<T>(0)), _ul(static_cast<T>(0)) {} 
};







template<typename T>
class MakeNormRand : public unary_function<unsigned int,T>
{
public:
  MakeNormRand(const T& mu, const T& sigma) : _mu(mu), _sigma(sigma) {}
  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::normal_distribution<T> dist(_mu,_sigma);
    rng.discard(n);
    return(dist(rng));
  }
private:
  const T _mu;
  const T _sigma;
  MakeNormRand() : _mu(static_cast<T>(0)), _sigma(static_cast<T>(1)) {} 
};






template<typename T>
class MulByScalar : public unary_function<T,T>
{
public:
  MulByScalar(const T& scalar) : _scalar(scalar) {}
  __host__ __device__ T operator()(const T& x) const { return(_scalar * x); }
private:
  const T _scalar;
  MulByScalar() : _scalar(static_cast<T>(1)) {} 
};






template<typename T1, typename T2>
class MaskedSquare : public binary_function<T1,T1,T2>
{
public:
  MaskedSquare() {}
  __host__ __device__ T2 operator()(const T1& arg1, const T1& arg2) const { return(static_cast<T2>(arg1*arg1*arg2)); }
};






template<typename T1, typename T2>
class SumSquare : public binary_function<T2,T1,T2>
{
public:
  SumSquare() {}
  __host__ __device__ T2 operator()(const T2& arg1, const T1& arg2) const { return(arg1 + static_cast<T2>(arg2*arg2)); }
};






template<typename T1, typename T2>
class Product : public binary_function<T1,T1,T2>
{
public:
  Product() {}
  __host__ __device__ T2 operator()(const T1& arg1, const T1& arg2) const { return(static_cast<T2>(arg1*arg2)); }
};






template<typename T1, typename T2>
class Sum : public binary_function<T2,T1,T2>
{
public:
  Sum() {}
  __host__ __device__ T2 operator()(const T2& arg1, const T1& arg2) const { return(arg1 + static_cast<T2>(arg2)); }
};






template<typename T>
class MulAndAdd : public binary_function<T,T,T>
{
public:
  MulAndAdd(const T& scalar) : _scalar(scalar) {}
  __host__ __device__ T operator()(const T& arg1, const T& arg2) const { return(arg1 + _scalar*arg2); }
private:
  const T _scalar;
  MulAndAdd() : _scalar(static_cast<T>(1)) {} 
};

} 


#endif 

