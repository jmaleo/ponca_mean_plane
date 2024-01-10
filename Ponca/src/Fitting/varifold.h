#pragma once

#include <Ponca/Fitting>
#include <Eigen/Eigenvalues>
#include "meanPlaneFit.h"

// TODO
// - inherit from plane primitive
// - inherit from curvature class
// - remove v0

namespace Ponca
{

// forward declarations
namespace v0 {
    template<class D, class W, typename T>
    class Varifold;
    // weight kernel not really used (see comments below)
    template<class S>
    using VarifoldWeightKernel = ConstantWeightKernel<S>; 
}
namespace v1 {
    template<class D, class W, typename T>
    class Varifold;
    template<class S>
    class VarifoldWeightKernel;
}

// ============================================================================

// 
// use version 0 or 1
//
// v0: stores a 3x3x3 tensor (close to article's notations)
// v1: stores a 3x3 matrix
// 
template<class D, class W, typename T>
using Varifold = v1::Varifold<D,W,T>;
template<class S>
using VarifoldWeightKernel = v1::VarifoldWeightKernel<S>;

// ============================================================================

namespace v0 
{

//! 
//! Implementation of 
//!     Weak and approximate curvatures of a measure: a varifold perspective
//!     B Buet, GP Leonardi, S Masnou
//!     
//! See Equation 7.2 of the article
//!
//! The _WFunctor is not used except for the centered basis and the scale.
//! Weights values of the _WFunctor are not used.
//! 
//! Remarks
//! - point weights (m_l) are set to 1
//! - only valid in 3D
//! - normals are required
//! 
template<class DataPoint, class _WFunctor, typename T>
class Varifold : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    using Mat32 = Eigen::Matrix<Scalar,3,2>;
    using Mat22 = Eigen::Matrix<Scalar,2,2>;

    static_assert(DataPoint::Dim == 3); // only valid in 3D

    enum 
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE &&
                Base::PROVIDES_MEAN_NORMAL &&
                Base::PROVIDES_LOCAL_FRAME
    };

public:
    // input
    VectorType m_n_l0;
    MatrixType m_P_l0;

    // accumulation
    MatrixType m_B_ijk[3];
    Scalar m_deno;

    // results
    Scalar m_k1, m_k2;
    VectorType m_dir1, m_dir2;

protected:
    static Scalar rho_der(Scalar t);
    Mat32 tangentPlane() const;

public:
    PONCA_EXPLICIT_CAST_OPERATORS(Varifold,varifold)
    void init(const VectorType& x_l0, const VectorType& n_l0);
    PONCA_FITTING_DECLARE_ADDNEIGHBOR
    PONCA_FITTING_DECLARE_FINALIZE

    // plane
    VectorType project(const VectorType& p) const;
    VectorType primitiveGradient(const VectorType& p) const;

};

} //namespace v0

// ============================================================================

namespace v1
{

//! 
//! Implementation of 
//!     Weak and approximate curvatures of a measure: a varifold perspective
//!     B Buet, GP Leonardi, S Masnou
//!     
//! See Equation 7.2 of the article
//!
//! The _WFunctor must be DistWeightFunc<D,VarifoldWeightKernel<S>>.
//! Since rho' is negative and the basket ignores negative weights,
//! VarifoldWeightKernel::f() returns the opposite (positive) weight
//! and Varifold::addLocalNeighbor() takes the opposite back (see WARNING).
//! 
//! Remarks
//! - point weights (m_l) are set to 1
//! - only valid in 3D
//! - normals are required
//! 
template<class DataPoint, class _WFunctor, typename T>
class Varifold : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    using Mat32 = Eigen::Matrix<Scalar,3,2>;
    using Mat22 = Eigen::Matrix<Scalar,2,2>;

    static_assert(DataPoint::Dim == 3); // only valid in 3D
    // force the use of VarifoldWeightKernel
    static_assert(std::is_same<_WFunctor, DistWeightFunc<DataPoint,VarifoldWeightKernel<Scalar>>>());

public:
    // input
    VectorType m_n_l0;
    MatrixType m_P_l0;

    // accumulation
    MatrixType m_A;

    // results
    Scalar m_k1, m_k2;
    VectorType m_dir1, m_dir2;

    // plane
    bool m_planeIsReady = false;
    VectorType m_sumN; // sum of normals
    Scalar     m_sumWeight; // sum of weights


protected:
    Mat32 tangentPlane() const;

public:
    PONCA_EXPLICIT_CAST_OPERATORS(Varifold,varifold)
    void init(const VectorType& x_l0, const VectorType& n_l0);
    void init(const VectorType& _evalPos);
    PONCA_FITTING_DECLARE_ADDNEIGHBOR
    PONCA_FITTING_DECLARE_FINALIZE

    PONCA_MULTIARCH inline Scalar f_smooth  (const Scalar& _x) const { Scalar v = _x*_x - Scalar(1.); return v*v; }
    // PONCA_MULTIARCH inline Scalar f_smooth  (const Scalar& _x) const { Scalar v = Scalar(1); return v; }

    // plane
    VectorType project(const VectorType& p) const;
    VectorType primitiveGradient (const VectorType& p) const;
    VectorType primitiveGradient () const { return primitiveGradient(Base::m_w.basisCenter()); }



    PONCA_MULTIARCH inline Scalar kmin () const { return m_k1; }
    PONCA_MULTIARCH inline Scalar kmax () const { return m_k2; }
    PONCA_MULTIARCH inline Scalar kMean () const { return (m_k1 + m_k2)/2; }
    PONCA_MULTIARCH inline Scalar GaussianCurvature () const { return m_k1 * m_k2; }
    
    
    PONCA_MULTIARCH inline VectorType kminDirection() const { return m_dir1; }
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return m_dir2; }

};

template<class _Scalar>
class VarifoldWeightKernel
{
public:
    using Scalar = _Scalar;

    Scalar f(const Scalar& _x) const {
        const Scalar y = Scalar(1) / (1 - _x*_x);
        //
        // WARNING
        // rho prime is negative but the basket ignores negative weights
        // (see Basket::addNeighbor()), so the opposite is returned here
        // and Varifold::addLocalNeighbor() takes the opposite again
        // 
        // return - 2 * _x * y * y * std::exp(-y);
        return + 2 * _x * y * y * std::exp(-y);
    }
};

} //namespace v1
} //namespace Ponca

#include "varifold.hpp"