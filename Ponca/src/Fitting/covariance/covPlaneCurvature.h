/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "./../defines.h"

#include <Eigen/Dense>

namespace Ponca
{

template < class DataPoint, class _WFunctor, typename T>
class CovPlaneCurvature : public T
{
PONCA_FITTING_DECLARE_DEFAULT_TYPES
PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum { Check = Base::PROVIDES_LOCAL_FRAME };

public:
    using Solver = Eigen::SelfAdjointEigenSolver<MatrixType>;
protected:

    using Matrix32 = Eigen::Matrix<Scalar, 3, 2>;
    using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

    VectorType   m_dmin;
    VectorType   m_dmax;

    Scalar       m_kmin;
    Scalar       m_kmax;


public:
    PONCA_EXPLICIT_CAST_OPERATORS(CovPlaneCurvature,covPlaneCurvature)
    // PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    PONCA_MULTIARCH void init( const VectorType& _evalPos )
    {
        Base::init(_evalPos);
        m_dmin = VectorType::Zero();
        m_dmax = VectorType::Zero();
        m_kmin = Scalar(0);
        m_kmax = Scalar(0);
    }

    PONCA_MULTIARCH bool addLocalNeighbor( Scalar w, const VectorType &localQ, const DataPoint &attributes ){
        return Base::addLocalNeighbor(w, localQ, attributes);
    }

    PONCA_MULTIARCH FIT_RESULT finalize()
    {
        auto res = Base::finalize();

        m_dmin = Base::m_solver.eigenvectors().col(1);
        m_dmax = Base::m_solver.eigenvectors().col(2);
        m_kmin = ( Base::m_solver.eigenvalues()(1) / Base::m_solver.eigenvalues()(0) ) * Base::m_w.evalScale();
        m_kmax = ( Base::m_solver.eigenvalues()(2) / Base::m_solver.eigenvalues()(0) ) * Base::m_w.evalScale();
        return res;
    }

    //! \brief Returns an estimate of the mean curvature
    PONCA_MULTIARCH inline Scalar kMean() const { return ( m_kmin + m_kmax ) / Scalar(2); }

    //! \brief Returns an estimate of the Gaussian curvature
    PONCA_MULTIARCH inline Scalar GaussianCurvature() const { return m_kmin * m_kmax; }

    //! \brief Returns an estimate of the minimum curvature
    PONCA_MULTIARCH inline Scalar kmin() const { return m_kmin; }

    //! \brief Returns an estimate of the maximum curvature
    PONCA_MULTIARCH inline Scalar kmax() const { return m_kmax; }

    //! \brief Returns an estimate of the minimum curvature direction $
    PONCA_MULTIARCH inline VectorType kminDirection() const { return m_dmin; }

    //! \brief Returns an estimate of the maximum curvature direction
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return m_dmax; }

    //! \brief Orthogonal projecting on the patch, such that h = f(u,v)
    PONCA_MULTIARCH inline VectorType project (const VectorType& _q) const { return Base::project(_q); }

    PONCA_MULTIARCH inline VectorType primitiveGradient() const { return Base::normal(); }

    PONCA_MULTIARCH inline Matrix32 tangentPlane (  ) const { 
        Matrix32 B;
        B.col(0) = Base::getFrameU();
        B.col(1) = Base::getFrameV();
        return B;
     }
};

/// \brief Helper alias for Covariance2DFit on points
//! [Covariance2DFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using CovPlaneCurvatureFit =
        Ponca::CovPlaneCurvature<DataPoint, _WFunctor,
            Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                    Ponca::MeanPosition<DataPoint, _WFunctor,
                        Ponca::LocalFrame<DataPoint, _WFunctor,
                            Ponca::Plane<DataPoint, _WFunctor,
                                Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>;

} //namespace Ponca
