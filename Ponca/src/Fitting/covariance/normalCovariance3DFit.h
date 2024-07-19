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
class NormalCovariance3D : public T
{
PONCA_FITTING_DECLARE_DEFAULT_TYPES
PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum { Check = Base::PROVIDES_LOCAL_FRAME };

public:
    using Solver = Eigen::SelfAdjointEigenSolver<MatrixType>;
protected:

    MatrixType   m_cov;      /*!< \brief Covariance matrix */
    Solver       m_solver;     /*!< \brief Solver used to analyse the covariance matrix */
    VectorType   m_normal_centroid = VectorType::Zero();
    bool         m_planeIsReady = false;

public:
    PONCA_EXPLICIT_CAST_OPERATORS(NormalCovariance3D,normalCovariance3D)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    //! \brief Returns an estimate of the mean curvature
    PONCA_MULTIARCH inline Scalar kMean() const;

    //! \brief Returns an estimate of the Gaussian curvature
    PONCA_MULTIARCH inline Scalar GaussianCurvature() const;

    //! \brief Returns an estimate of the minimum curvature
    PONCA_MULTIARCH inline Scalar kmin() const;

    //! \brief Returns an estimate of the maximum curvature
    PONCA_MULTIARCH inline Scalar kmax() const;

    //! \brief Returns an estimate of the minimum curvature direction $
    PONCA_MULTIARCH inline VectorType kminDirection() const;

    //! \brief Returns an estimate of the maximum curvature direction
    PONCA_MULTIARCH inline VectorType kmaxDirection() const;

    //! \brief Orthogonal projecting on the patch, such that h = f(u,v)
    PONCA_MULTIARCH inline VectorType project (const VectorType& _q) const
    {
        return _q;
    }

    PONCA_MULTIARCH inline VectorType primitiveGradient() const
    {
        return Base::normal();
    }
};

/// \brief Helper alias for Covariance2DFit on points
//! [Covariance2DFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using NormalCovariance3DFit =
        Ponca::NormalCovariance3D<DataPoint, _WFunctor,
            Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                    Ponca::MeanPosition<DataPoint, _WFunctor,
                            Ponca::LocalFrame<DataPoint, _WFunctor,
                                Ponca::Plane<DataPoint, _WFunctor,
                                    Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>;


#include "normalCovariance3DFit.hpp"

} //namespace Ponca
