/*
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "./algebraicSphere.h"
#include "./unorientedSphereFit.h"
#include "./orientedSphereFit.h"
#include "./sphereFit.h"
#include "./mean.h"          // used to define OrientedSphereFit

namespace Ponca
{

/*!
    \brief Algebraic Sphere fitting procedure on oriented point sets

    Method published in \cite Guennebaud:2007:APSS.

    \inherit Concept::FittingProcedureConcept

    \see AlgebraicSphere
*/
template < class DataPoint, class _WFunctor, typename T >
class SphereCurvature : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES

protected:
    enum
    {
        Check = Base::PROVIDES_ALGEBRAIC_SPHERE, 
        PROVIDES_SPHERE_CURVATURE
    };

    // computation data
    Scalar  m_mean; /*!< \brief Mean curvature */

public:
    PONCA_EXPLICIT_CAST_OPERATORS(SphereCurvature,sphereCurvature)

    /// \brief Initialize the computation
    PONCA_MULTIARCH inline void init(const VectorType& _evalPos)
    {
        Base::init(_evalPos);
        m_mean = Scalar(0);
    }

    PONCA_MULTIARCH inline FIT_RESULT finalize () {
        PONCA_MULTIARCH_STD_MATH(abs);
        PONCA_MULTIARCH_STD_MATH(pow);
        
        FIT_RESULT res = Base::finalize();
        if (res != STABLE)
            return res;
        
        // Compute the mean curvature
        const Scalar invSumW = Scalar(1.) / Base::getWeightSum();

        // Apply the normalisation
        Base::applyPrattNorm();

        // Compute the mean curvature
        m_mean = Scalar(2) * Base::m_uq;

        return res;
    }

    /// \brief Accessor to the mean curvature
    PONCA_MULTIARCH inline Scalar kMean() const { return m_mean; }

    /// \brief Accessor to the Gaussian curvature
    PONCA_MULTIARCH inline Scalar GaussianCurvature() const { return Scalar(0); }

    /// \brief Accessor to the min curvature
    PONCA_MULTIARCH inline Scalar kmin() const { return Scalar(0); }

    /// \brief Accessor to the max curvature
    PONCA_MULTIARCH inline Scalar kmax() const { return Scalar(0); }

    /// \brief Accessor to the kmin direction
    PONCA_MULTIARCH inline VectorType kminDirection() const { return VectorType(1, 0, 0); }

    /// \brief Accessor to the kmax direction
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return VectorType(1, 0, 0); }

}; //class SphereCurvature

/// \brief Helper alias for simple Oriented Sphere estimator on 3D points using UnorientedSphereFitImpl
template < class DataPoint, class _WFunctor, typename T>
using SimpleUnorientedSphereFit =
SphereCurvature<DataPoint, _WFunctor,
    UnorientedSphereFitImpl<DataPoint, _WFunctor,
            MeanPosition<DataPoint, _WFunctor,
                    AlgebraicSphere<DataPoint, _WFunctor,T>>>>;

/// \brief Helper alias for simple Sphere estimator on 3D points using SphereFitImpl
template < class DataPoint, class _WFunctor, typename T>
using SimpleSphereFit =
SphereCurvature<DataPoint, _WFunctor,
    SphereFitImpl<DataPoint, _WFunctor,
        AlgebraicSphere<DataPoint, _WFunctor,T>>>;

/// \brief Helper alias for simple Oriented Sphere estimator on 3D points using OrientedSphereFitImpl
template < class DataPoint, class _WFunctor, typename T>
using SimpleOrientedSphereFit =
SphereCurvature<DataPoint, _WFunctor,
    OrientedSphereFitImpl<DataPoint, _WFunctor,
            MeanPosition<DataPoint, _WFunctor,
                MeanNormal<DataPoint, _WFunctor,
                    AlgebraicSphere<DataPoint, _WFunctor,T>>>>>;

} // namespace Ponca
