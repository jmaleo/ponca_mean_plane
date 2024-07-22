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
class MeanCurvature : public T
{
PONCA_FITTING_DECLARE_DEFAULT_TYPES
PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum { Check = Base::PROVIDES_LOCAL_FRAME };

public:
    using Solver = Eigen::SelfAdjointEigenSolver<MatrixType>;
protected:

    VectorType   m_pos;

public:
    PONCA_EXPLICIT_CAST_OPERATORS(MeanCurvature,meanCurvature)
    // PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    PONCA_MULTIARCH void init( const VectorType& _evalPos )
    {
        Base::init(_evalPos);
        m_pos = _evalPos;
    }

    PONCA_MULTIARCH bool addLocalNeighbor( Scalar w, const VectorType &localQ, const DataPoint &attributes ){
        auto res = Base::addLocalNeighbor(w, localQ, attributes);
        m_pos += w * attributes.pos();
        return res;

    }

    PONCA_MULTIARCH FIT_RESULT finalize()
    {
        auto res = Base::finalize();
        
        m_pos /= Base::getWeightSum();

        return res;
    }

    //! \brief Returns an estimate of the mean curvature
    PONCA_MULTIARCH inline Scalar kMean() const { 
        PONCA_MULTIARCH_STD_MATH(pow);
        PONCA_MULTIARCH_STD_MATH(abs);
        Scalar four = Scalar(4);
        Scalar two = Scalar(2);
        VectorType proj = m_pos - Base::m_w.evalPos();
        Scalar dist = proj.norm();
        return four * dist / ( pow( Base::m_w.evalScale(), two ) ); 
}

    //! \brief Returns an estimate of the Gaussian curvature
    PONCA_MULTIARCH inline Scalar GaussianCurvature() const { return Scalar(0); }

    //! \brief Returns an estimate of the minimum curvature
    PONCA_MULTIARCH inline Scalar kmin() const { return Scalar(0); }

    //! \brief Returns an estimate of the maximum curvature
    PONCA_MULTIARCH inline Scalar kmax() const { return Scalar(0); }

    //! \brief Returns an estimate of the minimum curvature direction $
    PONCA_MULTIARCH inline VectorType kminDirection() const { return VectorType::Zero(); }

    //! \brief Returns an estimate of the maximum curvature direction
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return VectorType::Zero(); }

    //! \brief Orthogonal projecting on the patch, such that h = f(u,v)
    PONCA_MULTIARCH inline VectorType project (const VectorType& _q) const
    {
        return Base::project(_q);
    }

    PONCA_MULTIARCH inline VectorType primitiveGradient() const
    {
        return Base::normal();
    }
};

/// \brief Helper alias for Covariance2DFit on points
//! [Covariance2DFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using MeanCurvatureFit =
        Ponca::MeanCurvature<DataPoint, _WFunctor,
            Ponca::MeanPlaneFitImpl<DataPoint, _WFunctor,
                    Ponca::MeanNormal<DataPoint, _WFunctor,
                        Ponca::MeanPosition<DataPoint, _WFunctor,
                            Ponca::LocalFrame<DataPoint, _WFunctor,
                                Ponca::Plane<DataPoint, _WFunctor,
                                    Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>;

} //namespace Ponca
