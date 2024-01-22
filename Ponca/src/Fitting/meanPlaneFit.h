/*
 Copyright (C) 2018 Nicolas Mellado <nmellado0@gmail.com>

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "./defines.h"
#include "./mean.h"       // used to define MeanPlaneFit
#include "./plane.h"      // used to define MeanPlaneFit
#include "./localFrame.h" // used to define MeanPlaneFit


namespace Ponca
{

/*!
    \brief Plane fitting procedure computing the frame plane using mean position and normal

    \inherit Concept::FittingProcedureConcept

    \see Plane
    \see localFrame
*/
template < class DataPoint, class _WFunctor, typename T >
class MeanPlaneFitImpl : public T
{
PONCA_FITTING_DECLARE_DEFAULT_TYPES
PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum { Check = Base::PROVIDES_PLANE
                && Base::PROVIDES_MEAN_POSITION 
                && Base::PROVIDES_MEAN_NORMAL 
                && Base::PROVIDES_LOCAL_FRAME
         };


public:
    PONCA_EXPLICIT_CAST_OPERATORS(MeanPlaneFitImpl,meanPlaneFit)

    /*!
     * \brief This function fits the plane using mean normal and position.

     * We use the localFrame class to store the frame informations.
     * Given the mean normal, we can compute the frame plane.
     * m_u and m_v are computed using the cross product, to ensure orthogonality.
     * \see LocalFrame
     * \see computeFrameFromNormalVector
     */
    PONCA_FITTING_APIDOC_FINALIZE
    PONCA_MULTIARCH inline FIT_RESULT finalize()
    {
        // handle specific configurations
        bool conflict = false;
        if(Base::finalize() == STABLE)
        {
            if (Base::plane().isValid()) conflict = true;
            VectorType norm = Base::m_sumN / Base::getWeightSum();
            Base::setPlane(norm, Base::barycenter());
            Base::computeFrameFromNormalVector(norm);
            Base::m_eCurrentState = STABLE;
        }
        if (conflict)
            return CONFLICT_ERROR_FOUND;
        else
            return STABLE;
    }

}; //class MeanPlaneFitImpl

/// \brief Helper alias for Plane fitting on points using MeanPlaneFitImpl
//! [MeanPlaneFit Definition]
    template < class DataPoint, class _WFunctor, typename T>
    using MeanPlaneFit =
    MeanPlaneFitImpl<DataPoint, _WFunctor,
        MeanNormal<DataPoint, _WFunctor,
            MeanPosition<DataPoint, _WFunctor,
                LocalFrame<DataPoint, _WFunctor,
                    Plane<DataPoint, _WFunctor,T>>>>>;
//! [MeanPlaneFit Definition]

/*!
    \brief Internal generic class computing the derivatives of mean plane fits
    \inherit Concept::FittingProcedureConcept

    \see Plane
    \see localFrame
*/
template < class DataPoint, class _WFunctor, int DiffType, typename T>
class MeanPlaneDerImpl : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE
    PONCA_FITTING_DECLARE_DEFAULT_DER_TYPES

protected:
    enum { Check = Base::PROVIDES_PLANE
                && Base::PROVIDES_MEAN_NORMAL_DERIVATIVE
                && Base::PROVIDES_MEAN_POSITION_DERIVATIVE, 
                PROVIDES_MEAN_PLANE_DERIVATIVE,                    /*!< \brief Provides derivatives for hyper-planes */
                PROVIDES_NORMAL_DERIVATIVE
         };

private:    
    VectorArray m_dNormal {VectorArray::Zero()};    /*!< \brief Derivatives of the hyper-plane normal */
    ScalarArray m_dDist {ScalarArray::Zero()};      /*!< \brief Derivatives of the MLS scalar field */


public:
    PONCA_EXPLICIT_CAST_OPERATORS_DER(MeanPlaneDerImpl,meanPlaneDer)

    PONCA_MULTIARCH inline FIT_RESULT finalize(){
        Base::finalize();
        // Test if base finalize end on a viable case (stable / unstable)
        if (this->isReady())
        {
            VectorType   barycenter = Base::barycenter();
            VectorArray dBarycenter = Base::barycenterDerivatives();

            VectorType normal = Base::primitiveGradient();
            m_dNormal = Base::m_dSumN / Base::getWeightSum();

            for (int k = 0; k < DataPoint::Dim; ++k){
                VectorType dDiff = dBarycenter.col(k);
                if (k > 0 || !Base::isScaleDer())
                    dDiff(Base::isScaleDer() ? k - 1 : k) += 1;
                m_dDist(k) = m_dNormal.col(k).dot(barycenter) + normal.dot(dDiff);
            }
            
            // \fixme we shouldn't need this normalization, however currently the derivatives are overestimated by a factor 2
            // m_dNormal /= Scalar(2.);
        }
        return Base::m_eCurrentState;
    }

    /*! \brief Returns the derivatives of the scalar field at the evaluation point */
    PONCA_MULTIARCH inline ScalarArray dPotential() const { return m_dDist; }

    /*! \brief Returns the derivatives of the primitive normal */
    PONCA_MULTIARCH inline VectorArray dNormal() const { return m_dNormal; }

}; //class MeanPlaneFitImpl

/// \brief Helper alias for Plane fitting on points using MeanPlaneDerImpl
//! [MeanPlaneDer Definition]
    template < class DataPoint, class _WFunctor, int DiffType, typename T>
    using MeanPlaneDer =
        MeanPlaneDerImpl<DataPoint, _WFunctor, DiffType,
            MeanNormalDer<DataPoint, _WFunctor, DiffType,
                MeanPositionDer<DataPoint, _WFunctor, DiffType, T>>>;
//! [MeanPlaneDer Definition]

} //namespace Ponca
