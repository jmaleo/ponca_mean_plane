#pragma once

#include <Ponca/Fitting>

#include <Eigen/Dense>


namespace Ponca
{

/*!
    \brief 
    \see 
*/
template < class DataPoint, class _WFunctor, typename T >
class ParabolicCylinderFitImpl: public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    // PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE &&
                Base::PROVIDES_LOCAL_FRAME &&
                Base::PROVIDES_PARABOLIC_CYLINDER, /*!< \brief Requires PrimitiveBase and plane*/
    };

public:
    using SampleMatrix  = Eigen::Matrix<Scalar, 7, 7>;
    using SampleVector  = Eigen::Matrix<Scalar, 7, 1>;
    using Vector7       = Eigen::Matrix<Scalar, 7, 1>;

    // Todo : remove when differential properties are implemented in other class
    using Matrix2       = Eigen::Matrix<Scalar, 2, 2>;
    using Vector2       = Eigen::Matrix<Scalar, 2, 1>;
// results
protected:

    SampleMatrix m_A_cov;
    SampleVector m_F_cov;

    Vector2      m_vector_uq;

    bool m_planeIsReady {false};
public:
    
    PONCA_EXPLICIT_CAST_OPERATORS(ParabolicCylinderFitImpl,parabolicCylinderFitImpl)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

private:

    PONCA_MULTIARCH inline void    m_fitting_process         ();

    PONCA_MULTIARCH inline void    m_ellipsoid_fitting       ();
    PONCA_MULTIARCH inline void    m_uq_parabolic_fitting    ();
    PONCA_MULTIARCH inline void    m_a_parabolic_fitting     ();
    PONCA_MULTIARCH inline void    m_uc_ul_parabolic_fitting ();

    PONCA_MULTIARCH inline void    m_compute_curvature       ();

    
}; //class ParabolicCylinderFitImpl

/// \brief Helper alias for ParabolicCylinder fitting on points
//! [ParabolicCylinderFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using BaseParabolicCylinderFit =
    Ponca::ParabolicCylinderFitImpl<DataPoint, _WFunctor,
        Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                        Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                            Ponca::MeanNormal<DataPoint, _WFunctor,
                                Ponca::MeanPosition<DataPoint, _WFunctor,
                                        Ponca::LocalFrame<DataPoint, _WFunctor,
                                            Ponca::Plane<DataPoint, _WFunctor,
                                                Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>>;
                                            
// Base Method PC-MLS         cov + non oriented
template < class DataPoint, class _WFunctor, typename T>
    using BaseOrientedParabolicCylinderFit =
        Ponca::ParabolicCylinderFitImpl<DataPoint, _WFunctor,
            Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                    Ponca::MeanPlaneFitImpl<DataPoint, _WFunctor,
                        Ponca::MeanNormal<DataPoint, _WFunctor,
                                Ponca::MeanPosition<DataPoint, _WFunctor,
                                        Ponca::LocalFrame<DataPoint, _WFunctor,
                                            Ponca::Plane<DataPoint, _WFunctor,
                                                Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>;
//! [ParabolicCylinderFit Definition]

#include "parabolicCylinderFit.hpp"
}
