#pragma once

#include <Eigen/Dense>

#include <Ponca/Fitting>

namespace Ponca
{

/*!
    \brief 
    \see 
*/
template < class DataPoint, class _WFunctor, typename T >
class OrientedParabolicCylinderFitImpl: public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    // PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE &&
                Base::PROVIDES_MEAN_NORMAL &&  
                Base::PROVIDES_LOCAL_FRAME && 
                Base::PROVIDES_PARABOLIC_CYLINDER //, /*!< \brief Requires PrimitiveBase and plane*/
    };

public:
    using Matrix2       = Eigen::Matrix<Scalar, 2, 2>;
    using Vector2       = Eigen::Matrix<Scalar, 2, 1>;
// results
protected:

    // 2D data
    Vector2    m_sumN2D,      /*!< \brief Sum of the normal 2D vectors */
               m_sumP2D;      /*!< \brief Sum of the relative 2D positions */
    Scalar     m_sumDotPN2D,  /*!< \brief Sum of the dot product betwen relative 2D positions and 2D normals */
               m_sumDotPP2D;  /*!< \brief Sum of the squared relative 2D positions */
    Matrix2    m_prodPP2D,    /*!< \brief Sum of exterior product of 2D positions */
               m_prodPN2D;    /*!< \brief Sum of exterior product of 2D positions and 2D normals */
    Scalar     m_sumH;        /*!< \brief Sum of the weigthed height field */
    
    bool m_planeIsReady {false};

public:
    
    PONCA_EXPLICIT_CAST_OPERATORS(OrientedParabolicCylinderFitImpl,orientedParabolicCylinderFitImpl)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    PONCA_MULTIARCH inline OrientedParabolicCylinderFitImpl getOrientedParabolicCylinderFit() const { return *this; }
    
private:

    PONCA_MULTIARCH inline void    m_fitting_process         ();

    PONCA_MULTIARCH inline void    m_ellipsoid_fitting       ();
    PONCA_MULTIARCH inline void    m_uq_parabolic_fitting    ();
    PONCA_MULTIARCH inline void    m_a_parabolic_fitting     ();
    PONCA_MULTIARCH inline void    m_uc_ul_parabolic_fitting ();

}; //class OrientedParabolicCylinderFitImpl

/// \brief Helper alias for ParabolicCylinder fitting on points
//! [ParabolicCylinderFit Definition]
//! // Fully Oriented Method PC-MLS  Mean + oriented
template < class DataPoint, class _WFunctor, typename T>
    using FullyOrientedParabolicCylinderFit =
        Ponca::OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor,
            Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                Ponca::MeanPlaneFitImpl<DataPoint, _WFunctor,
                    Ponca::MeanNormal<DataPoint, _WFunctor,
                        Ponca::MeanPosition<DataPoint, _WFunctor,
                                Ponca::LocalFrame<DataPoint, _WFunctor,
                                    Ponca::Plane<DataPoint, _WFunctor,
                                        Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>;

// // Near oriented PC-MLS   COV + oriented
// template < class DataPoint, class _WFunctor, typename T>
//     using NearOrientedParabolicCylinderFit =
//     Ponca::OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor,
//         Ponca::ParabolicCylinder<DataPoint, _WFunctor,
//             Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
//                 Ponca::CovarianceFitBase<DataPoint, _WFunctor,
//                         Ponca::MeanNormal<DataPoint, _WFunctor,
//                             Ponca::MeanPosition<DataPoint, _WFunctor,
//                                 Ponca::LocalFrame<DataPoint, _WFunctor,
//                                     Ponca::Plane<DataPoint, _WFunctor,
//                                         Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>>;

//! [ParabolicCylinderFit Definition]

#include "orientedParabolicCylinderFit.hpp"
}
