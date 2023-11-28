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
class AlgebraicEllipsoid2DFitImpl: public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    // PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        Check = Base::PROVIDES_PARABOLIC_CYLINDER  //, /*!< \brief Requires PrimitiveBase, plane frame and parabolic cylinder aka special ellipsoid */
    };
    
    PONCA_EXPLICIT_CAST_OPERATORS(AlgebraicEllipsoid2DFitImpl,algebraicEllipsoid2DFitImpl)

public:
    PONCA_MULTIARCH inline void init(const VectorType& _evalPos)
    {
        Base::init(_evalPos);
        Base::m_isCylinder = false;
    }

    PONCA_MULTIARCH inline FIT_RESULT finalize () {
        PONCA_MULTIARCH_STD_MATH(abs);
        
        if (! Base::m_planeIsReady) {

            FIT_RESULT res = Base::finalize();
            return res;
        }
        else {
            FIT_RESULT res = Base::finalize();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(-2.0 * Base::m_uq);
            Eigen::Vector2d values = eig.eigenvalues();
            Eigen::MatrixXd eigenVec = eig.eigenvectors();

            Base::m_k1 = values(0);
            VectorType v1 = VectorType(0, eigenVec.col(0)(0), eigenVec.col(0)(1));
            Base::m_k2 = values(1);
            VectorType v2 = VectorType(0, eigenVec.col(1)(0), eigenVec.col(1)(1));


            if (Base::m_k1 > Base::m_k2) {
                std::swap(Base::m_k1, Base::m_k2);
                std::swap(Base::m_v1, Base::m_v2);
            }

            Base::m_v1 = Base::template localFrameToWorld<true>(v1);
            Base::m_v2 = Base::template localFrameToWorld<true>(v2);

            return res;
        }
    }


}; //class OrientedParabolicCylinderFitImpl

/// \brief Helper alias for ParabolicCylinder fitting on points
//! [ParabolicCylinderFit Definition]
//! // Fully Oriented Method PC-MLS  Mean + oriented
template < class DataPoint, class _WFunctor, typename T>
    using FullyOrientedEllipsoid2DFit =
        Ponca::AlgebraicEllipsoid2DFitImpl<DataPoint, _WFunctor,
            Ponca::OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor,
                Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                    Ponca::MeanPlaneFitImpl<DataPoint, _WFunctor,
                        Ponca::MeanNormal<DataPoint, _WFunctor,
                            Ponca::MeanPosition<DataPoint, _WFunctor,
                                    Ponca::LocalFrame<DataPoint, _WFunctor,
                                        Ponca::Plane<DataPoint, _WFunctor,
                                            Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>>;

// Near oriented PC-MLS   COV + oriented
template < class DataPoint, class _WFunctor, typename T>
    using NearOrientedEllipsoid2DFit =
        Ponca::AlgebraicEllipsoid2DFitImpl<DataPoint, _WFunctor, 
            Ponca::OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor,
                Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                    Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                        Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                            Ponca::MeanNormal<DataPoint, _WFunctor,
                                Ponca::MeanPosition<DataPoint, _WFunctor,
                                        Ponca::LocalFrame<DataPoint, _WFunctor,
                                            Ponca::Plane<DataPoint, _WFunctor,
                                                Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>>>;


template < class DataPoint, class _WFunctor, typename T>
    using BaseEllipsoid2DFit =
        Ponca::AlgebraicEllipsoid2DFitImpl<DataPoint, _WFunctor, 
    Ponca::ParabolicCylinderFitImpl<DataPoint, _WFunctor,
            Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                Ponca::CovariancePlaneFitImpl<DataPoint, _WFunctor,
                    Ponca::CovarianceFitBase<DataPoint, _WFunctor,
                        Ponca::MeanPosition<DataPoint, _WFunctor,
                                Ponca::LocalFrame<DataPoint, _WFunctor,
                                    Ponca::Plane<DataPoint, _WFunctor,
                                        Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>>;
                                            
// Base Method PC-MLS         cov + non oriented
template < class DataPoint, class _WFunctor, typename T>
    using BaseOrientedEllipsoid2DFit =
            Ponca::AlgebraicEllipsoid2DFitImpl<DataPoint, _WFunctor, 
        Ponca::ParabolicCylinderFitImpl<DataPoint, _WFunctor,
                Ponca::ParabolicCylinder<DataPoint, _WFunctor,
                    Ponca::MeanPlaneFitImpl<DataPoint, _WFunctor,
                        Ponca::MeanNormal<DataPoint, _WFunctor,
                            Ponca::MeanPosition<DataPoint, _WFunctor,
                                    Ponca::LocalFrame<DataPoint, _WFunctor,
                                        Ponca::Plane<DataPoint, _WFunctor,
                                            Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>>>>>>>;

//! [ParabolicCylinderFit Definition]
}
