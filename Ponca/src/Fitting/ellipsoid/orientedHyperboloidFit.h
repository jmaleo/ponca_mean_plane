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
class OrientedHyperboloidFitImpl: public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        Check = Base::PROVIDES_PRIMITIVE_BASE &&
                Base::PROVIDES_MEAN_NORMAL &&
                // Base::PROVIDES_HYPERBOLOID //, /*!< \brief Requires Hyperboloid */
                Base::PROVIDES_ALGEBRAIC_ELLIPSOID /*!< \brief Requires Algebraic Ellipsoid */
    };
// results
protected:

    // computation data
    Scalar  m_sumDotPN,  /*!< \brief Sum of the dot product betwen relative positions and normals */
            m_sumDotPP;  /*!< \brief Sum of the squared relative positions */
    MatrixType m_sumProdPP, /*!< \brief Sum of exterior product of positions */
               m_sumProdPN; /*!< \brief Sum of exterior product of positions and normals */
    MatrixType m_A;

public:
    
    PONCA_EXPLICIT_CAST_OPERATORS(OrientedHyperboloidFitImpl,orientedHyperboloidFitImpl)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

private:

    PONCA_MULTIARCH inline void    m_fitting_process         ();

    PONCA_MULTIARCH inline void    m_ellipsoid_fitting       ();
    PONCA_MULTIARCH inline void    m_uq_parabolic_fitting    ();
    PONCA_MULTIARCH inline void    m_a_parabolic_fitting     ();
    PONCA_MULTIARCH inline void    m_uc_ul_parabolic_fitting ();

    PONCA_MULTIARCH inline void    m_compute_curvature       ();


}; //class OrientedHyperboloidFitImpl

/// \brief Helper alias for Hyperboloid fitting on points
//! [OrientedHyperboloidFit Definition]
template < class DataPoint, class _WFunctor, typename T>
    using OrientedHyperboloid_ACP_Fit =
        Ponca::OrientedHyperboloidFitImpl<DataPoint, _WFunctor,
            Ponca::MeanNormal<DataPoint, _WFunctor,
                Ponca::MeanPosition<DataPoint, _WFunctor,
                    Ponca::Hyperboloid<DataPoint, _WFunctor,T>>>>;

//! [ParabolicHyperboloidFit Definition]

#include "orientedHyperboloidFit.hpp"
}
