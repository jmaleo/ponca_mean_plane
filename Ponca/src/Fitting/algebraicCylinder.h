#pragma once

#include <Ponca/Fitting>
#include <Eigen/Core>

namespace Ponca
{

/*!
    \brief Algebraic Cylinder primitive

    This primitive provides:
    \verbatim PROVIDES_ALGEBRAIC_ELLIPSOID \endverbatim

    \see AlgebraicSphere
*/
template < class DataPoint, class _WFunctor, typename T >
class AlgebraicCylinder : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        check = Base::PROVIDES_PRIMITIVE_BASE,  /*!< \brief Requires PrimitiveBase */
        PROVIDES_ALGEBRAIC_CYLINDER,           /*!< \brief Provides Algebraic Cylinder */
    };

public:
    PONCA_EXPLICIT_CAST_OPERATORS(AlgebraicCylinder, algebraicCylinder)

// results
public:
    Scalar m_uc;       /*!< \brief Constant parameter of the Algebraic hyper-cylinder  */
    VectorType m_ul;   /*!< \brief Linear parameter of the Algebraic hyper-cylinder    */
    MatrixType m_uq;   /*!< \brief Quadratic parameter of the Algebraic hyper-cylinder */
    Scalar m_a; 

    // Curvature info
    Scalar     m_k1;
    Scalar     m_k2;
    VectorType m_v1;
    VectorType m_v2;
public:
    PONCA_MULTIARCH inline void init(const VectorType& _basisCenter = VectorType::Zero())
    {
        Base::init(_basisCenter);

        m_a = Scalar(0);
        m_uc = Scalar(0);
        m_ul = VectorType::Zero();
        m_uq = MatrixType::Zero();

        m_k1 = Scalar(0);
        m_k2 = Scalar(0);
        m_v1 = VectorType::Zero();
        m_v2 = VectorType::Zero();
    }

    //! \brief Value of the scalar field at the location \f$ \mathbf{q} \f$
    PONCA_MULTIARCH inline Scalar potential (const VectorType& _q) const;

    /*! \brief Value of the scalar field at the evaluation point */
    PONCA_MULTIARCH inline Scalar potential() const { return m_uc; }

    //! \brief Project a point on the ellipsoid
    PONCA_MULTIARCH inline VectorType project (const VectorType& _q) const;

    //! \brief Approximation of the scalar field gradient at \f$ \mathbf{q} (not normalized) \f$
    PONCA_MULTIARCH inline VectorType primitiveGradient (const VectorType& _q) const;

    /*! \brief Approximation of the scalar field gradient at the evaluation point */
    PONCA_MULTIARCH inline VectorType primitiveGradient () const { return primitiveGradient(Base::m_w.basisCenter()); }

    MatrixType dNormal() const {return 2 * m_uq / m_ul.norm();}

    // f(x) = uc + ul^T x + x^T Uq x
    //      = uc + ul^T x + x^T P D P^T x
    //      = uc + ul^T P (P^T x) + (P^T x)^T D (P^T x)
    //      = uc + ul'^T y + y^T D y
    std::pair<VectorType,VectorType> canonical() const
    {
        Eigen::SelfAdjointEigenSolver<MatrixType> eig(m_uq);
        return std::make_pair(
            m_ul.transpose() * eig.eigenvectors(),
            eig.eigenvalues());
    }
    
    PONCA_MULTIARCH inline Scalar curvature_k (const VectorType& _q) const {
        return std::abs(kMean());
    }

    PONCA_MULTIARCH inline Scalar kmin () const { return m_k1; }

    PONCA_MULTIARCH inline Scalar kmax () const { return m_k2; }

    PONCA_MULTIARCH inline Scalar kMean () const { return (m_k1 + m_k2) / Scalar(2); }

    PONCA_MULTIARCH inline Scalar GaussianCurvature () const { return m_k1 * m_k2; }

    PONCA_MULTIARCH inline VectorType kminDirection() const { return m_v1; }
    
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return m_v2; }


}; //class AlgebraicCylinder

#include "algebraicCylinder.hpp"
}
