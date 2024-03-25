#pragma once


#include <Eigen/Core>
#include <Ponca/Fitting>

namespace Ponca
{

/*!
    \brief Algebraic parabolical-cylinder primitive

    This primitive provides:
    \verbatim PROVIDES_PARABOLIC_CYLINDER \endverbatim

    \see ParabolicCylinder
*/
template < class DataPoint, class _WFunctor, typename T >
class ParabolicCylinder : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE

protected:
    enum
    {
        check = Base::PROVIDES_PRIMITIVE_BASE 
        && Base::PROVIDES_PLANE,                                        /*!< \brief Requires PrimitiveBase      */
        PROVIDES_PARABOLIC_CYLINDER,                                    /*!< \brief Provides Parabolic cylinder */
        PROVIDES_NORMAL_DERIVATIVE,                                     /*!< \brief Provides Normal Derivative  */
    };

public:
    PONCA_EXPLICIT_CAST_OPERATORS(ParabolicCylinder,parabolicCylinder)

public:

// results
public:

    using Matrix2       = Eigen::Matrix<Scalar, 2, 2>;
    using Vector2       = Eigen::Matrix<Scalar, 2, 1>;



    // f(q) = m_uc + m_ul^T * q + m_a * q^T * (m_uq^T * m_uq) * q;
    Scalar     m_uc;       /*!< \brief Constant parameter of the Algebraic hyper-ellipsoid   */
    Vector2    m_ul;       /*!< \brief Linear parameter of the Algebraic hyper-ellipsoid     */
    Scalar     m_a;        /*!< \brief Alpha quadratic parameter, used for ambiguous point   */
    Matrix2    m_uq;       /*!< \brief Quadratic parameter of the Algebraic hyper-ellipsoid  */

    // Fix to correct the orientation of the primitive
    Scalar m_correctOrientation = Scalar(1);

    // FOR 2D ELLIPSOID.
    // [TODO] Look if it is more usefull to use an Ellipsoid2D class or this one,
    //        allowing the swith between 2D Ellipsoid and Parabolic Cylinder.
    
    // Parameter saying if we're using Demi Ellipsoid or Parabolic Cylinder
    bool       m_isCylinder = true;

    // Curvature info
    Scalar     m_k1;
    Scalar     m_k2;
    VectorType m_v1;
    VectorType m_v2;

public:
    PONCA_MULTIARCH inline void init(const VectorType& _basisCenter = VectorType::Zero())
    {
        Base::init(_basisCenter);

        m_uc = Scalar(0);
        m_ul = Vector2::Zero();
        m_a  = Scalar(1);
        m_uq = Matrix2::Zero();

        m_correctOrientation = Scalar(1);

        m_v1 = VectorType::Zero();
        m_v2 = VectorType::Zero();
        m_k1 = Scalar(0);
        m_k2 = Scalar(0);

        m_isCylinder = true;
    }

    PONCA_MULTIARCH inline Scalar eval_quadratic_function(Scalar q1, Scalar q2) const {
        Vector2 q {q1, q2};
        Scalar first = m_uc + m_ul.transpose() * q;
        Scalar second = q.transpose() * m_uq * q;

        return m_correctOrientation * ( first + m_a * second ) ;
    }

    //! \brief Make the primitive fitting to be a demi-ellipsoid instead of a parabolic cylinder
    PONCA_MULTIARCH inline void setCylinder(bool b) { m_isCylinder = b; }

    PONCA_MULTIARCH inline void correct_orientation() {
        // Check the angle between Base::primitiveGradient() and Base::primitiveGradient()
        // If the angle is > 135°, we need to correct the orientation of the primitive
        VectorType n = Base::primitiveGradient();
        VectorType nGrad = primitiveGradient();
        Scalar angle = std::acos(n.dot(nGrad) / (n.norm() * nGrad.norm()));
        // if (angle > Scalar(3.0) * M_PI/Scalar(4.0)) {
        if (angle > M_PI_2) {
            m_correctOrientation = Scalar(-1);
        }
    }

    //! \brief Value of the scalar field at the location \f$ \mathbf{q} \f$
    PONCA_MULTIARCH inline Scalar potential (const VectorType& _q) const;

    /*! \brief Value of the scalar field at the evaluation point */
    PONCA_MULTIARCH inline Scalar potential() const { return potential(Base::m_w.basisCenter()); }

    //! \brief Project a point on the ellipsoid
    PONCA_MULTIARCH inline VectorType project (const VectorType& _q) const;

    //! \brief Approximation of the scalar field gradient at \f$ \mathbf{q} \f$
    PONCA_MULTIARCH inline VectorType primitiveGradient (const VectorType& _q) const;

    /*! \brief Approximation of the scalar field gradient at the evaluation point */
    PONCA_MULTIARCH inline VectorType primitiveGradient () const {
        VectorType n = primitiveGradient(Base::m_w.basisCenter());
        return n / n.norm();
    }

    // The result seems to be the same as ACP on the Hessian when using dNormal() with curvatureEstimation.h
    PONCA_MULTIARCH inline MatrixType dNormal() const;

    PONCA_MULTIARCH inline Scalar alpha_curvature () const { return m_a; }

    PONCA_MULTIARCH inline Scalar kmin () const { return m_k1; }

    PONCA_MULTIARCH inline Scalar kmax () const { return m_k2; }

    PONCA_MULTIARCH inline Scalar kMean () const { return (m_k1 + m_k2) / Scalar(2); }

    PONCA_MULTIARCH inline Scalar GaussianCurvature () const { return m_k1 * m_k2; }

    PONCA_MULTIARCH inline VectorType kminDirection() const { return m_v1; }
    
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return m_v2; }

    // template <typename BaseType>
    // PONCA_MULTIARCH inline void operator+= (const BaseType& _b)
    // {
    //     Base::operator+=(_b);
    //     const ParabolicCylinder p = _b.getParabolicCylinder();
    //     m_uc += p.m_uc;
    //     m_ul += p.m_ul;
    //     m_a  += p.m_a;
    //     m_uq += p.m_uq;
    // }

    // template <typename BaseType>
    // PONCA_MULTIARCH inline void operator-= (const BaseType& _b)
    // {
    //     Base::operator-=(_b);
    //     const ParabolicCylinder p = _b.getParabolicCylinder();
    //     // Maybe _b::ParabolicCylinder.m_uc / m_ul and so on should be protected
    //     // But it could be working as well
    //     m_uc -= p.m_uc;
    //     m_ul -= p.m_ul;
    //     m_a  -= p.m_a;
    //     m_uq -= p.m_uq;
    // }

    // PONCA_MULTIARCH inline void operator*= (const Scalar& _s)
    // {
    //     Base::operator*=(_s);
    //     m_uc *= _s;
    //     m_ul *= _s;
    //     m_a  *= _s;
    //     m_uq *= _s;
    // }

    // PONCA_MULTIARCH inline void operator/= (const Scalar& _s)
    // {
    //     Base::operator/=(_s);
    //     m_uc /= _s;
    //     m_ul /= _s;
    //     m_a  /= _s;
    //     m_uq /= _s;
    // }

    // PONCA_MULTIARCH inline ParabolicCylinder getParabolicCylinder() const 
    // {
    //     return *this;
    // }


}; //class ParabolicCylinder

#include "parabolicCylinder.hpp"
}
