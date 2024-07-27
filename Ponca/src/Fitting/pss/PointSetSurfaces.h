#pragma once

#include "./../defines.h"
#include "./../plane.h" 

#include <Eigen/Dense>

namespace Ponca
{

//!
//! Direct Computing of Surface Curvatures for Point-Set Surfaces
//! Yang and Qian
//! Eurographics Symposium on Point-Based Graphics (2007)
//!
template < class DataPoint, class _WFunctor, typename T >
class PointSetSurfaceFitImpl : public T
{
    PONCA_FITTING_DECLARE_DEFAULT_TYPES
    PONCA_FITTING_DECLARE_MATRIX_TYPE
    using Matrix33 = Eigen::Matrix<Scalar,3,3>;
    using Matrix32 = Eigen::Matrix<Scalar,3,2>;
    using Matrix22 = Eigen::Matrix<Scalar,2,2>;
    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector2 = Eigen::Matrix<Scalar,2,1>;

    enum { check = Base::PROVIDES_PLANE };
    static_assert(DataPoint::Dim == 3); // only valid in 3D
    // static_assert(std::is_same<_WFunctor, DistWeightFunc<DataPoint,PointSetSurfaceWeightKernel<Scalar>>>());

public:
    // 1st step computation data
    Scalar m_sum_w;
    Vector3 m_sum_n;
    Vector3 m_sum_p;
    Matrix33 m_sum_dn;
    Matrix33 m_sum_d2n[3];

    // 1st step results
    Vector3 m_n;
    Matrix33 m_dn;
    Matrix33 m_d2n[3];

    // 2nd step computation data
    Vector3  m_grad;
    Matrix33 m_hess;

    // 2nd step results
    Matrix22 m_W;
    Scalar m_kmin, m_kmax;
    Vector3 m_dmin, m_dmax;

    bool m_first_step;

protected:
    static Matrix32 tangentPlane(const VectorType& n);

public:
    PONCA_EXPLICIT_CAST_OPERATORS(PointSetSurfaceFitImpl,pointSetSurfaceFit)
    PONCA_FITTING_DECLARE_INIT_ADD_FINALIZE

    //! \brief Returns an estimate of the mean curvature
    PONCA_MULTIARCH inline Scalar kMean() const { return (m_kmin + m_kmax) / Scalar(2); }

    //! \brief Returns an estimate of the Gaussian curvature
    PONCA_MULTIARCH inline Scalar GaussianCurvature() const { return m_kmin * m_kmax; }

    //! \brief Returns an estimate of the minimum curvature
    PONCA_MULTIARCH inline Scalar kmin() const { return m_kmin; }

    //! \brief Returns an estimate of the maximum curvature
    PONCA_MULTIARCH inline Scalar kmax() const { return m_kmax; }

    //! \brief Returns an estimate of the minimum curvature direction $
    PONCA_MULTIARCH inline VectorType kminDirection() const { return m_dmin; }

    //! \brief Returns an estimate of the maximum curvature direction
    PONCA_MULTIARCH inline VectorType kmaxDirection() const { return m_dmax; } 
};

template < class DataPoint, class _WFunctor, typename T>
using PointSetSurfaceFit = 
    PointSetSurfaceFitImpl<DataPoint, _WFunctor, 
        Plane<DataPoint, _WFunctor,
            Ponca::PrimitiveBase<DataPoint,_WFunctor,T>>>;

// ===============================================================
// ===============================================================
// ===============================================================

template < class DataPoint, class _WFunctor, typename T >
void PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& point)
{
    Base::init(point);

    // 1st step computation data
    m_sum_w = 0;
    m_sum_n = Vector3::Zero();
    m_sum_p = Vector3::Zero();
    m_sum_dn = Matrix33::Zero();
    for(int i = 0; i < 3; ++i)
        m_sum_d2n[i] = Matrix33::Zero();

    // 1st step results
    m_n = Vector3::Zero();
    m_dn = Matrix33::Zero();
    for(int i = 0; i < 3; ++i)
        m_d2n[i] = Matrix33::Zero();

    // 2nd step computation data
    m_grad = Vector3::Zero();
    m_hess = Matrix33::Zero();

    // 2nd step results
    m_W = Matrix22::Zero();
    m_kmin = 0;
    m_kmax = 0;
    m_dmin = Vector3::Zero();
    m_dmax = Vector3::Zero();

    m_first_step = true;
}

template < class DataPoint, class _WFunctor, typename T >
bool PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(
    Scalar w, const VectorType& localQ, const DataPoint& attributes)
{
    if( Base::addLocalNeighbor(w, localQ, attributes) ) 
    {
        if(m_first_step)
        {
            const Vector3 dw = Base::m_w.spacedw(attributes.pos(), attributes);
            const Matrix33 d2w = Base::m_w.spaced2w(attributes.pos(), attributes);
            m_sum_w += w;
            m_sum_p += w * localQ;
            m_sum_n += w * attributes.normal();
            m_sum_dn += attributes.normal() * dw.transpose();
            for(int i = 0; i < 3; ++i)
                m_sum_d2n[i] += d2w * attributes.normal()[i];
        }
        else
        {
            const Scalar h = Base::m_w.evalScale();
            const Scalar h2 = h * h;

            const Scalar A = w;
            const Scalar B = -localQ.dot(m_n);
            const Vector3 C = -localQ;
            const Vector3 D = m_n + m_dn.transpose() * C;

            const Vector3 dA = -2.0/h2 * A * C;
            const Vector3 dB = m_n + m_dn.transpose() * C;
            const Matrix33 dC = Matrix33::Identity();
            Matrix33 dD = m_dn + m_dn.transpose();
            for(int i = 0; i < 3; ++i)
                dD += C[i] * m_d2n[i];

            const Scalar E = 2.0/h2*B * (1.0/h2*B*B - 1.0);
            const Scalar F = (1.0 - 3.0/h2*B*B);
            const Vector3 dE = 2.0/h2 * (3.0/h2 * B*B - 1.0) * dB;
            const Vector3 dF = -6.0/h2 * B * dB;

            m_grad += 2 * A * (E * C + F * D);
            m_hess += 2 * (
                (E*C + F*D) * dA.transpose()
                + A * (
                    E * dC +
                    C * dE.transpose() +
                    F * dD +
                    D * dF.transpose()
                ));
        }
        return true;
    }
    return false;
}

template < class DataPoint, class _WFunctor, typename T >
FIT_RESULT PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::finalize()
{
    if(Base::finalize() != STABLE  || Base::getNumNeighbors() < 3) {
        return Base::m_eCurrentState = UNSTABLE;
    }

    if(m_first_step)
    {
        m_n = m_sum_n.normalized();
        m_dn = (Matrix33::Identity() - m_n*m_n.transpose()) * m_sum_dn / m_sum_n.norm();

        const Scalar G = 1.0 / m_sum_n.norm();
        const Scalar H = 1.0 / (m_sum_n.norm() * m_sum_n.norm());
        const Vector3 dG = - 1.0 / std::pow(m_sum_n.norm(),3) * m_sum_dn.transpose() * m_sum_n;
        const Vector3 dH = - 2.0 / std::pow(m_sum_n.norm(),4) * m_sum_dn.transpose() * m_sum_n;

        for(int i = 0; i < 3; ++i)
        {
            m_d2n[i] = dG[i] * (Matrix33::Identity() - m_n*m_n.transpose()) * m_sum_dn 
                - (
                    dH[i] * m_sum_n*m_sum_n.transpose()
                    + H * m_sum_dn.col(i) * m_sum_n.transpose()
                    + H * m_sum_n * m_sum_dn.col(i).transpose()
                  ) * m_sum_dn
                + G * (Matrix33::Identity() - m_n*m_n.transpose()) * m_d2n[i];
        }

        Base::setPlane(m_n, m_sum_p/m_sum_w);

        m_first_step = false;
        return Base::m_eCurrentState = NEED_OTHER_PASS;
    }
    else
    {
        const Matrix33 W3D = m_hess / m_grad.norm();
        const Matrix32 P = tangentPlane(m_grad.normalized());
        Matrix22 W = P.transpose() * W3D * P;

        // symmetrize
        W(0,1) = W(1,0) = (W(0,1) + W(1,0))/Scalar(2);

        // sign convention
        W *= -1;

        Eigen::SelfAdjointEigenSolver<Matrix22> solver;
        solver.computeDirect(W);

        if(solver.info() != Eigen::Success) {
            return UNSTABLE;
        }

        m_kmin = solver.eigenvalues()[0];
        m_dmin = P * solver.eigenvectors().col(0);
        m_kmax = solver.eigenvalues()[1];
        m_dmax = P * solver.eigenvectors().col(1);
        
        return Base::m_eCurrentState = STABLE;
    }
}

// static
template<class DataPoint, class _WFunctor, typename T>
typename PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::Matrix32 
PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::tangentPlane(const VectorType& n)
{
    Matrix32 B;
    int i0=-1, i1=-1, i2=-1;
    n.array().abs().minCoeff(&i0); // i0: dimension where n extends the least
    i1 = (i0+1)%3;
    i2 = (i0+2)%3;

    B.col(0)[i0] = 0;
    B.col(0)[i1] = n[i2];
    B.col(0)[i2] = -n[i1];

    B.col(0).normalize();
    B.col(1) = B.col(0).cross(n);
    return B;
}

} //namespace Ponca
