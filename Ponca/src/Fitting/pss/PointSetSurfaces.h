#pragma once

#include "./../defines.h"
#include "./../plane.h" 

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

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
    using Matrix44 = Eigen::Matrix<Scalar,4,4>;
    using Matrix33 = Eigen::Matrix<Scalar,3,3>;
    using Matrix32 = Eigen::Matrix<Scalar,3,2>;
    using Matrix22 = Eigen::Matrix<Scalar,2,2>;
    using Vector3 = Eigen::Matrix<Scalar,3,1>;
    using Vector2 = Eigen::Matrix<Scalar,2,1>;
    using Row3 = Eigen::Matrix<Scalar,1,3>;
    using Tensor333 = Eigen::TensorFixedSize<Scalar,Eigen::Sizes<3,3,3>>;

    enum { check = Base::PROVIDES_PLANE };
    static_assert(DataPoint::Dim == 3); // only valid in 3D
    // static_assert(std::is_same<_WFunctor, DistWeightFunc<DataPoint,PointSetSurfaceWeightKernel<Scalar>>>());

    Scalar H() const;
    Scalar K() const;

public:
    // 1st step computation data
    Scalar m_sum_w;
    Vector3 m_sum_n;
    Vector3 m_sum_p;
    Matrix33 m_sum_dn;
    Tensor333 m_sum_d2n;

    // 1st step results
    Vector3 m_n;
    Matrix33 m_dn;
    Tensor333 m_d2n;

    // 2nd step computation data
    Row3  m_grad;
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
    PONCA_MULTIARCH inline Scalar kMean() const { return H(); }

    //! \brief Returns an estimate of the Gaussian curvature
    PONCA_MULTIARCH inline Scalar GaussianCurvature() const { return K(); }

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
    m_sum_n.setZero();
    m_sum_p.setZero();
    m_sum_dn.setZero();
    m_sum_d2n.setZero();

    // 1st step results
    m_n.setZero();
    m_dn.setZero();
    m_d2n.setZero();

    // 2nd step computation data
    m_grad.setZero();
    m_hess.setZero();

    // 2nd step results
    m_W.setZero();
    m_kmin = 0;
    m_kmax = 0;
    m_dmin.setZero();
    m_dmax.setZero();

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
            const Row3 dw = Base::m_w.spacedw(attributes.pos(), attributes).transpose();
            const Matrix33 d2w = Base::m_w.spaced2w(attributes.pos(), attributes); // symmetric
            m_sum_w += w;
            m_sum_p += w * localQ;
            m_sum_n += w * attributes.normal();
            m_sum_dn += attributes.normal() * dw;
            for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
            for(int k = 0; k < 3; ++k)
                m_sum_d2n(i,j,k) += attributes.normal()[i] * d2w(k,j);
        }
        else
        {
            const Scalar h = Base::m_w.evalScale();
            const Scalar h2 = h * h;

            const Scalar A = w;
            const Scalar B = -localQ.dot(m_n);
            const Vector3 C = -localQ;
            const Vector3 D = m_n + m_dn.transpose() * C;

            const Row3 dA = -2.0/h2 * A * C.transpose();
            const Row3 dB = m_n.transpose() + C.transpose() * m_dn;
            const Matrix33 dC = Matrix33::Identity();
            Matrix33 dD = m_dn + m_dn.transpose();
            for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
            for(int k = 0; k < 3; ++k)
                dD(i,j) += C[k] * m_d2n(k,i,j);

            const Scalar E = 2.0/h2*B * (1.0/h2*B*B - 1.0);
            const Scalar F = (1.0 - 3.0/h2*B*B);
            const Row3 dE = 2.0/h2 * (3.0/h2 * B*B - 1.0) * dB;
            const Row3 dF = -6.0/h2 * B * dB;

            m_grad += 2 * A * (E * C.transpose() + F * D.transpose());
            m_hess += 2 * (
                (E*C + F*D) * dA
                + A * (
                    E * dC +
                    C * dE +
                    F * dD +
                    D * dF
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
        const Scalar d = m_sum_n.norm();

        m_n = m_sum_n / d;
        m_dn = (Matrix33::Identity() - m_n*m_n.transpose()) * m_sum_dn / d;

        const Scalar G = 1.0 / d;
        const Row3 dG = - 1.0 / (d*d*d) * m_sum_n.transpose() * m_sum_dn;
 
        for(int k = 0; k < 3; ++k)
        {
            Matrix33 sum_d2n_k = Matrix33::Zero();
            for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                sum_d2n_k(i,j) = m_sum_d2n(i,j,k);

            const Matrix33 d2n_k = dG[k] * (Matrix33::Identity() - m_n*m_n.transpose()) * m_sum_dn 
                 - G * (
                     m_dn.col(k) * m_n.transpose()
                     + m_n * m_dn.col(k).transpose()
                   ) * m_sum_dn;
                 + G * (Matrix33::Identity() - m_n*m_n.transpose()) * sum_d2n_k;

            for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                m_d2n(i,j,k) = d2n_k(i,j);
        }

        Base::setPlane(m_n, m_sum_p/m_sum_w);

        m_first_step = false;
        return Base::m_eCurrentState = NEED_OTHER_PASS;
    }
    else
    {
        const Matrix33 W3D = m_hess / m_grad.norm();
        const Matrix32 P = tangentPlane(m_grad.transpose().normalized());
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

template < class DataPoint, class _WFunctor, typename T >
typename PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::Scalar 
PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::H() const
{
    const Scalar norm_grad = m_grad.norm();
    return (m_grad * m_hess * m_grad.transpose() - norm_grad*norm_grad * m_hess.trace()) / std::pow(norm_grad, 3) / 2;
}

template < class DataPoint, class _WFunctor, typename T >
typename PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::Scalar 
PointSetSurfaceFitImpl<DataPoint, _WFunctor, T>::K() const
{
    Matrix44 M;
    M.template block<3,3>(0,0) = m_hess;
    M.template block<3,1>(0,3) = m_grad.transpose();
    M.template block<1,3>(3,0) = m_grad;
    M(3,3) = 0.0;
    const Scalar norm_grad = m_grad.norm();
    return - M.determinant() / std::pow(norm_grad, 4);
}

} //namespace Ponca
