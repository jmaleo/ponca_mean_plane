#include <Eigen/Geometry>

namespace Internal
{

//!
//! \brief solve DX + XD = C
//! \param D a vector of size N representing a NxN diagonal matrix
//! \param C any NxN matrix
//!
template<typename MatrixType, class VectorType>
MatrixType solve_diagonal_sylvester(const VectorType& D, const MatrixType& C)
{
    using Index = typename VectorType::Index;
    using Scalar = typename VectorType::Scalar;

    const Index n = D.size();
    const Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    MatrixType X;
    for(Index i = 0; i < n; ++i)
    {
        for(Index j = 0; j < n; ++j)
        {
            const Scalar d = D(i) + D(j);
            if(abs(d) < epsilon)
            {
                X(i,j) = Scalar(0);
            }
            else
            {
                X(i,j) = C(i,j) / d;
            }
        }
    }
    return X;
}

//!
//! \brief solve AX + XA = C
//! \param A a symmetric NxN matrix
//! \param C any NxN matrix
//!
template<typename MatrixType>
MatrixType solve_symmetric_sylvester(const MatrixType& A, const MatrixType& C)
{
    Eigen::SelfAdjointEigenSolver<MatrixType> eig(A);
    const auto& D = eig.eigenvalues();
    const MatrixType& P = eig.eigenvectors();
    const MatrixType Pinv = P.transpose();
    const MatrixType F = Pinv * C * P;
    const MatrixType Y = solve_diagonal_sylvester(D, F);
    const MatrixType X = P * Y * Pinv;
    return X;
}

template<typename MatrixType>
MatrixType solve_symmetric_sylvester_2d(const MatrixType& A, const MatrixType& C)
{
    using Scalar = typename MatrixType::Scalar;
    using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
    using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
    if constexpr (A.rows() == 2)
    {
        Matrix4 M = Matrix4::Zero();
        M(0,0) = 2 * A(0,0); // 2*a
        M(0,1) = 2 * A(1,0); // 2*b
        M(1,0) = A(1,0); // b
        M(1,1) = A(0,0) + A(1,1); // a + d
        M(1,2) = A(1,0); // b
        M(2,1) = 2 * A(1,0); // 2*b
        M(2,2) = 2 * A(1,1); // 2*d

        Vector4 b;
        b[0] = C(0,0);
        b[1] = C(1,0);
        b[2] = C(1,1);

        // solve Mx = b
        const Vector4 x = M.colPivHouseholderQr().solve(b);

        MatrixType sol;
        sol(0,0) = x[0];
        sol(1,0) = x[1];
        sol(0,1) = x[1];
        sol(1,1) = x[2];

        return sol;
    }
    else
    {
        return MatrixType::Zero();
    }
}
} // namespace Internal

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    // 2D data
    m_sumN2D.setZero();
    m_sumP2D.setZero();
    m_sumDotPN2D = Scalar(0);
    m_sumDotPP2D = Scalar(0);
    m_prodPP2D.setZero();
    m_prodPN2D.setZero();
    m_sumH = Scalar(0);

    m_planeIsReady = false;
}

template < class DataPoint, class _WFunctor, typename T>
bool
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    auto res = Base::addLocalNeighbor(w, localQ, attributes);
    if(! m_planeIsReady)
    {   
        return res; // To change into the IF if the master is modified
    }
    else // base plane is ready, we can now fit the demi ellipsoid / Cylinder
    {
        // express neighbor in local coordinate frame
        VectorType localPos = Base::worldToLocalFrame(attributes.pos());
        Vector2 planePos = Vector2 ( *(localPos.data()+1), *(localPos.data()+2) );

        VectorType localNorm =  Base::template worldToLocalFrame<true>(attributes.normal());
        Vector2 planeNorm = Vector2 ( *(localNorm.data()+1), *(localNorm.data()+2) );

        m_sumN2D     += w * planeNorm;
        m_sumP2D     += w * planePos;
        m_sumDotPN2D += w * planeNorm.dot(planePos);
        m_sumDotPP2D += w * planePos.squaredNorm();
        m_prodPP2D   += w * planePos * planePos.transpose();
        m_prodPN2D   += w * planePos * planeNorm.transpose();
        m_sumH       += w * *(localPos.data());

        return true;
    }
}


template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::finalize () {
    PONCA_MULTIARCH_STD_MATH(abs);
    if (! m_planeIsReady) {

        FIT_RESULT res = Base::finalize();

        if (res == STABLE) {
            m_planeIsReady = true;
            return Base::m_eCurrentState = NEED_OTHER_PASS;
        }
        return res;
    }
    else {
        m_fitting_process();
        return Base::m_eCurrentState = STABLE;
    }
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_fitting_process () {
    
    m_ellipsoid_fitting();
    Base::m_correctOrientation = 1;
    
    if (Base::m_isCylinder) {
        m_uq_parabolic_fitting();
        m_a_parabolic_fitting();
        m_uc_ul_parabolic_fitting();
        m_compute_curvature();
    }
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_ellipsoid_fitting () {
    const Scalar weight = Base::getWeightSum();
    const Scalar invSumW = Scalar(1.)/weight;

    const Matrix2 A = 2 * (weight * m_prodPP2D -  m_sumP2D * m_sumP2D.transpose());
    Matrix2 C = weight * m_prodPN2D - m_sumP2D * m_sumN2D.transpose();
    C = C + C.transpose().eval();

    Base::m_uq = Internal::solve_symmetric_sylvester(A, C);
    Base::m_ul = invSumW * (m_sumN2D - Scalar(2) * Base::m_uq * m_sumP2D);
    Base::m_uc = - invSumW * ( Base::m_ul.transpose() * m_sumP2D + (m_prodPP2D * Base::m_uq).trace() + m_sumH);

    Base::m_a = Scalar(1);
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_uq_parabolic_fitting() {
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    Eigen::SelfAdjointEigenSolver<Matrix2> eig(Base::m_uq);
    Vector2 values = eig.eigenvalues();

    int higher = abs(values(0)) > abs(values(1)) ? 0 : 1;

    Scalar lambda0 = values(higher);
    Scalar lambda1 = values((higher + 1) % 2);
    Scalar t = Base::m_w.evalScale();

    // Compute alpha to put away the ambiguity of the solution
    Scalar alpha = 1;
    if (abs(lambda0) + 1/t > abs(epsilon))
        alpha = 2 * (abs(lambda0) - abs(lambda1)) / (abs(lambda0) + 1 / t);
    Base::m_a = ( alpha < Scalar(1) ) ? alpha : Scalar(1);

    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigenVec = eig.eigenvectors();
    Base::m_uq = eigenVec.col(higher) * eigenVec.col(higher).transpose();

    VectorType v1 = VectorType(0, eigenVec.col( 0 )(0), eigenVec.col( 0 )(1));
    VectorType v2 = VectorType(0, eigenVec.col( 1 )(0), eigenVec.col( 1 )(1));
    Base::m_v1 = Base::template localFrameToWorld<true>(v1);
    Base::m_v2 = Base::template localFrameToWorld<true>(v2);

}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_a_parabolic_fitting () {
    
    const Scalar weight = Base::getWeightSum();
    const Scalar invSumW = Scalar(1.)/weight;

    Matrix2 Q_squared = Base::m_uq * Base::m_uq; // 2x2
    
    Scalar num = (m_prodPP2D * Base::m_uq).trace();                    // 2x2 * 2x2 = 2x2 -> trace = 1
    Scalar denom = (m_prodPP2D * Q_squared).trace();                   // 2x2 * 2x2 = 2x2 -> trace = 1

    Scalar weight_num = (1 - invSumW);
    Scalar weight_denom = 2 * weight_num;

    Scalar a = weight_num * num / (weight_denom * denom);

    Base::m_a *= a;
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_uc_ul_parabolic_fitting () {
    const Scalar weight = Base::getWeightSum();
    const Scalar invSumW = Scalar(1.)/weight;

    Base::m_ul = invSumW * ( m_sumN2D - 2 * Base::m_a * Base::m_uq * m_sumP2D );


    Scalar A = Base::m_ul.transpose() * m_sumP2D;
    Scalar B = Base::m_a * (m_prodPP2D * Base::m_uq).trace();
    Base::m_uc = - invSumW * (A + B + m_sumH);
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_compute_curvature() {
    Scalar curv = Base::m_correctOrientation * Scalar(2) * Base::m_a;

    if (curv <= 0) {
        Base::m_k1 = curv;
        Base::m_k2 = Scalar(0);
    }
    else {
        Base::m_k1 = Scalar(0);
        Base::m_k2 = curv;
    }

}
