#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_A_cov.setZero();
    m_F_cov.setZero();

    m_planeIsReady = false;
}

template < class DataPoint, class _WFunctor, typename T>
bool
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    auto res = Base::addLocalNeighbor(w, localQ, attributes);
    if(! m_planeIsReady)
    {   
        return res; // To change into the IF if the master is modified
    }
    else // base plane is ready, we can now fit the patch
    {
        // express neighbor in local coordinate frame
        VectorType local = Base::worldToLocalFrame(attributes.pos());

        const Scalar& f = *(local.data());
        const Scalar& x = *(local.data()+1);
        const Scalar& y = *(local.data()+2);

        Scalar xy = x*y;
        Scalar xx = x*x;
        Scalar yy = y*y;
        
        Eigen::Vector<Scalar, 7> v {1, x, y , xx, xy, xy, yy};

        m_A_cov += w * v * v.transpose();
        m_F_cov += w * v * f;

        return true;
    }
}



template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::finalize () {
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
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_fitting_process () {
    
    m_ellipsoid_fitting();

    if (Base::m_isCylinder) {
        m_uq_parabolic_fitting();
        m_a_parabolic_fitting();
        m_uc_ul_parabolic_fitting();
        m_compute_curvature();
    }

}

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_ellipsoid_fitting () {

    Eigen::BDCSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(m_A_cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Vector7 x = svd.solve(m_F_cov);

    Base::m_uc      = x(0,0);     
    Base::m_ul(0)   = x(1,0);  
    Base::m_ul(1)   = x(2,0);  
    Base::m_uq(0,0) = x(3,0);
    Base::m_uq(0,1) = x(4,0);
    Base::m_uq(1,0) = x(5,0);
    Base::m_uq(1,1) = x(6,0);
}

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_uq_parabolic_fitting () {
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    Eigen::SelfAdjointEigenSolver<Matrix2> eig(Base::m_uq);
    Vector2 values = eig.eigenvalues();

    int higher = abs(values(0)) > abs(values(1)) ? 0 : 1;

    Scalar lambda0 = values(higher);
    Scalar lambda1 = values((higher + 1) % 2);
    Scalar t = Base::m_w.evalScale();

    Scalar alpha = 1;
    if (abs(lambda0 + 1/t) > epsilon)
        alpha = 2 * (abs(lambda0) - abs(lambda1)) / (abs(lambda0) + 1 / t);
    Base::m_a = ( alpha < 1 ) ? alpha : Scalar(1);
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigenVec = eig.eigenvectors();

    Base::m_uq = eigenVec.col(higher) * eigenVec.col(higher).transpose();

    VectorType v1 = VectorType(0, eigenVec.col(0)(0), eigenVec.col(0)(1));
    VectorType v2 = VectorType(0, eigenVec.col(1)(0), eigenVec.col(1)(1));
    Base::m_v1 = Base::template localFrameToWorld<true>(v1);
    Base::m_v2 = Base::template localFrameToWorld<true>(v2);

}

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_a_parabolic_fitting () {

    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    Eigen::Matrix<Scalar, 4, 1> uq (Base::m_uq(0,0), Base::m_uq(1,0), Base::m_uq(0,1), Base::m_uq(1,1));
    Eigen::Matrix<Scalar, 3, 1> uc_ul (Base::m_uc, Base::m_ul(0), Base::m_ul(1));

    Eigen::Matrix<Scalar, 7, 1> A = m_A_cov.block(0, 3, 7, 4) * uq;

    Eigen::Matrix<Scalar, 7, 1> F = m_F_cov - m_A_cov.block(0, 0, 7, 3) * uc_ul;
    
    const Eigen::Matrix<Scalar, 1, 1> x = (A).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(F);
    Base::m_a *= x(0,0);
}

template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_uc_ul_parabolic_fitting () {
    
    Eigen::Matrix<Scalar, 4, 1> uq (Base::m_uq(0,0), Base::m_uq(1,0), Base::m_uq(0,1), Base::m_uq(1,1));
    Eigen::Matrix<Scalar, 7, 1> F = m_F_cov - ( m_A_cov.block(0, 3, 7, 4) * uq * Base::m_a );

    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> x = (m_A_cov.block(0, 0, 7, 3)).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(F);

    Base::m_uc    = x(0,0);
    Base::m_ul(0) = x(1,0);
    Base::m_ul(1) = x(2,0);
}


template < class DataPoint, class _WFunctor, typename T>
void
ParabolicCylinderFitImpl<DataPoint, _WFunctor, T>::m_compute_curvature() {
    Scalar curv = - Scalar(2) * Base::m_a;

    if (curv <= 0) {
        Base::m_k1 = curv;
        Base::m_k2 = Scalar(0);
    }
    else {
        Base::m_k1 = Scalar(0);
        Base::m_k2 = curv;
    }
}
