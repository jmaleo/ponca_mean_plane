#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    // Setup fitting internal values
    m_sumDotPN = Scalar(0.0);
    m_sumDotPP = Scalar(0.0);
    m_sumProdPP = MatrixType::Zero();
    m_sumProdPN = MatrixType::Zero();
    m_A = MatrixType::Zero();
}

template < class DataPoint, class _WFunctor, typename T>
bool
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    if( Base::addLocalNeighbor(w, localQ, attributes) )
    {   
        m_sumDotPN += w * attributes.normal().dot(localQ);
        m_sumDotPP += w * localQ.squaredNorm();
        m_sumProdPP += w * localQ * localQ.transpose();
        m_sumProdPN += w * localQ * attributes.normal().transpose();
        return true;
    }
    return false;
}


template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::finalize () {
        // handle specific configurations
    // With less than 3 neighbors the fitting is undefined
    if(Base::finalize() != STABLE || Base::getNumNeighbors() < 3)
    {
        return Base::m_eCurrentState;
    }

    m_fitting_process();    

    return Base::m_eCurrentState = STABLE;
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::m_fitting_process () {
    
    m_ellipsoid_fitting();

    // m_uq_parabolic_fitting();
    // m_a_parabolic_fitting();
    // m_uc_ul_parabolic_fitting();
    
    m_compute_curvature();

}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::m_ellipsoid_fitting () {
    const Scalar sumW = Base::getWeightSum();
    const Scalar invSumW = Scalar(1.)/sumW;

    m_A = 2 * ( sumW * m_sumProdPP -  Base::m_sumP * Base::m_sumP.transpose());
    MatrixType C =  sumW * m_sumProdPN - Base::m_sumP * Base::m_sumN.transpose();
    C = C + C.transpose().eval();

    Base::m_uq = internal::solve_symmetric_sylvester(m_A, C);
    Base::m_ul = invSumW * (Base::m_sumN - Scalar(2) * Base::m_uq * Base::m_sumP);
    Base::m_uc = - invSumW * ( Base::m_ul.dot(Base::m_sumP) + (m_sumProdPP * Base::m_uq).trace() );

    Base::m_a = Scalar(1);

}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::m_uq_parabolic_fitting() {
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(Base::m_uq);
    VectorType values = eig.eigenvalues();


    const Eigen::MatrixXd eigenVec = eig.eigenvectors();
    
    // 1 higher than 0
    const VectorType eigenVec0 = eigenVec.col(1);
    // 0 is the littlest
    const VectorType eigenVec1 = eigenVec.col(0);
    Eigen::Matrix<Scalar, 3, 2> eigenMat;
    eigenMat.col(0) = eigenVec0;
    eigenMat.col(1) = eigenVec1;
    Base::m_v1 = eigenVec0;
    Base::m_v2 = eigenVec1;
    
    Base::m_uq = eigenMat * eigenMat.transpose();

    // // Compute alpha to put away the ambiguity of the solution
    // Scalar t = Base::m_w.evalScale();
    // Scalar alpha = 1;

    // // Find a way to fix the ambiguity
    // if (abs(values(max)) + 1/t > abs(epsilon))
    //     alpha = 2 * (abs(values(max)) - abs(values(med))) / (abs(values(max)) + 1 / t);
    
    // Base::m_a = ( alpha < Scalar(1) ) ? alpha : Scalar(1);
    // Base::m_a = alpha;
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::m_a_parabolic_fitting () {
    
    const Scalar weight = Base::getWeightSum();
    const Scalar invSumW = Scalar(1.)/weight;

    // Matrix2 Q = Base::m_uq * Base::m_uq.transpose();                       // 3x3
    MatrixType Q_squared = Base::m_uq * Base::m_uq;                           // 3x3
    Scalar A = (m_sumProdPP.array() * Q_squared.array()).sum(); // 1
    VectorType B = Base::m_uq * Base::m_sumP;                                  // 3x3 * 3x1 = 3x1
    Scalar C = (m_sumProdPN.array() * Base::m_uq.array()).sum();         // 1

    Scalar first = invSumW * Base::m_sumN.transpose() * B;         // 1 * 1x3 * 3x1 = 1
    Scalar second = invSumW * B.transpose() * B;               // 1 * 1x3 * 3x1 = 1

    Scalar a = (C - first) / (4*A - second);                 // (1 - 1) / (1 * 1 - 1)
    Base::m_a = a;

}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::m_uc_ul_parabolic_fitting () {
    const Scalar weight = Base::getWeightSum();
    const Scalar invSumW = Scalar(1.)/weight;
    // Matrix2 Q = Base::m_uq * Base::m_uq.transpose();
    VectorType B = Base::m_uq * Base::m_sumP;
    Base::m_ul = invSumW * ( Base::m_sumN - Base::m_a * B);


    Scalar A = Base::m_ul.transpose() * Base::m_sumP;
    Scalar C = (m_sumProdPN.array() * Base::m_uq.array()).sum();
    Base::m_uc = - invSumW * (A + Base::m_a * C);
}

template < class DataPoint, class _WFunctor, typename T>
void
OrientedHyperboloidFitImpl<DataPoint, _WFunctor, T>::m_compute_curvature() {
    
    Scalar curv = Scalar(2) * Base::m_a;
    MatrixType H = Base::m_uq * curv;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> eig(H);
    VectorType values = eig.eigenvalues();

    // grab the 2 heighest eigenvalues

    Base::m_k1 = values(0);
    Base::m_v1 = eig.eigenvectors().col(0);
    Base::m_k2 = values(1);
    Base::m_v2 = eig.eigenvectors().col(1);


    if (Base::m_k1 >= Base::m_k2) {
        std::swap(Base::m_k1, Base::m_k2);
        std::swap(Base::m_v1, Base::m_v2);
    }

}
