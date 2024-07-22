
#include <Eigen/SVD>
#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
NormalCovariance3D<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_cov.setZero();
    m_normal_centroid.setZero();
    m_minDirIndex = 0;
    m_maxDirIndex = 0;
}

// findNormalDirection
template < class DataPoint, class _WFunctor, typename T>
void
NormalCovariance3D<DataPoint, _WFunctor, T>::findMinDirection(){

    // normal
    VectorType n = Base::normal();

    // find the smallest angle between the normal and the 2 firsts eigenvectors
    Scalar minAngle = Scalar(2) * Scalar(M_PI);
    Scalar maxAngle = Scalar(0);
    int minIndex = -1;
    int maxIndex = -1;
 
    for(int i = 0; i < 3; i++){
        Scalar dot_prod = m_solver.eigenvectors().col(i).dot ( n );
        if ( dot_prod < 0. ){
            dot_prod = m_solver.eigenvectors().col(i).dot ( -n );
        }
        dot_prod = ( dot_prod > 1. ) ? 1 : dot_prod;
        dot_prod = ( dot_prod < -1. ) ? 1 : dot_prod;

        Scalar angle = std::acos( dot_prod / ( m_solver.eigenvectors().col(i).norm() * n.norm() ) );
        if(angle <= minAngle){
            minAngle = angle;
            minIndex = i;
        }
        if(angle >= maxAngle){
            maxAngle = angle;
            maxIndex = i;
        }
    }
    m_minDirIndex = 3 - minIndex - maxIndex;
    m_maxDirIndex = maxIndex;
}

template < class DataPoint, class _WFunctor, typename T>
bool
NormalCovariance3D<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    bool res = Base::addLocalNeighbor(w, localQ, attributes);
    if( ! res ){
        return res;
    }
    else {
        m_cov  += w * attributes.normal() * attributes.normal().transpose();
        m_normal_centroid += w * attributes.normal();
        return true;
    }
    return false;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
NormalCovariance3D<DataPoint, _WFunctor, T>::finalize ()
{
    FIT_RESULT res = Base::finalize();
    if ( ! res == FIT_RESULT::STABLE ) {
        return res;
    }
    else {
        m_normal_centroid /= Base::getWeightSum();
        m_cov = ( m_cov / Base::getWeightSum() ) - ( m_normal_centroid * m_normal_centroid.transpose() );
        // m_cov = (m_cov + m_cov.transpose()) / Scalar(2);
        
        m_solver.compute(m_cov);

        if (m_solver.info() != Eigen::Success) {
            return Base::m_eCurrentState = UNDEFINED;
        }

        findMinDirection();
        
        return Base::m_eCurrentState = STABLE;
    }
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kMean() const {
  return ( kmin() + kmax() ) / Scalar(2);
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::GaussianCurvature() const {
    return kmin() * kmax();
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kmin() const {
    PONCA_MULTIARCH_STD_MATH(pow);
    PONCA_MULTIARCH_STD_MATH(sqrt);
    constexpr Scalar four = Scalar(4);
    constexpr Scalar PI = Scalar(M_PI);
    Scalar factor = pow ( Base::m_w.evalScale(), four ) * PI / four;
    return sqrt( m_solver.eigenvalues()(m_minDirIndex) / factor );
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kmax() const {
    PONCA_MULTIARCH_STD_MATH(pow);
    PONCA_MULTIARCH_STD_MATH(sqrt);
    constexpr Scalar four = Scalar(4);
    constexpr Scalar PI = Scalar(M_PI);
    Scalar factor = pow ( Base::m_w.evalScale(), four ) * PI / four;
    return sqrt( m_solver.eigenvalues()(m_maxDirIndex) / factor );
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::VectorType
NormalCovariance3D<DataPoint, _WFunctor, T>::kminDirection() const {
    VectorType kMinDirection = m_solver.eigenvectors().col(m_minDirIndex);
    return kMinDirection;
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::VectorType
NormalCovariance3D<DataPoint, _WFunctor, T>::kmaxDirection() const {
    VectorType kMaxDirection = m_solver.eigenvectors().col(m_maxDirIndex);
    return kMaxDirection;
}