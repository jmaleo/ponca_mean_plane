
#include <Eigen/SVD>
#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
NormalCovariance3D<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_cov.setZero();
    m_normal_centroid.setZero();
    m_planeIsReady = false;
}

template < class DataPoint, class _WFunctor, typename T>
bool
NormalCovariance3D<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    bool res = Base::addLocalNeighbor(w, localQ, attributes);
    if( ! m_planeIsReady ) {
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
    if (! m_planeIsReady) {
        FIT_RESULT res = Base::finalize();

        if(res == STABLE) {  // plane is ready
            m_planeIsReady = true;
            return Base::m_eCurrentState = NEED_OTHER_PASS;
        }
        return res;
    }
    else {
        m_normal_centroid /= Base::getWeightSum();
        m_cov = ( m_cov / Base::getWeightSum() ) - ( m_normal_centroid * m_normal_centroid.transpose() );
        // symmetrize the covariance matrix
        m_cov = (m_cov + m_cov.transpose()) / Scalar(2);
        
        m_solver.compute(m_cov);

        if (m_solver.info() != Eigen::Success) {
            return Base::m_eCurrentState = UNDEFINED;
        }
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
    return m_solver.eigenvalues()(1);
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::Scalar
NormalCovariance3D<DataPoint, _WFunctor, T>::kmax() const {
    return m_solver.eigenvalues()(2);
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::VectorType
NormalCovariance3D<DataPoint, _WFunctor, T>::kminDirection() const {
    VectorType kMinDirection = m_solver.eigenvectors().col(1);
    return kMinDirection;
}

template < class DataPoint, class _WFunctor, typename T>
typename NormalCovariance3D<DataPoint, _WFunctor, T>::VectorType
NormalCovariance3D<DataPoint, _WFunctor, T>::kmaxDirection() const {
    VectorType kMaxDirection = m_solver.eigenvectors().col(2);
    return kMaxDirection;
}