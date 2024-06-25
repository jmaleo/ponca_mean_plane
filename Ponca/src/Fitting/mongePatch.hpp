﻿
#include <Eigen/SVD>
#include <Eigen/Geometry>

template < class DataPoint, class _WFunctor, typename T>
void
MongePatch<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);

    m_b.setZero();
    m_x.setZero();
    m_planeIsReady = false;
}

template < class DataPoint, class _WFunctor, typename T>
bool
MongePatch<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                      const VectorType &localQ,
                                                      const DataPoint &attributes)
{
    auto res = Base::addLocalNeighbor(w, localQ, attributes);
    if(! m_planeIsReady)
    {
        return res;
    }
    else // base plane is ready, we can now fit the patch
    {
        // express neighbor in local coordinate frame
        const VectorType local = Base::worldToLocalFrame(attributes.pos());
        const Scalar& h = *(local.data());
        const Scalar& u = *(local.data()+1);
        const Scalar& v = *(local.data()+2);

        Eigen::Matrix<Scalar, 6, 1 > p;
        p << u*u, v*v, u*v, u, v, 1;
        m_A    += w*p*p.transpose();
        m_b    += w*h*p;

        return true;
    }
    return false;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
MongePatch<DataPoint, _WFunctor, T>::finalize ()
{
    // end of the fitting process, check plane is ready
    if (! m_planeIsReady) {
        FIT_RESULT res = Base::finalize();

        if(res == STABLE) {  // plane is ready
            m_planeIsReady = true;
            m_A = SampleMatrix(6,6);
            m_A.setZero();
            m_b.setZero();

            return Base::m_eCurrentState = NEED_OTHER_PASS;
        }
        return res;
    }
    // end of the monge patch fitting process
    else {
        // we use BDCSVD as the matrix size is 36
        // http://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html
        m_x = m_A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(m_b);
        return Base::m_eCurrentState = STABLE;
    }
}

template < class DataPoint, class _WFunctor, typename T>
typename MongePatch<DataPoint, _WFunctor, T>::Scalar
MongePatch<DataPoint, _WFunctor, T>::kMean() const {
  PONCA_MULTIARCH_STD_MATH(pow);
  static const Scalar one (1);
  static const Scalar two (2);
  static const Scalar threeOverTwo (Scalar(3)/Scalar(2));
  return ( h_uu() * ( one + pow( h_v(), two ) ) - h_u() * h_v() * h_uv() + h_vv() * ( one + pow( h_u(), two ) ) * h_vv() ) /
      ( pow(one +pow(h_u(),two) + pow(h_v(),two),threeOverTwo ) );
}

template < class DataPoint, class _WFunctor, typename T>
typename MongePatch<DataPoint, _WFunctor, T>::Scalar
MongePatch<DataPoint, _WFunctor, T>::GaussianCurvature() const {
    PONCA_MULTIARCH_STD_MATH(pow);
    static const Scalar one (1);
    static const Scalar two (2);
    static const Scalar four (4);
    return (4 * h_uu()*h_vv() - pow(h_uv(),two)) /
        pow((one + pow(h_u(),two) + pow(h_v(),two) ), two);
}

template < class DataPoint, class _WFunctor, typename T>
typename MongePatch<DataPoint, _WFunctor, T>::Scalar
MongePatch<DataPoint, _WFunctor, T>::kmin() const {

    // Eigen::Matrix<Scalar, 2, 2> hess;
    // hess << h_uu(), h_uv(), h_uv(), h_vv();

    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> es(-hess);
    // Scalar kmin = es.eigenvalues()(0);

    Scalar gauss = GaussianCurvature();
    Scalar mean = kMean();
    Scalar kMin_from_fundamentals = mean - sqrt(mean*mean - gauss);
    return kMin_from_fundamentals;
}

template < class DataPoint, class _WFunctor, typename T>
typename MongePatch<DataPoint, _WFunctor, T>::Scalar
MongePatch<DataPoint, _WFunctor, T>::kmax() const {

    // Eigen::Matrix<Scalar, 2, 2> hess;
    // hess << h_uu(), h_uv(), h_uv(), h_vv();

    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> es(-hess);
    // Scalar kmax = es.eigenvalues()(1);

    Scalar gauss = GaussianCurvature();
    Scalar mean = kMean();
    Scalar kMax_from_fundamentals = mean + sqrt(mean*mean - gauss);
    return kMax_from_fundamentals;
}

template < class DataPoint, class _WFunctor, typename T>
typename MongePatch<DataPoint, _WFunctor, T>::VectorType
MongePatch<DataPoint, _WFunctor, T>::kminDirection() const {

    Eigen::Matrix<Scalar, 2, 2> hess;
    hess << h_uu(), h_uv(), h_uv(), h_vv();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> es(-hess);
    Eigen::Vector<Scalar, 2> dir = es.eigenvectors().col(0);

    VectorType v1 = VectorType(0, dir(0), dir(1));
    return Base::template localFrameToWorld<true>(v1);
}

template < class DataPoint, class _WFunctor, typename T>
typename MongePatch<DataPoint, _WFunctor, T>::VectorType
MongePatch<DataPoint, _WFunctor, T>::kmaxDirection() const {

    Eigen::Matrix<Scalar, 2, 2> hess;
    hess << h_uu(), h_uv(), h_uv(), h_vv();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> es(-hess);

    Eigen::Vector<Scalar, 2> dir = es.eigenvectors().col(1);

    VectorType v2 = VectorType(0, dir(0), dir(1));
    return Base::template localFrameToWorld<true>(v2);
}