
template<class DataPoint, class _WFunctor, typename T>
void 
VarifoldImpl<DataPoint, _WFunctor, T>::init(
    const VectorType& _evalPos)
{
    Base::init(_evalPos);
    m_sumN = VectorType::Zero();
    m_sumP = VectorType::Zero();
    m_sumWeight = Scalar(0);

    m_n_l0 = VectorType::Zero();

    m_nume = MatrixType::Zero();
    m_deno = Scalar(0);

    m_k1 = Scalar(0);
    m_dir1 = VectorType::Zero();
    m_k2 = Scalar(0);
    m_dir2 = VectorType::Zero();

    m_planeIsReady = false;
}

template<class DataPoint, class _WFunctor, typename T>
bool 
VarifoldImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(
    Scalar w, const VectorType &localQ, const DataPoint &attributes)
{
    auto res = true;
    if (! m_planeIsReady)
    {
        VectorType baryCenter = Base::m_w.basisCenter();
        VectorType pos = attributes.pos();
        VectorType centerPos = pos - baryCenter;
        Scalar dist = centerPos.norm();
        Scalar val = f_smooth(dist / Base::m_w.evalScale());

        res = Base::addLocalNeighbor(val, localQ, attributes);

        m_sumWeight += val;

        return res;
    }
    else {

        if(localQ.squaredNorm() < Eigen::NumTraits<Scalar>::dummy_precision())
            return false;

        const VectorType n_l = attributes.normal();
        const MatrixType P_l = MatrixType::Identity() - n_l * n_l.transpose();
        const Scalar d_l = localQ.norm();
        const VectorType delta_x = localQ / d_l; // = localQ.normalized()
        const VectorType u = P_l * delta_x;
        const MatrixType DeltaP = P_l - m_P_l0;
        const VectorType Pl_nl0 = P_l * m_n_l0;

        const MatrixType a1 = u * Pl_nl0.transpose();
        const MatrixType a2 = u.dot(m_n_l0) * DeltaP;

        w *= -1; // see WARNING in VarifoldWeightKernel::f() (nor necessary because of the ratio nume/deno)

        m_nume += w * (a1 + a1.transpose() - a2);
        m_deno += w * d_l;
    }

    return true;
}

template<class DataPoint, class _WFunctor, typename T>
FIT_RESULT 
VarifoldImpl<DataPoint, _WFunctor, T>::finalize()
{

    if ( ! m_planeIsReady ){
        FIT_RESULT res = Base::finalize();
        m_planeIsReady = true;
        m_sumP = Base::barycenter();
        reorientPlane();
        
        return Base::m_eCurrentState = NEED_OTHER_PASS;
    }
    else {

        constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

        if(std::abs(m_deno) < epsilon)
            return UNSTABLE;

        const MatrixType A = m_nume / m_deno * Base::m_w.evalScale(); /* const MatrixType A = m_nume / m_deno * Base::m_w.evalScale(); */

        const Mat32 Q = tangentPlane();
        const Mat22 B = - Q.transpose() * A * Q; // minus for sign convention

        Eigen::SelfAdjointEigenSolver<Mat22> solver;
        solver.computeDirect(B);

        if(solver.info() != Eigen::Success)
            return UNSTABLE;

        m_k1 = solver.eigenvalues()[0] / Base::m_w.evalScale();
        m_dir1 = Q * solver.eigenvectors().col(0);
        m_k2 = solver.eigenvalues()[1] / Base::m_w.evalScale();
        m_dir2 = Q * solver.eigenvectors().col(1);

        if (m_k1 > m_k2)
        {
            std::swap(m_k1, m_k2);
            std::swap(m_dir1, m_dir2);
        }
    }

    return STABLE;
}

template<class DataPoint, class _WFunctor, typename T>
typename VarifoldImpl<DataPoint, _WFunctor, T>::Mat32 
VarifoldImpl<DataPoint, _WFunctor, T>::tangentPlane() const
{
    Mat32 B;
    B.col(0) = Base::getFrameU();
    B.col(1) = Base::getFrameV();
    return B;
}

template<class DataPoint, class _WFunctor, typename T>
void
VarifoldImpl<DataPoint, _WFunctor, T>::reorientPlane () {
    VectorType planeNormal = Base::primitiveGradient();
    VectorType meanNormal = Base::meanNormalVector();
    if (planeNormal.dot(meanNormal) < 0) {
        m_n_l0 = -planeNormal;
    }
    else {
        m_n_l0 = planeNormal;
    }
    m_P_l0 = MatrixType::Identity() - m_n_l0 * m_n_l0.transpose();
}


template<class DataPoint, class _WFunctor, typename T>
typename VarifoldImpl<DataPoint, _WFunctor, T>::VectorType 
VarifoldImpl<DataPoint, _WFunctor, T>::project(const VectorType& p) const
{
    return p - (p - m_sumP).dot(m_n_l0) * m_n_l0;
}

template<class DataPoint, class _WFunctor, typename T>
typename VarifoldImpl<DataPoint, _WFunctor, T>::VectorType 
VarifoldImpl<DataPoint, _WFunctor, T>::primitiveGradient(const VectorType&) const
{
    return m_n_l0;
}

