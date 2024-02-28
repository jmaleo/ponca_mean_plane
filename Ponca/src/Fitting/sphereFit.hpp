/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


template < class DataPoint, class _WFunctor, typename T>
void
SphereFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);
    m_matA.setZero();
    m_minId = -1;
}

template < class DataPoint, class _WFunctor, typename T>
bool
SphereFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                     const VectorType &localQ,
                                                     const DataPoint &attributes)
{
    if( Base::addLocalNeighbor(w, localQ, attributes) ) {
        VectorA a;
#ifdef __CUDACC__
        a(0) = 1;
        a.template segment<DataPoint::Dim>(1) = localQ;
        a(DataPoint::Dim+1) = localQ.squaredNorm();
#else
        a << 1, localQ, localQ.squaredNorm();
#endif
        m_matA     += w * a * a.transpose();
        return true;
    }

    return false;
}


template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
SphereFitImpl<DataPoint, _WFunctor, T>::finalize ()
{
    // Compute status
    if(Base::finalize() != STABLE)
        return Base::m_eCurrentState;
    if(Base::getNumNeighbors() < DataPoint::Dim)
        return Base::m_eCurrentState = UNDEFINED;
    if (Base::algebraicSphere().isValid())
        Base::m_eCurrentState = CONFLICT_ERROR_FOUND;
    else
        Base::m_eCurrentState = Base::getNumNeighbors() < 2*DataPoint::Dim ? UNSTABLE : STABLE;

    MatrixA matC;
    matC.setIdentity();
    matC.template topRightCorner<1,1>()    << -2;
    matC.template bottomLeftCorner<1,1>()  << -2;
    matC.template topLeftCorner<1,1>()     << 0;
    matC.template bottomRightCorner<1,1>() << 0;

    MatrixA invCpratt;
    invCpratt.setIdentity();
    invCpratt.template topRightCorner<1,1>()    << -0.5;
    invCpratt.template bottomLeftCorner<1,1>()  << -0.5;
    invCpratt.template topLeftCorner<1,1>()     << 0;
    invCpratt.template bottomRightCorner<1,1>() << 0;

    // Remarks:
    //   A and C are symmetric so all eigenvalues and eigenvectors are real
    //   we look for the minimal positive eigenvalue (eigenvalues may be negative)
    //   C^{-1}A is not symmetric
    //   calling Eigen::GeneralizedEigenSolver on (A,C) and Eigen::EigenSolver on C^{-1}A is equivalent
    //   C is not positive definite so Eigen::GeneralizedSelfAdjointEigenSolver cannot be used
#ifdef __CUDACC__
    m_solver.computeDirect(invCpratt * m_matA);
#else
    m_solver.compute(invCpratt * m_matA);
#endif
    VectorA eivals = m_solver.eigenvalues().real();
    m_minId = -1;
    for(int i=0 ; i<DataPoint::Dim+2 ; ++i)
    {
    Scalar ev = eivals(i);
    if((ev>0) && (m_minId==-1 || ev<eivals(m_minId)))
    m_minId = i;
    }

    //mLambda = eivals(m_minId);
    VectorA vecU = m_solver.eigenvectors().col(m_minId).real();
    Base::m_uq = vecU[1+DataPoint::Dim];
    Base::m_ul = vecU.template segment<DataPoint::Dim>(1);
    Base::m_uc = vecU[0];

    Base::m_isNormalized = false;

    return Base::m_eCurrentState;
}

// //////////////////////////
// // Derivative TEMP
// //////////////////////////

template < class DataPoint, class _WFunctor, int DiffType, typename T>
void
SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);
    for(int dim = 0; dim < Base::NbDerivatives; ++dim)
        m_dmatA[dim] = MatrixA::Zero();
    // m_dSumDotPP = ScalarArray::Zero();
    m_dUc = ScalarArray::Zero();
    m_dUl = VectorArray::Zero();
    m_dUq = ScalarArray::Zero();
}

template < class DataPoint, class _WFunctor, int DiffType, typename T>
bool
SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::addLocalNeighbor(
    Scalar w,
    const VectorType &localQ,
    const DataPoint &attributes,
    ScalarArray& dw)
{
    if( Base::addLocalNeighbor(w, localQ, attributes, dw) )
    {
        VectorA basis;
        basis << 1, localQ, localQ.squaredNorm();
        const MatrixA prod = basis * basis.transpose();

        for(int dim = 0; dim < Base::NbDerivatives; ++dim)
        {
            m_dmatA[dim] += dw[dim] * prod;
        }

        return true;
    }
    return false;
}

template < class DataPoint, class _WFunctor, int DiffType, typename T>
FIT_RESULT
SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::finalize()
{
    constexpr int Dim = DataPoint::Dim;
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();

    Base::finalize();
    if(this->isReady())
    {
        int i = Base::m_minId;
        const Scalar eigenval_i = Base::m_solver.eigenvalues().real()[i];
        const VectorA eigenvec_i = Base::m_solver.eigenvectors().real().col(i);

        MatrixA matC;
        matC.setIdentity();
        matC.template topRightCorner<1,1>()    << -2;
        matC.template bottomLeftCorner<1,1>()  << -2;
        matC.template topLeftCorner<1,1>()     << 0;
        matC.template bottomRightCorner<1,1>() << 0;

        MatrixA invCpratt;
        invCpratt.setIdentity();
        invCpratt.template topRightCorner<1,1>()    << -0.5;
        invCpratt.template bottomLeftCorner<1,1>()  << -0.5;
        invCpratt.template topLeftCorner<1,1>()     << 0;
        invCpratt.template bottomRightCorner<1,1>() << 0;

        const Scalar invSumW = Scalar(1) / Base::getWeightSum();

        MatrixA deigvec = MatrixA::Zero();
        for(int j = 0; j < Dim+2; ++j)
        {
            if(j != i)
            {
                const Scalar eigenval_j = Base::m_solver.eigenvalues().real()[j];
                const Scalar eigengap = eigenval_i - eigenval_j; // positive since eigenval_i is the maximal eigenvalue

                if(eigengap > epsilon)
                {
                    const VectorA eigenvec_j = Base::m_solver.eigenvectors().real().col(j);
                    deigvec += ( Scalar(1)/eigengap ) * eigenvec_j * eigenvec_j.transpose();
                }
            }
        }

        for(int dim = 0; dim < Base::NbDerivatives; ++dim)
        {
            m_dmatA[dim] = invCpratt * m_dmatA[dim];

            VectorA vec = m_dmatA[dim] * eigenvec_i;


            vec = deigvec * vec;

            m_dUc[dim] = vec[0];
            m_dUq[dim] = vec[1+Dim];

            // For m_dUl[dim], we need to extract the vector part of vec (1 to 1+Dim)
            m_dUl.col(dim) = vec.template segment<Dim>(1);

        }

        // VectorA vecU = m_solver.eigenvectors().col(m_minId).real();
        // Base::m_uq = vecU[1+DataPoint::Dim];
        // Base::m_ul = vecU.template segment<DataPoint::Dim>(1);
        // Base::m_uc = vecU[0];

        return FIT_RESULT::STABLE;
    }
    return Base::m_eCurrentState;
}

template < class DataPoint, class _WFunctor, int DiffType, typename T>
typename SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::ScalarArray
SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::dPotential() const
{
    // same as OrientedSphereDerImpl::dPotential()
    ScalarArray dfield = m_dUc;
    if(Base::isSpaceDer())
        dfield.template tail<DataPoint::Dim>() += Base::m_ul;
    return dfield;
}

template < class DataPoint, class _WFunctor, int DiffType, typename T>
typename SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::VectorArray
SphereFitDerImpl<DataPoint, _WFunctor, DiffType, T>::dNormal() const
{
    // same as OrientedSphereDerImpl::dNormal()
    VectorArray dgrad = m_dUl;
    if(Base::isSpaceDer())
        dgrad.template rightCols<DataPoint::Dim>().diagonal().array() += Scalar(2)*Base::m_uq;
    Scalar norm  = Base::m_ul.norm();
    Scalar norm3 = norm*norm*norm;
    return dgrad / norm - Base::m_ul * (Base::m_ul.transpose() * dgrad) / norm3;
}
