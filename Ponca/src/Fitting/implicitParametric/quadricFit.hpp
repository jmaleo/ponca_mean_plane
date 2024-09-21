template < class DataPoint, class _WFunctor, typename T>
void
QuadricFitImpl<DataPoint, _WFunctor, T>::init(const VectorType& _evalPos)
{
    Base::init(_evalPos);
    m_matA = MatrixA::Zero();
    m_minId = -1;
}

template < class DataPoint, class _WFunctor, typename T>
bool
QuadricFitImpl<DataPoint, _WFunctor, T>::addLocalNeighbor(Scalar w,
                                                     const VectorType &localQ,
                                                     const DataPoint &attributes)
{
    if( Base::addLocalNeighbor(w, localQ, attributes) ) {
        VectorA a = Base::convertToParameters(localQ);
        m_matA     += w * a * a.transpose();
        return true;
    }

    return false;
}


template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
QuadricFitImpl<DataPoint, _WFunctor, T>::finalize ()
{
    // Compute status
    FIT_RESULT res = Base::finalize();

    if( res != STABLE )
        return Base::m_eCurrentState;

    if( Base::getNumNeighbors() < 9 )
        return Base::m_eCurrentState = UNDEFINED;
    
    m_solver.compute( m_matA );

    VectorA eivals = m_solver.eigenvalues().real();

    m_minId = 0;

    for(int i=0 ; i < eivals.size() ; ++i)
    {
        Scalar ev = eivals(i);
        if( ev < eivals(m_minId) )
            m_minId = i;
    }

    Base::m_coefficients = m_solver.eigenvectors().col(m_minId).real();
    Base::computeFrameFromNormalVector( Base::primitiveGradient() );

    FIT_RESULT solverRes = constructTensor();
    if ( solverRes != STABLE )
        return res = solverRes;

    return res = Base::m_eCurrentState;
}

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
QuadricFitImpl<DataPoint, _WFunctor, T>::constructTensor()
{

    VectorType evalProj = Base::project( Base::m_w.basisCenter() );

    // std::cout << "============" << std::endl;
    // std::cout << "Potential : " << Base::potential(evalProj) << std::endl;
    // std::cout << "Algebraic parameters : " << Base::m_coefficients << std::endl;

    Matrix32 P = tangentPlane( Base::primitiveGradient(evalProj) );

    Matrix2 I; 
    I << Base::dE(evalProj), Base::dF(evalProj),
         Base::dF(evalProj), Base::dG(evalProj);
    
    Matrix2 II; 
    II << Base::dL(evalProj), Base::dM(evalProj),
         Base::dM(evalProj), Base::dN(evalProj);


    Matrix2 W = I.inverse() * II;

    // Solve the eigenvalue problem
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eigW( W );


    Vector2 eivals = eigW.eigenvalues().real();
    Matrix2 eivecs = eigW.eigenvectors().real();

    Base::m_kmin = eivals(0);
    Base::m_kmax = eivals(1);
    Base::m_dmin = P * eivecs.col(0);
    Base::m_dmax = P * eivecs.col(1);
    // Base::m_dmin = eivecs.col(0);
    // Base::m_dmax = eivecs.col(1);

    if (std::abs( Base::m_kmin ) > std::abs( Base::m_kmax ))
    {
        std::swap(Base::m_kmin, Base::m_kmax);
        std::swap(Base::m_dmin, Base::m_dmax);
    }

    return STABLE;


}

template<class DataPoint, class _WFunctor, typename T>
typename QuadricFitImpl<DataPoint, _WFunctor, T>::Matrix32 
QuadricFitImpl<DataPoint, _WFunctor, T>::tangentPlane(const VectorType& n)
{
    Matrix32 B;
    // int i0=-1, i1=-1, i2=-1;
    // n.array().abs().minCoeff(&i0); // i0: dimension where n extends the least
    // i1 = (i0+1)%3;
    // i2 = (i0+2)%3;

    // B.col(0)[i0] = 0;
    // B.col(0)[i1] = n[i2];
    // B.col(0)[i2] = -n[i1];

    // B.col(0).normalize();
    // B.col(1) = B.col(0).cross(n);
    B.col(0) = Base::getFrameU();
    B.col(1) = Base::getFrameV();
    return B;
}