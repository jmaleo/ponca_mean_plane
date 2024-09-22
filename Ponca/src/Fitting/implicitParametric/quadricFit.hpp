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

    auto res = constructTensor2();
    // VectorType x = Base::project( Base::m_w.basisCenter() );
    VectorType x = Base::m_w.convertToLocalBasis( Base::m_w.basisCenter() ); // 0

    // std::cout << "============" << std::endl;
    // std::cout << "Potential : " << Base::potential(x) << std::endl;
    // std::cout << "Algebraic parameters : " << Base::m_coefficients << std::endl;

    Matrix32 P = tangentPlane( Base::primitiveGradient(x) );

    Matrix2 I; 
    I << Base::dE(x), Base::dF(x),
         Base::dF(x), Base::dG(x);
    
    Matrix2 II; 
    II << Base::dL(x), Base::dM(x),
         Base::dM(x), Base::dN(x);


    Matrix2 W = I.inverse() * II;

    // Solve the eigenvalue problem
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 2, 2>> eigW( W );


    Vector2 eivals = eigW.eigenvalues().real();
    Matrix2 eivecs = eigW.eigenvectors().real();
    // Base::m_kmin = eivals(0);
    // Base::m_kmax = eivals(1);
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

template < class DataPoint, class _WFunctor, typename T>
FIT_RESULT
QuadricFitImpl<DataPoint, _WFunctor, T>::constructTensor2()
{

    VectorType x = Base::project( Base::m_w.basisCenter() );
    Scalar f_x, f_y, f_z;
    f_x = Base::f_x(x);
    f_y = Base::f_y(x);
    f_z = Base::f_z(x);
    Scalar f_x2 = f_x * f_x;
    Scalar f_y2 = f_y * f_y;
    Scalar f_z2 = f_z * f_z;
    Scalar two = Scalar(2);

    Scalar a = ( Base::f_xx() * (- Base::f_yy() * f_z2 + two * Base::f_yz() * f_y * f_z - Base::f_zz() * f_y2)
            + Base::f_xy() * ( 
                            Base::f_xy() * f_z2 
                            + Base::f_yz() * ( -f_x*f_z - f_y*f_z - two * f_x*f_y - f_x*f_z ) 
                            - Base::f_xz() * f_y * f_z + two * Base::f_zz()*f_x*f_y + two * Base::f_yy()*f_x*f_z + f_y2
                            + Base::f_yz() * Base::f_yz() * f_x - Base::f_yy() * Base::f_zz() * f_x2 
                )
        );
    
    Scalar b = ( 
          Base::f_xx() * ( f_z2 + f_y2 ) 
        + Base::f_yy() * ( f_x2 + f_z2 )
        + Base::f_zz() * ( f_x2 + f_y2 )
        - two * Base::f_xy() * f_x * f_y
        - two * Base::f_yz() * f_y * f_z
        - Base::f_xz() * f_x * f_z
        - Base::f_xy() * f_x * f_z
    );

    Scalar c = (
        - f_x2 - f_y2 - f_z2
    );

    Scalar K = ( a / c ) / ( f_x2 + f_y2 + f_z2 );
    Scalar H = Scalar(-1) * ( b / c ) / ( two * std::sqrt( f_x2 + f_y2 + f_z2 ) );
                
    Base::m_kmin = std::abs(H - std::sqrt( H*H - K ));
    Base::m_kmax = std::abs(H + std::sqrt( H*H - K ));
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