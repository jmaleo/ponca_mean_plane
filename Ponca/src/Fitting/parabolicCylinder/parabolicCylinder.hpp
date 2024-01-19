template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{
    VectorType x = Base::worldToLocalFrame(_q);
    return m_correctOrientation * eval_quadratic_function(*(x.data() +1 ), *(x.data() + 2)) - *(x.data());
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{
    PONCA_MULTIARCH_STD_MATH(abs);
    constexpr Scalar epsilon = Eigen::NumTraits<Scalar>::dummy_precision();
    VectorType x = Base::worldToLocalFrame(_q);
    x(0) = m_correctOrientation * eval_quadratic_function(x(1), x(2));
    return Base::localFrameToWorld(x);
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::primitiveGradient( const VectorType& _q ) const
{
    // Convexe = m_a >= 0    Concave = m_a <= 0

    VectorType proj = Base::worldToLocalFrame(_q);
    Vector2 temp {proj(1),  proj(2)};
    Vector2 df = m_ul + 2 * m_a * m_uq * temp;
    VectorType local_gradient { 1, -df(0) , -df(1) };
    local_gradient *= m_correctOrientation;

    VectorType world_gradient = Base::template localFrameToWorld<true>(local_gradient);

    return world_gradient;
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::MatrixType
ParabolicCylinder<DataPoint, _WFunctor, T>::dNormal() const
{ 
    // Grab the normal (primitiv gradient)
    VectorType u = Base::getFrameU();
    VectorType v = Base::getFrameV();
    VectorType n = primitiveGradient();

    MatrixType B;
    B << n, u, v;

    Matrix2 dN_2D = (2 * m_a * m_uq);
    MatrixType dN = MatrixType::Zero();
    dN.block(1,1,2,2) = dN_2D;
    // put 1 on the first row and column
    dN(0,0) = 1; 

    VectorType ul {1, m_ul(0), m_ul(1)};
    
    return ( B * dN * B.transpose() ) / Base::template localFrameToWorld<true>(ul).norm(); 

}