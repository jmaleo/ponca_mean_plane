template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Scalar
ParabolicCylinder<DataPoint, _WFunctor, T>::potential( const VectorType &_q ) const
{
    VectorType x = Base::worldToLocalFrame(_q);
    return  eval_quadratic_function(*(x.data() +1 ), *(x.data() + 2));
}


template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::VectorType
ParabolicCylinder<DataPoint, _WFunctor, T>::project( const VectorType& _q ) const
{
    VectorType x = Base::worldToLocalFrame(_q);
    return Base::project(_q) + ( eval_quadratic_function(*(x.data()+1), *(x.data()+2)) * Base::primitiveGradient(_q));
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
    local_gradient.normalize();

    VectorType world_gradient = Base::template localFrameToWorld<true>(local_gradient);
    world_gradient.normalize();

    return m_correctOrientation * world_gradient;
}

template < class DataPoint, class _WFunctor, typename T>
typename ParabolicCylinder<DataPoint, _WFunctor, T>::Matrix2
ParabolicCylinder<DataPoint, _WFunctor, T>::dPrimitiveGradient( ) const
{
    // Matrix2 dH = 2 * m_a * (m_uq * m_uq.transpose());
    Matrix2 dH = 2 * m_a * m_uq;
    return dH;
}
